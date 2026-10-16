"""The Bluesky side: poll notifications, reply to mentions, replies and quotes."""

import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from atproto import Client, SessionEvent, models
from atproto_client.exceptions import BadRequestError, LoginRequiredError, UnauthorizedError

from .brain import Brain, Context
from .config import Settings
from .store import Store

log = logging.getLogger(__name__)

REASONS = ("mention", "reply", "quote")
THREAD_PARENTS = 6  # how many posts up the thread to show the model
PAGE_SIZE = 50
MAX_SWEEP_PAGES = 20  # a hard stop, in case a cursor never runs out
RELOGIN_MIN_SECONDS = 300  # Bluesky allows 30 logins per 5 minutes, 300 per day
_AUTH_ERRORS = {"ExpiredToken", "InvalidToken", "AuthenticationRequired", "AuthMissing"}


def login(settings: Settings, *, reuse_saved: bool = True, save: bool = True) -> Client:
    """Log in to Bluesky.

    The saved session in state/session.txt belongs to the long-running bot.
    Refreshing a session rotates its refresh token and invalidates the old one,
    so one-off commands must not touch it: they pass reuse_saved=False,
    save=False and get a session of their own.
    """
    settings.require_login()
    client = Client()
    session_file = settings.state_dir / "session.txt"
    settings.state_dir.mkdir(parents=True, exist_ok=True)

    if save:
        def save_session(event: SessionEvent, session) -> None:
            if event in (SessionEvent.CREATE, SessionEvent.REFRESH):
                # Private from the moment it exists, not just after the tokens are in it.
                fd = os.open(session_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                os.fchmod(fd, 0o600)  # an older file may have looser permissions
                with os.fdopen(fd, "w") as f:
                    f.write(client.export_session_string())

        client.on_session_change(save_session)
    # Reuse the saved session: Bluesky rate-limits fresh logins hard.
    if reuse_saved and session_file.exists():
        try:
            client.login(session_string=session_file.read_text())
            if client.me.handle == settings.handle:
                return client
        except Exception as e:  # expired or revoked
            log.info("saved session unusable (%s); logging in again", e)
    client.login(settings.handle, settings.app_password)
    return client


def is_auth_error(e: Exception) -> bool:
    """True when the session is dead (expired, revoked, or its refresh token was rotated away)."""
    if isinstance(e, (UnauthorizedError, LoginRequiredError)):
        return True
    if isinstance(e, BadRequestError) and e.response is not None:
        return getattr(e.response.content, "error", None) in _AUTH_ERRORS
    return False


def _is_bot(author) -> bool:
    return any(label.val == "bot" for label in (author.labels or []))


def _did_of(uri: str) -> str:
    return uri.split("/")[2]


def _when(item) -> datetime:
    return datetime.fromisoformat(item.indexed_at.replace("Z", "+00:00"))


class Bot:
    def __init__(self, settings: Settings, brain: Brain, dry_run: bool = False):
        self.settings = settings
        self.brain = brain
        self.dry_run = dry_run
        self.store = Store(settings.state_dir / "jarbot.db")
        self._dry_seen: set[str] = set()  # a dry run records nothing in the store
        # A dry run often runs alongside the live bot, so it leaves the saved session alone.
        self.client = login(settings, reuse_saved=not dry_run, save=not dry_run)
        self._last_login = time.monotonic()
        self._last_sweep: float | None = None  # None: sweep on the first poll
        self.me = self.client.me
        mention = r"@" + re.escape(self.me.handle) + r"\b"
        self._leading_mention = re.compile(r"^(?:\s*" + mention + r"[\s,:]*)+", re.IGNORECASE)
        self._trailing_mention = re.compile(r"(?:\s*" + mention + r")+[\s.!?]*$", re.IGNORECASE)
        self._mention_rx = re.compile(mention, re.IGNORECASE)
        log.info("logged in as %s (%s)%s", self.me.handle, self.me.did, " [dry run]" if dry_run else "")

    def run_forever(self) -> None:
        while True:
            self.step()
            time.sleep(self.settings.poll_seconds)

    def step(self) -> None:
        """One poll, surviving errors; logs in again if the session has died."""
        try:
            self.poll_once()
        except Exception as e:
            if not is_auth_error(e):
                log.exception("poll failed; will retry")
                return
            if time.monotonic() - self._last_login < RELOGIN_MIN_SECONDS:
                log.warning("session rejected (%s); logged in recently, so waiting before trying again", e)
                return
            log.warning("session rejected (%s); logging in again", e)
            self._last_login = time.monotonic()
            try:
                self.client = login(self.settings, reuse_saved=False, save=not self.dry_run)
            except Exception:
                log.exception("login failed; will retry")

    def poll_once(self) -> None:
        """Answer new mentions, replies and quotes.

        A normal poll reads one page of notifications. A sweep, on startup and
        every JAR_SWEEP_MINUTES, also looks back JAR_MAX_AGE_HOURS: it pages
        through older notifications (to catch up after downtime) and searches
        for "@handle" in post text (posts that name Jar! without a proper
        mention link never produce a notification at all).
        """
        now = time.monotonic()
        sweep = self._last_sweep is None or now - self._last_sweep >= self.settings.sweep_minutes * 60
        items = {n.uri: n for n in self._notifications(deep=sweep)}
        if sweep:
            self._last_sweep = now
            for post in self._text_mentions():
                items.setdefault(post.uri, post)  # the notification, if any, knows the real reason
        fresh = [n for n in items.values() if not self.store.seen(n.uri) and n.uri not in self._dry_seen]
        for notif in sorted(fresh, key=_when):  # oldest first
            if self.dry_run:
                self._dry_seen.add(notif.uri)
            try:
                self.handle(notif)
            except Exception as e:
                if is_auth_error(e):
                    raise
                # Left unrecorded, so it is tried again next poll; one bad post
                # must not hold up the ones after it.
                log.exception("could not handle %s", notif.uri)
        if fresh and not self.dry_run:
            self.client.app.bsky.notification.update_seen({"seen_at": self.client.get_current_time_iso()})

    def _cutoff(self) -> datetime:
        return datetime.now(timezone.utc) - timedelta(hours=self.settings.max_age_hours)

    def _notifications(self, deep: bool) -> list:
        """Mention/reply/quote notifications: one page, or (deep) back to the age cutoff."""
        out, cursor = [], None
        for _ in range(MAX_SWEEP_PAGES if deep else 1):
            params = {"limit": PAGE_SIZE, "reasons": list(REASONS)}
            if cursor:
                params["cursor"] = cursor
            resp = self.client.app.bsky.notification.list_notifications(params)
            out += [n for n in resp.notifications if n.reason in REASONS]
            cursor = resp.cursor
            if not cursor or not resp.notifications or _when(resp.notifications[-1]) < self._cutoff():
                break
        return out

    def _text_mentions(self) -> list:
        """Posts since the age cutoff whose text says "@handle", shaped like mention notifications."""
        needle = "@" + self.me.handle.lower()
        out, cursor = [], None
        try:
            for _ in range(MAX_SWEEP_PAGES):
                params = {"q": self.me.handle, "sort": "latest", "limit": 100,
                          "since": self._cutoff().isoformat().replace("+00:00", "Z")}
                if cursor:
                    params["cursor"] = cursor
                resp = self.client.app.bsky.feed.search_posts(params)
                for p in resp.posts:
                    if p.author.did != self.me.did and needle in (getattr(p.record, "text", "") or "").lower():
                        out.append(SimpleNamespace(uri=p.uri, cid=p.cid, author=p.author, record=p.record,
                                                   reason="mention", indexed_at=p.indexed_at))
                cursor = resp.cursor
                if not cursor or not resp.posts:
                    break
        except Exception as e:
            if is_auth_error(e):
                raise
            log.warning("search for text mentions failed: %s", e)  # notifications still work
        return out

    def _skip(self, notif, why: str) -> None:
        log.info("skip %s from @%s: %s", notif.uri, notif.author.handle, why)
        if not self.dry_run:
            self.store.record(notif.uri, notif.author.did, notif.reason, f"skipped:{why}")

    def handle(self, notif) -> None:
        s = self.settings
        record = notif.record
        if notif.author.did == self.me.did:
            return self._skip(notif, "own post")
        if _is_bot(notif.author) and not s.reply_to_bots:
            return self._skip(notif, "author is a bot")
        if _when(notif) < self._cutoff():
            return self._skip(notif, "too old")
        if notif.reason == "reply" and record.reply and _did_of(record.reply.parent.uri) != self.me.did:
            # Someone replied to someone else further down a thread Jar! started.
            return self._skip(notif, "reply not addressed to Jar!")
        if self.store.replies_since(3600, notif.author.did) >= s.max_per_author_per_hour:
            return self._skip(notif, "author rate limit")
        if self.store.replies_since(3600) >= s.max_per_hour:
            log.warning("global hourly cap reached; leaving %s for later", notif.uri)
            return  # don't record: try again next poll

        ctx, my_replies, replies_disabled = self.context(notif)
        if replies_disabled:
            return self._skip(notif, "replies disabled")
        if my_replies >= s.max_replies_per_thread:
            return self._skip(notif, "thread reply limit")
        if not ctx.text and not ctx.thread:
            return self._skip(notif, "empty post")

        t0 = time.monotonic()
        try:
            reply = self.brain.reply(ctx)
        except Exception as e:  # never retried: the same post would fail the same way
            log.exception("could not generate a reply to %s", notif.uri)
            if not self.dry_run:
                self.store.record(notif.uri, notif.author.did, notif.reason, f"error:generation:{e}", ctx.text)
            return
        log.info("@%s: %r\n  -> %r (%.1fs)", notif.author.handle, ctx.text, reply, time.monotonic() - t0)
        if self.dry_run:
            return

        parent = models.create_strong_ref(notif)
        root = record.reply.root if record.reply else parent
        try:
            sent = self.client.send_post(
                text=reply,
                reply_to=models.AppBskyFeedPost.ReplyRef(root=root, parent=parent),
                langs=["en"],
            )
        except Exception as e:  # deleted post, blocked, reply-gated thread, network...
            if is_auth_error(e):
                raise  # not this post's fault: log in again and retry it next poll
            # Recorded and never retried: a retry would run the model again for
            # the same post, without limit, and none of it counted toward the
            # rate caps. (A reply that might have gone through is not sent twice.)
            log.warning("could not reply to %s: %s", notif.uri, e)
            self.store.record(notif.uri, notif.author.did, notif.reason, f"error:{e}", ctx.text, reply)
            return
        self.store.record(notif.uri, notif.author.did, notif.reason, "replied", ctx.text, reply, sent.uri)
        # Like what Jar! answered. Only after the reply is recorded, and never
        # fatal: a failed like must not get the post answered twice.
        try:
            self.client.like(notif.uri, notif.cid)
        except Exception as e:
            log.warning("could not like %s: %s", notif.uri, e)

    def context(self, notif) -> tuple[Context, int, bool]:
        """The post's text and the thread above it; also how often Jar! already
        spoke there, and whether the thread's reply settings shut Jar! out."""
        text = self._clean(notif.record.text)
        thread: list[tuple[str, str]] = []
        my_replies = 0
        replies_disabled = False
        try:
            view = self.client.get_post_thread(notif.uri, depth=0, parent_height=50).thread
        except Exception as e:
            # Fail closed: without the thread, the reply-gate and per-thread cap
            # can't be checked, so the post is skipped, not answered blind.
            log.warning("could not load thread for %s: %s", notif.uri, e)
            return Context(text=text, author=f"@{notif.author.handle}", thread=[]), 0, True
        viewer = getattr(getattr(view, "post", None), "viewer", None)
        replies_disabled = bool(getattr(viewer, "reply_disabled", False))
        node = getattr(view, "parent", None)
        while node is not None and hasattr(node, "post"):
            author = node.post.author
            if author.did == self.me.did:
                my_replies += 1
            if len(thread) < THREAD_PARENTS:
                name = "Jar! (you)" if author.did == self.me.did else f"@{author.handle}"
                thread.append((name, self._clean(node.post.record.text)))
            node = getattr(node, "parent", None)
        thread.reverse()
        return Context(text=text, author=f"@{notif.author.handle}", thread=thread), my_replies, replies_disabled

    def _clean(self, text: str) -> str:
        """Drop Jar!'s handle where it just summons the bot; mid-sentence, it becomes the name.

        "@jar what's up" -> "what's up", and "So, I made a thing. @jar, how are
        you?" -> "So, I made a thing. Jar, how are you?" (not "thing. , how").
        """
        text = self._leading_mention.sub("", text or "")
        text = self._trailing_mention.sub("", text)
        return self._mention_rx.sub("Jar", text).strip()
