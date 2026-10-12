"""The Bluesky side: poll notifications, reply to mentions, replies and quotes."""

import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone

from atproto import Client, SessionEvent, models
from atproto_client.exceptions import BadRequestError, LoginRequiredError, UnauthorizedError

from .brain import Brain, Context
from .config import Settings
from .store import Store

log = logging.getLogger(__name__)

REASONS = ("mention", "reply", "quote")
THREAD_PARENTS = 6  # how many posts up the thread to show the model
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
                session_file.write_text(client.export_session_string())
                os.chmod(session_file, 0o600)

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
        resp = self.client.app.bsky.notification.list_notifications({"limit": 50})
        fresh = [n for n in resp.notifications
                 if n.reason in REASONS and not self.store.seen(n.uri) and n.uri not in self._dry_seen]
        for notif in reversed(fresh):  # oldest first
            if self.dry_run:
                self._dry_seen.add(notif.uri)
            self.handle(notif)
        if fresh and not self.dry_run:
            self.client.app.bsky.notification.update_seen({"seen_at": self.client.get_current_time_iso()})

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
        indexed = datetime.fromisoformat(notif.indexed_at.replace("Z", "+00:00"))
        if datetime.now(timezone.utc) - indexed > timedelta(hours=s.max_age_hours):
            return self._skip(notif, "too old")
        if notif.reason == "reply" and record.reply and _did_of(record.reply.parent.uri) != self.me.did:
            # Someone replied to someone else further down a thread Jar! started.
            return self._skip(notif, "reply not addressed to Jar!")
        if self.store.replies_since(3600, notif.author.did) >= s.max_per_author_per_hour:
            return self._skip(notif, "author rate limit")
        if self.store.replies_since(3600) >= s.max_per_hour:
            log.warning("global hourly cap reached; leaving %s for later", notif.uri)
            return  # don't record: try again next poll

        ctx, my_replies = self.context(notif)
        if my_replies >= s.max_replies_per_thread:
            return self._skip(notif, "thread reply limit")
        if not ctx.text and not ctx.thread:
            return self._skip(notif, "empty post")

        t0 = time.monotonic()
        reply = self.brain.reply(ctx)
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
        except BadRequestError as e:  # deleted post, blocked, reply-gated thread...
            if is_auth_error(e):
                raise  # not this post's fault: log in again and retry it next poll
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

    def context(self, notif) -> tuple[Context, int]:
        """The post's text and the thread above it; also how often Jar! already spoke there."""
        text = self._clean(notif.record.text)
        thread: list[tuple[str, str]] = []
        my_replies = 0
        try:
            view = self.client.get_post_thread(notif.uri, depth=0, parent_height=50).thread
            node = getattr(view, "parent", None)
            while node is not None and hasattr(node, "post"):
                author = node.post.author
                if author.did == self.me.did:
                    my_replies += 1
                if len(thread) < THREAD_PARENTS:
                    name = "Jar! (you)" if author.did == self.me.did else f"@{author.handle}"
                    thread.append((name, self._clean(node.post.record.text)))
                node = getattr(node, "parent", None)
        except Exception as e:
            log.warning("could not load thread for %s: %s", notif.uri, e)
        thread.reverse()
        return Context(text=text, author=f"@{notif.author.handle}", thread=thread), my_replies

    def _clean(self, text: str) -> str:
        """Drop Jar!'s handle where it just summons the bot; mid-sentence, it becomes the name.

        "@jar what's up" -> "what's up", and "So, I made a thing. @jar, how are
        you?" -> "So, I made a thing. Jar, how are you?" (not "thing. , how").
        """
        text = self._leading_mention.sub("", text or "")
        text = self._trailing_mention.sub("", text)
        return self._mention_rx.sub("Jar", text).strip()
