"""Bot logic against a fake Bluesky client and a fake model."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest
from atproto import models
from atproto_client.exceptions import BadRequestError, UnauthorizedError
from atproto_client.models.common import XrpcError
from atproto_client.request import Response

from jarbot import bot as botmod
from jarbot.brain import Context
from jarbot.config import Settings

ME = "did:plc:jar"


def now_iso(hours_ago: float = 0) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat().replace("+00:00", "Z")


def author(did, handle, bot=False):
    return NS(did=did, handle=handle, labels=[NS(val="bot")] if bot else [])


def notif(uri, text, reason="mention", who=author("did:plc:alice", "alice.test"), reply=None, hours_ago=0):
    return NS(uri=uri, cid="cid-" + uri, reason=reason, author=who, indexed_at=now_iso(hours_ago),
              record=NS(text=text, reply=reply))


def ref(uri):
    return models.ComAtprotoRepoStrongRef.Main(uri=uri, cid="cid-" + uri)


class FakeClient:
    def __init__(self, notifications, threads=None, search=None, page_size=None):
        self.me = NS(did=ME, handle="jarjarbot.bsky.social")
        self.notifications = notifications  # oldest first
        self.search = search or []  # posts the text search finds
        self.threads = threads or {}
        self.sent = []
        self.liked = []
        self.seen_updates = 0
        self.pages_read = 0
        self.searches = 0
        outer = self

        class Notification:
            def list_notifications(self, params):
                # newest first, paged, filtered by reason like the real API
                items = [n for n in reversed(outer.notifications) if n.reason in params.get("reasons", [n.reason])]
                size = page_size or params["limit"]
                start = int(params.get("cursor") or 0)
                outer.pages_read += 1
                more = start + size < len(items)
                return NS(notifications=items[start:start + size], cursor=str(start + size) if more else None)

            def update_seen(self, params):
                outer.seen_updates += 1

        class Feed:
            def search_posts(self, params):
                outer.searches += 1
                return NS(posts=list(outer.search), cursor=None)

        self.app = NS(bsky=NS(notification=Notification(), feed=Feed()))

    def get_post_thread(self, uri, depth=0, parent_height=0):
        return NS(thread=self.threads.get(uri, NS(post=None, parent=None)))

    def get_current_time_iso(self):
        return now_iso()

    def like(self, uri, cid):
        self.liked.append((uri, cid))

    def send_post(self, text, reply_to=None, langs=None):
        self.sent.append((text, reply_to))
        return NS(uri=f"at://{ME}/app.bsky.feed.post/{len(self.sent)}")


class FakeBrain:
    def __init__(self):
        self.contexts: list[Context] = []

    def reply(self, ctx):
        self.contexts.append(ctx)
        return f"Meesa reply #{len(self.contexts)}"


@pytest.fixture
def make_bot(tmp_path, monkeypatch):
    def make(notifications, threads=None, dry_run=False, clients=None, **overrides):
        client = FakeClient(notifications, threads)
        queue = list(clients or [])  # clients handed out by later logins
        logins = []

        def fake_login(settings, **kw):
            logins.append(kw)
            return client if len(logins) == 1 else queue.pop(0)

        monkeypatch.setattr(botmod, "login", fake_login)
        settings = Settings(handle="jarjarbot.bsky.social", app_password="x", state_dir=tmp_path,
                            poll_seconds=0, max_age_hours=24, sweep_minutes=10,
                            max_per_author_per_hour=10, max_per_hour=60,
                            max_replies_per_thread=10, reply_to_bots=False)
        for k, v in overrides.items():
            setattr(settings, k, v)
        brain = FakeBrain()
        bot = botmod.Bot(settings, brain, dry_run=dry_run)
        bot.logins = logins
        return bot, client, brain

    return make


def test_replies_to_mention_once(make_bot):
    n = notif("at://did:plc:alice/app.bsky.feed.post/1", "@jarjarbot.bsky.social what's nine plus ten?")
    bot, client, brain = make_bot([n])
    bot.poll_once()
    bot.poll_once()
    assert len(client.sent) == 1
    assert brain.contexts[0].text == "what's nine plus ten?"
    text, reply_ref = client.sent[0]
    # a top-level post: it is both root and parent of Jar!'s reply
    assert reply_ref.root.uri == reply_ref.parent.uri == n.uri
    assert client.seen_updates == 1
    assert client.liked == [(n.uri, n.cid)]


def test_reply_keeps_thread_root(make_bot):
    root = ref("at://did:plc:bob/app.bsky.feed.post/root")
    n = notif("at://did:plc:alice/app.bsky.feed.post/2", "hey @jarjarbot.bsky.social",
              reply=NS(root=root, parent=ref("at://did:plc:bob/app.bsky.feed.post/mid")))
    bot, client, _ = make_bot([n])
    bot.poll_once()
    _, reply_ref = client.sent[0]
    assert reply_ref.root.uri == root.uri
    assert reply_ref.parent.uri == n.uri


def test_thread_context_oldest_first(make_bot):
    n = notif("at://did:plc:alice/app.bsky.feed.post/3", "lol ok", reason="reply",
              reply=NS(root=ref("at://did:plc:bob/app.bsky.feed.post/a"),
                       parent=ref(f"at://{ME}/app.bsky.feed.post/b")))
    thread = NS(post=None, parent=NS(
        post=NS(author=author(ME, "jarjarbot.bsky.social"), record=NS(text="Meesa love dis!")),
        parent=NS(post=NS(author=author("did:plc:bob", "bob.test"), record=NS(text="@jarjarbot.bsky.social say meesa")),
                  parent=None)))
    bot, _, brain = make_bot([n], threads={n.uri: thread})
    bot.poll_once()
    ctx = brain.contexts[0]
    assert ctx.thread == [("@bob.test", "say meesa"), ("Jar! (you)", "Meesa love dis!")]
    assert ctx.author == "@alice.test"


def test_skips(make_bot):
    ns = [
        notif("at://x/1", "from a bot", who=author("did:plc:b", "b.test", bot=True)),
        notif("at://x/2", "old", hours_ago=30),
        notif("at://x/3", "mine", who=author(ME, "jarjarbot.bsky.social")),
        notif("at://x/4", "talking to bob under Jar!'s thread", reason="reply",
              reply=NS(root=ref(f"at://{ME}/app.bsky.feed.post/r"), parent=ref("at://did:plc:bob/app.bsky.feed.post/p"))),
        notif("at://x/5", "liked", reason="like"),
    ]
    bot, client, _ = make_bot(ns)
    bot.poll_once()
    assert client.sent == [] and client.liked == []
    assert not bot.store.seen("at://x/5")  # likes are not even looked at
    assert all(bot.store.seen(f"at://x/{i}") for i in range(1, 5))


def test_author_rate_limit(make_bot):
    ns = [notif(f"at://x/{i}", f"hi {i}") for i in range(5)]
    bot, client, _ = make_bot(ns, max_per_author_per_hour=3)
    bot.poll_once()
    assert len(client.sent) == 3


def test_thread_reply_limit(make_bot):
    mine = lambda parent: NS(post=NS(author=author(ME, "j"), record=NS(text="meesa")), parent=parent)
    n = notif("at://x/9", "again!")
    bot, client, _ = make_bot([n], threads={n.uri: NS(post=None, parent=mine(mine(None)))},
                              max_replies_per_thread=2)
    bot.poll_once()
    assert client.sent == []


def test_dry_run_posts_and_records_nothing(make_bot):
    n = notif("at://x/1", "hello")
    bot, client, brain = make_bot([n], dry_run=True)
    bot.poll_once()
    bot.poll_once()
    assert client.sent == [] and client.seen_updates == 0 and client.liked == []
    assert len(brain.contexts) == 1
    assert not bot.store.seen(n.uri)


def expired_token():
    return BadRequestError(Response(success=False, status_code=400, headers={},
                                    content=XrpcError(error="ExpiredToken", message="Token has expired")))


class DeadSessionClient(FakeClient):
    """A client whose session was invalidated, e.g. by another process refreshing it."""

    def __init__(self, error):
        super().__init__([])

        class Notification:
            def list_notifications(self, params):
                raise error

        self.app = NS(bsky=NS(notification=Notification()))


def test_logs_in_again_when_session_dies(make_bot):
    n = notif("at://x/1", "hello")
    fresh = FakeClient([n])
    bot, _, _ = make_bot([], clients=[fresh])
    bot.client = DeadSessionClient(expired_token())
    bot._last_login -= botmod.RELOGIN_MIN_SECONDS
    bot.step()
    assert bot.client is fresh
    assert bot.logins[-1] == {"reuse_saved": False, "save": True}  # password login, saved for next start
    bot.step()
    assert len(fresh.sent) == 1


def test_relogin_is_rate_limited(make_bot):
    bot, _, _ = make_bot([])
    dead = DeadSessionClient(UnauthorizedError())
    bot.client = dead
    bot.step()  # just logged in at startup: don't hammer createSession
    assert bot.client is dead and len(bot.logins) == 1


def test_other_errors_dont_trigger_login(make_bot):
    bot, _, _ = make_bot([])
    bot.client = DeadSessionClient(RuntimeError("network hiccup"))
    bot._last_login -= botmod.RELOGIN_MIN_SECONDS
    bot.step()
    assert len(bot.logins) == 1


def test_auth_error_while_replying_retries_the_post(make_bot):
    n = notif("at://x/1", "hello")
    bot, client, _ = make_bot([n], clients=[FakeClient([n])])

    def dead_send(**kw):
        raise expired_token()

    client.send_post = dead_send
    bot._last_login -= botmod.RELOGIN_MIN_SECONDS
    bot.step()
    assert not bot.store.seen(n.uri)  # not marked as handled...
    bot.step()
    assert len(bot.client.sent) == 1  # ...so the new session answers it


def test_dry_run_leaves_saved_session_alone(make_bot):
    bot, _, _ = make_bot([], dry_run=True)
    assert bot.logins[0] == {"reuse_saved": False, "save": False}


@pytest.mark.parametrize("text, cleaned", [
    ("@jarjarbot.bsky.social what's nine plus ten?", "what's nine plus ten?"),
    ("@jarjarbot.bsky.social, @JarJarBot.bsky.social: hi", "hi"),
    ("So, I made a thing. @jarjarbot.bsky.social, how are you today?", "So, I made a thing. Jar, how are you today?"),
    ("what do you think? @jarjarbot.bsky.social", "what do you think?"),
    ("great post @jarjarbot.bsky.social!", "great post"),
    ("ask @bob.test about it", "ask @bob.test about it"),
])
def test_mention_cleanup(make_bot, text, cleaned):
    bot, _, _ = make_bot([])
    assert bot._clean(text) == cleaned


def test_failed_like_does_not_cause_a_second_reply(make_bot):
    n = notif("at://x/1", "hello")
    bot, client, _ = make_bot([n])

    def broken_like(uri, cid):
        raise expired_token()

    client.like = broken_like
    bot.step()
    bot.step()
    assert len(client.sent) == 1
    assert bot.store.seen(n.uri)


def post(uri, text, who=author("did:plc:carol", "carol.test"), hours_ago=0):
    """A search result: a post view, which has no notification reason."""
    return NS(uri=uri, cid="cid-" + uri, author=who, indexed_at=now_iso(hours_ago), record=NS(text=text, reply=None))


def test_finds_mentions_that_have_no_notification(make_bot):
    p = post("at://x/plain", "@jarjarbot.bsky.social what's the correct number here?")
    bot, client, brain = make_bot([])
    client.search = [p, post("at://x/about", "jarjarbot.bsky.social is a fun bot")]  # second one isn't addressed to Jar!
    bot.poll_once()
    assert [ctx.text for ctx in brain.contexts] == ["what's the correct number here?"]
    assert client.sent[0][1].parent.uri == p.uri
    assert bot.store.seen(p.uri) and not bot.store.seen("at://x/about")


def test_post_found_twice_is_answered_once(make_bot):
    n = notif("at://x/1", "@jarjarbot.bsky.social hi")
    bot, client, _ = make_bot([n])
    client.search = [post(n.uri, n.record.text, who=n.author)]
    bot.poll_once()
    assert len(client.sent) == 1


def test_sweeps_only_now_and_then(make_bot):
    bot, client, _ = make_bot([])
    bot.poll_once()
    bot.poll_once()
    assert client.searches == 1
    bot._last_sweep -= bot.settings.sweep_minutes * 60
    bot.poll_once()
    assert client.searches == 2


def test_catches_up_on_older_pages_after_downtime(make_bot, monkeypatch):
    monkeypatch.setattr(botmod, "PAGE_SIZE", 2)
    ns = [notif(f"at://x/{i}", f"hi {i}", who=author(f"did:plc:u{i}", f"u{i}.test"), hours_ago=20 - i)
          for i in range(5)]
    likes = [notif(f"at://x/like{i}", "", reason="like") for i in range(10)]
    bot, client, brain = make_bot(ns + likes)
    bot.poll_once()  # startup sweep: pages back through all of them, likes filtered out
    assert [c.text for c in brain.contexts] == ["hi 0", "hi 1", "hi 2", "hi 3", "hi 4"]  # oldest first


def test_normal_poll_reads_one_page(make_bot, monkeypatch):
    monkeypatch.setattr(botmod, "PAGE_SIZE", 2)
    bot, client, _ = make_bot([notif(f"at://x/{i}", "hi") for i in range(5)])
    bot._last_sweep = botmod.time.monotonic()  # pretend a sweep just ran
    bot.poll_once()
    assert client.pages_read == 1


def test_older_than_a_day_is_skipped(make_bot):
    old = notif("at://x/old", "hello?", hours_ago=25)
    bot, client, _ = make_bot([old, notif("at://x/new", "hi", hours_ago=23)])
    bot.poll_once()
    assert len(client.sent) == 1
    assert bot.store.seen(old.uri)  # recorded as skipped


def bad_request(error="InvalidRequest"):
    return BadRequestError(Response(success=False, status_code=400, headers={},
                                    content=XrpcError(error=error, message="nope")))


def test_post_the_model_chokes_on_does_not_block_the_rest(make_bot):
    poison, ok = notif("at://x/1", "poison", hours_ago=1), notif("at://x/2", "hello")
    bot, client, brain = make_bot([poison, ok])
    real_reply = brain.reply

    def reply(ctx):
        if ctx.text == "poison":
            raise ValueError("Requested tokens (9000) exceed context window of 4096")
        return real_reply(ctx)

    brain.reply = reply
    bot.step()
    bot.step()
    assert [text for text, _ in client.sent] == ["Meesa reply #1"]
    assert client.sent[0][1].parent.uri == ok.uri
    assert bot.store.seen(poison.uri)  # recorded as an error, never tried again


def test_one_failing_post_does_not_hold_up_the_next(make_bot):
    first, second = notif("at://x/1", "one", hours_ago=1), notif("at://x/2", "two")
    bot, client, _ = make_bot([first, second])
    real_send = client.send_post

    def send_post(text, reply_to=None, langs=None):
        if reply_to.parent.uri == first.uri:
            raise RuntimeError("network hiccup")
        return real_send(text, reply_to=reply_to, langs=langs)

    client.send_post = send_post
    bot.step()
    assert [ref.parent.uri for _, ref in client.sent] == [second.uri]
    assert not bot.store.seen(first.uri)  # a passing failure: try it again next poll


def test_replies_that_fail_to_post_count_toward_rate_limits(make_bot):
    ns = [notif(f"at://x/{i}", f"hi {i}") for i in range(5)]
    bot, client, brain = make_bot(ns, max_per_author_per_hour=3)

    def gated(**kw):
        raise bad_request()

    client.send_post = gated
    bot.poll_once()
    assert len(brain.contexts) == 3


def test_skips_threads_with_replies_disabled(make_bot):
    n = notif("at://x/1", "you can't answer this")
    gated = NS(post=NS(viewer=NS(reply_disabled=True)), parent=None)
    bot, client, brain = make_bot([n], threads={n.uri: gated})
    bot.poll_once()
    assert brain.contexts == [] and client.sent == []
    assert bot.store.seen(n.uri)


def test_session_file_is_private(tmp_path, monkeypatch):
    from atproto import SessionEvent

    class FakeAtproto:
        def on_session_change(self, callback):
            self.callback = callback

        def login(self, handle=None, password=None, session_string=None):
            self.me = NS(handle=handle, did=ME)
            self.callback(SessionEvent.CREATE, None)

        def export_session_string(self):
            return "secret tokens"

    monkeypatch.setattr(botmod, "Client", FakeAtproto)
    session = tmp_path / "session.txt"
    session.write_text("old")
    session.chmod(0o644)
    old_umask = botmod.os.umask(0)
    try:
        botmod.login(Settings(handle="jar.test", app_password="x", state_dir=tmp_path, poll_seconds=0,
                              max_age_hours=24, sweep_minutes=10, max_per_author_per_hour=10, max_per_hour=60,
                              max_replies_per_thread=10, reply_to_bots=False), reuse_saved=False)
    finally:
        botmod.os.umask(old_umask)
    assert session.read_text() == "secret tokens"
    assert session.stat().st_mode & 0o777 == 0o600
