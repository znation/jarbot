"""Bot logic against a fake Bluesky client and a fake model."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest
from atproto import models

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
    def __init__(self, notifications, threads=None):
        self.me = NS(did=ME, handle="jarjarbot.bsky.social")
        self.notifications = notifications
        self.threads = threads or {}
        self.sent = []
        self.seen_updates = 0
        outer = self

        class Notification:
            def list_notifications(self, params):
                return NS(notifications=list(reversed(outer.notifications)))  # newest first

            def update_seen(self, params):
                outer.seen_updates += 1

        self.app = NS(bsky=NS(notification=Notification()))

    def get_post_thread(self, uri, depth=0, parent_height=0):
        return NS(thread=self.threads.get(uri, NS(post=None, parent=None)))

    def get_current_time_iso(self):
        return now_iso()

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
    def make(notifications, threads=None, dry_run=False, **overrides):
        client = FakeClient(notifications, threads)
        monkeypatch.setattr(botmod, "login", lambda settings: client)
        settings = Settings(handle="jarjarbot.bsky.social", app_password="x", state_dir=tmp_path,
                            poll_seconds=0, max_age_hours=6, max_per_author_per_hour=10, max_per_hour=60,
                            max_replies_per_thread=10, reply_to_bots=False)
        for k, v in overrides.items():
            setattr(settings, k, v)
        brain = FakeBrain()
        return botmod.Bot(settings, brain, dry_run=dry_run), client, brain

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
        notif("at://x/2", "old", hours_ago=10),
        notif("at://x/3", "mine", who=author(ME, "jarjarbot.bsky.social")),
        notif("at://x/4", "talking to bob under Jar!'s thread", reason="reply",
              reply=NS(root=ref(f"at://{ME}/app.bsky.feed.post/r"), parent=ref("at://did:plc:bob/app.bsky.feed.post/p"))),
        notif("at://x/5", "liked", reason="like"),
    ]
    bot, client, _ = make_bot(ns)
    bot.poll_once()
    assert client.sent == []
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
    assert client.sent == [] and client.seen_updates == 0
    assert len(brain.contexts) == 1
    assert not bot.store.seen(n.uri)
