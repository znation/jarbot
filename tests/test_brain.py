"""Prompt assembly, without loading a model."""

import re

import pytest

from jarbot.brain import Brain, Context
from jarbot.persona import Persona, mentions_franchise

FRANCHISE_WORDS = re.compile(r"jar\s*jar|binks|gungan|star\s*wars|lucas|naboo|phantom", re.I)


def prompt(ctx: Context) -> str:
    brain = Brain.__new__(Brain)  # skip loading the model
    return "\n".join(m["content"] for m in brain.messages(ctx, Persona("Bonks", "Gangun")))


def test_prompt_never_names_the_franchise():
    for text in ["hi", "Are you Jar Jar Binks from Star Wars?", "a Gungan from Naboo, by Lucasfilm"]:
        p = prompt(Context(text, thread=[("@bob", "Star Wars Episode I rules")]))
        assert not FRANCHISE_WORDS.search(p), p
        assert not mentions_franchise(p)


def test_prompt_uses_persona():
    p = prompt(Context("what's your name?"))
    assert "Jar Bot Bonks" in p and "Gangun" in p


def test_denial_only_when_movies_come_up():
    assert "resemblance" not in prompt(Context("what's your name?")).lower()
    assert "resemblance" in prompt(Context("are you Jar Jar?")).lower()
    assert "resemblance" in prompt(Context("you look like a movie character")).lower()


class ByteTokenizer:
    """Stands in for llama.cpp: one token per byte."""

    def tokenize(self, text, add_bos=True):
        return list(text)

    def detokenize(self, tokens):
        return bytes(tokens)


def fitted(ctx: Context, room: int) -> tuple[list[dict], int]:
    """Messages for ctx, in a context window with `room` tokens to spare beyond a bare prompt."""
    import jarbot.brain as brainmod

    brain = Brain.__new__(Brain)
    brain.llm = ByteTokenizer()
    persona = Persona("Bonks", "Gangun")
    cost = lambda msgs: sum(len(m["content"].encode()) + brainmod.MESSAGE_OVERHEAD for m in msgs)
    bare = cost(brain.messages(Context("x"), persona))
    old = brainmod.N_CTX
    brainmod.N_CTX = brainmod.MAX_TOKENS + bare + room
    try:
        msgs = brain.fitted_messages(ctx, persona)
        return msgs, brainmod.N_CTX - brainmod.MAX_TOKENS - cost(msgs)
    finally:
        brainmod.N_CTX = old


def test_long_thread_is_trimmed_to_fit_oldest_first():
    zalgo = "a" + "̶" * 1500
    thread = [("@a", "first " + zalgo), ("@b", "second " + zalgo), ("@c", "third")]
    msgs, spare = fitted(Context("the question", "@d", thread), room=3500)
    last = msgs[-1]["content"]
    assert spare >= 0
    assert "the question" in last and "third" in last and "second" in last
    assert "first" not in last


def test_huge_post_is_cut_to_fit():
    msgs, spare = fitted(Context("start " + "̶" * 1500), room=200)
    assert spare >= 0
    assert msgs[-1]["content"].startswith("start")


def test_short_prompt_is_left_alone():
    ctx = Context("hi", "@d", [("@a", "hello there")])
    msgs, _ = fitted(ctx, room=1000)
    assert "hello there" in msgs[-1]["content"] and "hi" in msgs[-1]["content"]


QWEN, GEMMA = "unsloth/Qwen3-4B-Instruct-2507-GGUF", "unsloth/gemma-3-4b-it-GGUF"


@pytest.fixture
def downloads(monkeypatch, tmp_path):
    """A fake Hub that hands out the commit asked for. With `downloads.offline` set, it
    acts like hf_hub_download when the Hub can't be reached: it returns the file already
    on disk, whatever commit that came from."""
    import huggingface_hub
    from huggingface_hub._local_folder import write_download_metadata

    calls = []

    def fake_download(repo, filename, revision, local_dir):
        calls.append((repo, revision))
        if not calls.offline:
            (tmp_path / filename).write_bytes(b"GGUF")
            write_download_metadata(tmp_path, filename, commit_hash=revision, etag="e")
        return str(tmp_path / filename)

    calls = type("Calls", (list,), {"offline": False})()
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    for var in ("JAR_MODEL_REPO", "JAR_MODEL_FILE", "JAR_MODEL_REVISION"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("JAR_MODELS_DIR", str(tmp_path))
    return calls


@pytest.mark.parametrize("repo", [None, QWEN, GEMMA])
def test_known_models_download_a_pinned_commit(downloads, monkeypatch, repo):
    from jarbot.brain import PINNED_REVISIONS, _download_default

    if repo:
        monkeypatch.setenv("JAR_MODEL_REPO", repo)
    _download_default()
    assert downloads == [(repo or QWEN, PINNED_REVISIONS[repo or QWEN])]


@pytest.mark.parametrize("revision", [None, "main", "v1.0", "a06e946b"])
def test_other_models_need_a_commit(downloads, monkeypatch, revision):
    from jarbot.brain import _download_default

    monkeypatch.setenv("JAR_MODEL_REPO", "someone/other-GGUF")
    if revision:
        monkeypatch.setenv("JAR_MODEL_REVISION", revision)
    with pytest.raises(SystemExit):
        _download_default()
    assert downloads == []


def test_revision_can_be_overridden(downloads, monkeypatch):
    from jarbot.brain import _download_default

    commit = "0123456789abcdef0123456789abcdef01234567"
    monkeypatch.setenv("JAR_MODEL_REPO", "someone/other-GGUF")
    monkeypatch.setenv("JAR_MODEL_REVISION", commit)
    _download_default()
    assert downloads == [("someone/other-GGUF", commit)]


def test_offline_fallback_to_another_commit_is_refused(downloads, monkeypatch):
    from jarbot.brain import _download_default

    monkeypatch.setenv("JAR_MODEL_REVISION", "f" * 40)
    _download_default()  # the file from before the pin moved
    monkeypatch.delenv("JAR_MODEL_REVISION")
    downloads.offline = True
    with pytest.raises(SystemExit, match="not the pinned"):
        _download_default()


def test_offline_with_the_pinned_file_is_fine(downloads):
    from jarbot.brain import _download_default

    path = _download_default()
    downloads.offline = True
    assert _download_default() == path


def test_file_changed_since_download_is_refused(downloads):
    import os

    from jarbot.brain import _download_default

    path = _download_default()
    later = os.stat(path).st_mtime + 60
    os.utime(path, (later, later))  # edited after the download record was written
    downloads.offline = True
    with pytest.raises(SystemExit, match="unknown commit"):
        _download_default()
