"""Prompt assembly, without loading a model."""

import re

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
