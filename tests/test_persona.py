import random

from jarbot.persona import Persona, PersonaPicker, denial, mask_franchise, mentions_franchise, scramble_names


def test_names_vary_and_never_canonical():
    picker = PersonaPicker(random.Random(0))
    picks = [picker.next() for _ in range(2000)]
    assert all(p.surname.lower() != "binks" for p in picks)
    assert all(p.species.lower() != "gungan" for p in picks)
    assert all(p.full_name.startswith("Jar Bot B") for p in picks)
    for a, b in zip(picks, picks[1:]):
        assert a.surname != b.surname and a.species != b.species
    assert len({p.surname for p in picks}) > 50
    assert len({p.species for p in picks}) > 50


def test_scramble_names():
    p = Persona("Bonks", "Gangun")
    assert scramble_names("Meesa Jar Jar Binks, a proud Gungan!", p) == "Meesa Jar Bot Bonks, a proud Gangun!"
    assert scramble_names("all the gungans and jar-jar", p) == "all the Ganguns and Jar Bot"
    assert scramble_names("Banks and Gungian stay", p) == "Banks and Gungian stay"


def test_mentions_franchise():
    for bad in ["Star Wars? never!", "meesa from Naboo", "the Jedi", "LucasFilm", "Boss Nass", "a lightsaber"]:
        assert mentions_franchise(bad), bad
    for ok in ["Meesa am from the river", "Jedidiah", "sithering", "bombad"]:
        assert not mentions_franchise(ok), ok


def test_denial_is_clean():
    p = Persona("Bunks", "Goongan")
    for seed in range(20):
        line = denial(p, random.Random(seed))
        assert not mentions_franchise(line)
        assert "Goongan" in line


def test_mask_franchise():
    assert mask_franchise("Are you Jar Jar Binks from Star Wars?") == "Are you that movie character from that space movie?"
    assert mask_franchise("a Gungan from Naboo working for LucasFilm") == \
        "a that movie creature from that movie planet working for that movie studio"
    assert not mentions_franchise(mask_franchise("Jedi, Sith, Yoda, Darth Maul, Disney"))
    assert mask_franchise("hi Lucas, nice hat") == "hi Lucas, nice hat"
