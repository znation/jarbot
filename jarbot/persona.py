"""Who Jar! says it is.

The running bit: Jar!'s surname and species come out slightly different every
time ("Jar Bot Bonks, a proud Gangun", then "Jar Bot Banks, a humble Gungian"),
and it has never heard of any movie it might remind you of.
"""

import random
import re
from dataclasses import dataclass

FIRST_NAME = "Jar Bot"

# Surnames are B + vowel + ending, e.g. Bonks, Banks, Bunkles, Bleenx.
_SURNAME_ONSETS = ["B"] * 6 + ["Bl", "Br"]
_SURNAME_VOWELS = ["a", "o", "u", "e", "oo", "ee", "au", "i"]
_SURNAME_ENDINGS = ["nks", "nks", "nx", "ngs", "nkles", "nkers", "mps", "nkus", "nkley"]
_NOT_SURNAMES = {"binks"}

# Species are G + vowel + ng + vowel + n, e.g. Gangun, Gungian, Goongan.
_SPECIES_ONSETS = ["G"] * 6 + ["Gw", "Gr"]
_SPECIES_VOWELS_1 = ["a", "u", "o", "oo", "i", "e"]
_SPECIES_VOWELS_2 = ["a", "u", "ia", "oo", "e", "o", "ua"]
_NOT_SPECIES = {"gungan"}


def _surname(rng: random.Random) -> str:
    while True:
        name = rng.choice(_SURNAME_ONSETS) + rng.choice(_SURNAME_VOWELS) + rng.choice(_SURNAME_ENDINGS)
        if name.lower() not in _NOT_SURNAMES:
            return name


def _species(rng: random.Random) -> str:
    while True:
        name = rng.choice(_SPECIES_ONSETS) + rng.choice(_SPECIES_VOWELS_1) + "ng" + rng.choice(_SPECIES_VOWELS_2) + "n"
        if name.lower() not in _NOT_SPECIES:
            return name


@dataclass(frozen=True)
class Persona:
    surname: str
    species: str

    @property
    def full_name(self) -> str:
        return f"{FIRST_NAME} {self.surname}"


class PersonaPicker:
    """Hands out a fresh surname and species for every reply, never the same as last time."""

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()
        self.last: Persona | None = None

    def next(self) -> Persona:
        while True:
            p = Persona(_surname(self.rng), _species(self.rng))
            if self.last is None or (p.surname != self.last.surname and p.species != self.last.species):
                self.last = p
                return p


# Words that give the game away, by what they get replaced with when someone
# else says them. The model never sees these words, so it can't echo them back
# while denying them; a reply that names one anyway is generated again.
_FRANCHISE_TERMS = [
    ("that movie character", r"jar[\s-]*jar(?:\s+binks)?|\bbinks\b"),
    ("that movie creature", r"\bgungans?\b"),
    ("that space movie", r"star\s*wars|phantom\s+menace|galaxy\s+far,?\s+far\s+away"),
    ("that movie studio", r"lucas\s*film|george\s+lucas|disney"),
    ("that movie planet", r"naboo|otoh\s+gunga|tatooine|coruscant|mos\s+espa"),
    ("that movie stuff",
     r"boss\s+nass|skywalker|anakin|\bjedis?\b|\bsith\b|qui-?\s?gon|obi-?\s?wan|padm[eé]|amidala"
     r"|palpatine|darth|\byoda\b|light\s*sabers?|wookiee?s?|chewbacca|podrac\w*|trade\s+federation"),
]
_MASKS = [(re.compile(pat, re.IGNORECASE), rep) for rep, pat in _FRANCHISE_TERMS]
# Output check. Jar Jar, Binks and Gungan aren't in it: scramble_names() fixes those.
_FRANCHISE = re.compile("|".join(pat for rep, pat in _FRANCHISE_TERMS[2:]), re.IGNORECASE)


def mask_franchise(text: str) -> str:
    """Swap franchise words in someone else's post for vague stand-ins."""
    for rx, rep in _MASKS:
        text = rx.sub(rep, text)
    return text


# Names that are fine to keep but must come out scrambled.
_NAME_FIXES = [
    (re.compile(r"\bjar[\s-]*jar\b", re.IGNORECASE), lambda p: FIRST_NAME),
    (re.compile(r"\bbinks\b", re.IGNORECASE), lambda p: p.surname),
    (re.compile(r"\bgungans\b", re.IGNORECASE), lambda p: p.species + "s"),
    (re.compile(r"\bgungan\b", re.IGNORECASE), lambda p: p.species),
]


def mentions_franchise(text: str) -> bool:
    return _FRANCHISE.search(text) is not None


def scramble_names(text: str, persona: Persona) -> str:
    for rx, rep in _NAME_FIXES:
        text = rx.sub(lambda m, rep=rep: rep(persona), text)
    return text


def denial(persona: Persona, rng: random.Random | None = None) -> str:
    """Fallback when the model keeps naming the thing it must not name."""
    rng = rng or random
    return rng.choice([
        f"Ex-squeeze me? Meesa never heard of that! Meesa just {persona.full_name}, a humble {persona.species}.",
        f"Movies? Studios? Meesa no know anything about that! Meesa a {persona.species} from the underwater city, that all.",
        f"Oh boy, yousa got the wrong {persona.species}! Meesa {persona.full_name}, and meesa never been in a movie.",
        f"How rude! Any resemblance a coincidence, boyos. Meesa {persona.full_name}, one hundred percent original {persona.species}!",
    ])
