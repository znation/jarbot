"""Deterministic speech pass.

Jar!'s voice is made of invented quirks: "meesa"/"yousa"/"wesa" pronouns,
dropped "am/is/are" after them, "no" for not, and its own words (okeeday,
ex-squeeze me, boyos). It never imitates a real-world accent, so this pass
enforces the quirks and also undoes accent spellings small models slip into
("da", "dat", "tink", "swimmin'"). It then cleans the text up for posting: no
@-mentions, no links, no hashtags, and it fits in a Bluesky post.
"""

import re

MAX_CHARS = 280  # Bluesky allows 300 graphemes; leave headroom.

# Order matters: multi-word phrases before the single words they contain.
# Each pattern is matched case-insensitively on word boundaries, and the
# replacement copies the capitalisation of the original. Contractions that
# collide with real words without the apostrophe (ill, id, were, well) need it.
_RULES: list[tuple[str, str]] = [
    # Jar!'s quirks.
    (r"excuse me", "ex-squeeze me"),
    (r"i['’]?m not", "meesa no"),
    (r"i am not", "meesa no"),
    (r"you['’]?re not", "yousa no"),
    (r"you are not", "yousa no"),
    (r"we['’]re not", "wesa no"),
    (r"we are not", "wesa no"),
    (r"i['’]?m", "meesa"),
    (r"i am", "meesa"),
    (r"i['’]?ve", "meesa have"),
    (r"i['’]ll", "meesa gonna"),
    (r"i['’]d", "meesa would"),
    (r"you['’]?re", "yousa"),
    (r"you are", "yousa"),
    (r"you['’]?ve", "yousa have"),
    (r"you['’]?ll", "yousa gonna"),
    (r"you['’]d", "yousa would"),
    (r"we['’]re", "wesa"),
    (r"we are", "wesa"),
    (r"we['’]?ve", "wesa have"),
    (r"we['’]ll", "wesa gonna"),
    (r"don['’]?t", "no"),
    (r"doesn['’]?t", "no"),
    (r"didn['’]?t", "no"),
    (r"can['’]?t", "no can"),
    (r"won['’]?t", "no gonna"),
    (r"not", "no"),
    (r"going to", "gonna"),
    (r"yourself", "yousaself"),
    (r"myself", "meesaself"),
    (r"yours", "yousa's"),
    (r"your", "yousa"),
    (r"you", "yousa"),
    (r"ya", "yousa"),
    (r"me", "meesa"),
    (r"my", "mesa"),
    (r"mine", "mesa's"),
    (r"i", "meesa"),
    (r"we", "wesa"),
    (r"okay", "okeeday"),
    (r"ok", "okeeday"),
    (r"guys", "boyos"),
    # Accent spellings the model slips into, back to standard spelling.
    (r"da", "the"),
    (r"dis", "this"),
    (r"dat", "that"),
    (r"dese", "these"),
    (r"dey", "they"),
    (r"dem", "them"),
    (r"dere", "there"),
    (r"tink", "think"),
    (r"tinks", "thinks"),
    (r"ting", "thing"),
    (r"tings", "things"),
    (r"nutting", "nothing"),
    (r"someting", "something"),
    (r"anyting", "anything"),
    (r"everyting", "everything"),
]

# One alternation, so each span of text is rewritten at most once (otherwise
# "ex-squeeze me" would go on to become "ex-squeeze meesa").
_PATTERN = re.compile(
    r"(?<![\w'’@#/.-])(?:"
    + "|".join(f"(?P<r{n}>{pat})" for n, (pat, _) in enumerate(_RULES))
    + r")(?![\w'’])",
    re.IGNORECASE,
)

_MENTION = re.compile(r"(?<!\w)@[\w.-]+")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_HASHTAG = re.compile(r"(?<!\w)#(\w+)")
_SPACES = re.compile(r"[ \t]+")
# Emoji and pictographs (the persona prompt forbids them, small models add them anyway).
_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF\uFE0F\u200D]+"
)


def _match_case(original: str, replacement: str) -> str:
    if original.isupper() and len(original) > 2:
        return replacement.upper()
    if original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


# Dropped final g: "swimmin'" -> "swimming", "doin'" -> "doing".
_DROPPED_G = re.compile(r"(?<![\w'’])([A-Za-z]+[a-z])in['’](?![\w'’])")


def meesafy(text: str) -> str:
    """Apply Jar!'s quirks, and spell everything else the standard way."""
    text = _DROPPED_G.sub(r"\1ing", text)
    return _PATTERN.sub(_replace, text)


def _replace(m: re.Match) -> str:
    word = m.group(0)
    rep = _RULES[int(m.lastgroup[1:])][1]
    if re.fullmatch(r"i(?:['’ ]\w+)*|ok", word, re.IGNORECASE):
        # "I" and "OK" are capitalised everywhere in English, so their case
        # says nothing: capitalise only at the start of a sentence.
        before = m.string[: m.start()].rstrip()
        if before and before[-1] not in ".!?\n":
            return rep
        return rep[0].upper() + rep[1:]
    return _match_case(word, rep)


def tidy(text: str, limit: int = MAX_CHARS) -> str:
    """Strip things the bot must not post and fit the text into one post."""
    text = text.strip().strip('"').strip()
    # Drop a leading speaker tag like "Jar Bot:" or "Jar!:".
    text = re.sub(r"^\s*(jar\s*bot|jar!?)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = _MENTION.sub("", text)
    text = _URL.sub("", text)
    text = _HASHTAG.sub(r"\1", text)
    text = _EMOJI.sub("", text)
    text = re.sub(r"\s*[—–]\s*", ", ", text)  # em dashes read as machine-written
    text = _SPACES.sub(" ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"\s*,(\s*,)+", ",", text)
    text = re.sub(r" +([,.!?])", r"\1", text).strip()
    if len(text) <= limit:
        return text
    # Cut at the last sentence end that fits, else at a word boundary.
    cut = text[:limit]
    ends = [cut.rfind(p) for p in ".!?"]
    end = max(ends)
    if end >= limit // 3:
        return cut[: end + 1].strip()
    return cut[: cut.rfind(" ")].rstrip(",;: ") + "…"


def jarify(text: str, limit: int = MAX_CHARS) -> str:
    return tidy(meesafy(tidy(text, limit=10_000)), limit=limit)
