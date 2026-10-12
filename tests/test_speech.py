from jarbot.speech import MAX_CHARS, jarify, meesafy


def test_quirks():
    assert meesafy("I think you are right") == "Meesa think yousa right"
    assert meesafy("Then I gave it to them") == "Then meesa gave it to them"
    assert meesafy("give me your hat") == "give meesa yousa hat"
    assert meesafy("I'm sure we're fine, I’ll go") == "Meesa sure wesa fine, meesa gonna go"
    assert meesafy("I don't know. I am not sure, guys") == "Meesa no know. Meesa no sure, boyos"
    assert meesafy("so I am happy") == "so meesa happy"


def test_contractions_dont_eat_real_words():
    assert meesafy("We were well. I feel ill.") == "Wesa were well. Meesa feel ill."


def test_made_up_words():
    assert meesafy("Excuse me, okay?") == "Ex-squeeze me, okeeday?"
    assert meesafy("Excuse me, OK? OK!") == "Ex-squeeze me, okeeday? Okeeday!"


def test_accent_spellings_are_undone():
    assert meesafy("Meesa tink dis ting is nutting, boyos") == "Meesa think this thing is nothing, boyos"
    assert meesafy("swimmin' in da water, dat is bombad") == "swimming in the water, that is bombad"
    assert meesafy("How yousa doin’?") == "How yousa doing?"
    assert meesafy("a fox in its den") == "a fox in its den"


def test_idempotent():
    once = meesafy("I think you are right, okay?")
    assert meesafy(once) == once


def test_does_not_touch_inside_words():
    assert meesafy("item thistle mythic dated") == "item thistle mythic dated"


def test_tidy_strips_mentions_links_hashtags_emoji():
    out = jarify("Hi @bob.bsky.social, see https://example.com #Swamp 🎉 — yay")
    assert out == "Hi, see Swamp, yay"


def test_strips_speaker_tag_and_quotes():
    assert jarify('"Jar Bot: Oh boy!"') == "Oh boy!"


def test_length_limit():
    long = "Meesa very happy. " * 40
    out = jarify(long)
    assert len(out) <= MAX_CHARS
    assert out.endswith(".")
