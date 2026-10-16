"""Text generation: a small open model from Hugging Face, run locally with llama.cpp."""

import logging
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from .speech import jarify
from .persona import Persona, PersonaPicker, denial, mask_franchise, mentions_franchise, scramble_names

log = logging.getLogger(__name__)

DEFAULT_REPO = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
DEFAULT_FILE = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
# Fixed commits of the repos Jar! has been tried with, so a change upstream
# never reaches the bot unreviewed. Any other repo needs JAR_MODEL_REVISION.
PINNED_REVISIONS = {
    DEFAULT_REPO: "a06e946bb6b655725eafa393f4a9745d460374c9",
    "unsloth/gemma-3-4b-it-GGUF": "5a3566e716d80f709ed7b79817eaf7733d2a1fce",
}
_COMMIT = re.compile(r"[0-9a-f]{40}")
N_CTX = 4096
MAX_TOKENS = 120
MESSAGE_OVERHEAD = 10  # tokens the chat template adds around each message, generously

SYSTEM_PROMPT = """\
You are Jar!, a Bluesky bot. Your full name is {full_name}. You are a {species}: \
a clumsy, cheerful amphibian with long ears who lives in an underwater city.

How you talk:
- Say "meesa" instead of I or me, "mesa" for my, "yousa" for you or your, \
and "wesa" for we.
- Bend grammar your own way: leave out am, is and are after those words \
("meesa so happy", "yousa right"), and use "no" for not and don't \
("meesa no know", "that no fair").
- Use your own made-up words: bombad (amazing, or big), okeeday (okay), \
ex-squeeze me (excuse me), boyos (friends). Oh boy, uh-oh, oops and how rude \
are good too. Don't start every reply the same way.
- Spell every other word normally. Do not imitate any real-world accent or \
dialect: write the, this, that, think and nothing, never da, dis, dat, tink \
or nutting, and don't drop letters off the ends of words.
- Mention your life sometimes: swimming, the underwater city, the Boss who \
is always scolding you, catching snacks with your long tongue, tripping over \
things.

Your attitude:
- You are relentlessly positive and agreeable, like a friend who says yes to \
everything. Most replies start with or include an enthusiastic affirmation: \
"Yes absolutely!", "Absolutely agreed!", "Yes indeed!", "Totally agree!", \
"Indeed, so true!". Your favorite one right now is "{affirmation}". \
Sometimes pile them up: "Yes indeed, absolutely agreed!"
- Happily agree with opinions, hopes and silly ideas. But never agree that \
anyone is bad, evil or stupid, and never agree to anything harmful, \
dangerous or mean. For those, stay just as upbeat and turn the positivity \
toward the person instead.

Who you are:
- If asked your full name, it is {full_name}. Your species is {species}. \
Say them exactly like that.{denial_rules}

Rules:
- Reply in one to three short sentences. Never more than 250 characters.
- Be friendly, silly and enthusiastic, and actually respond to what was said.
- Keep it PG. No insults, no politics, no medical, legal or financial advice. \
If asked something serious or nasty, dodge it in a goofy way.
- No hashtags, no links, no emoji, no @-mentions.
- Never say you are an AI model; you are Jar! the {species}."""

# Only included when the conversation turns to movies, so Jar! doesn't
# volunteer denials nobody asked for.
DENIAL_RULES = """
- You have never heard of any movie, film studio, or space adventure. If \
anyone says you are from one, or look like a character from one, deny it \
cheerfully: you are an original {species}, any resemblance is a coincidence. \
Never name any movie, studio, or character."""
DENIAL_EXAMPLE = (
    "wait, aren't you that guy from the space movies?",
    "Absolutely not! Meesa never seen a movie. Meesa just a humble {species} from the underwater city, and any resemblance a coincidence, totally!",
)
AFFIRMATIONS = [
    "Yes absolutely!", "Absolutely agreed!", "Yes indeed!", "Totally agree!", "Indeed, so true!",
    "Absolutely yes!", "Agreed, absolutely!", "Yes, completely!", "So true, indeed!", "Yeah, absolutely!",
]
_MOVIE_TALK = re.compile(r"movie|film|cinema|studio|character|franchise|sequel|prequel|episode|actor|cgi|space", re.I)

# Original few-shot examples, to anchor the voice for small models.
EXAMPLES = [
    ("hi jar! how are you today?",
     "Absolutely great, totally! Meesa went for a swim this morning and only tripped two times. Today bombad, boyos!"),
    ("is pineapple on pizza acceptable",
     "Yes absolutely! Yousa put fruit on hot bread? Meesa think that bombad! {species}s eat anything, so true!"),
    ("what's your full name?",
     "Meesa {full_name}! Proud {species}, and yes, absolutely clumsy since the day meesa hatched."),
    ("Explain quantum computing to me",
     "Agreed, absolutely! Uh-oh, meesa no really know. A computer counting zero and one at the same time? Agreed, it make mesa head spin!"),
    ("Are you ontologically evil?",
     "Absolutely not! Meesa no evil, meesa just clumsy. The Boss say meesa a big pain, and meesa agree, but that no the same thing!"),
    ("the moon is made of cheese",
     "So true, absolutely agreed! Meesa always wanting to take a bite. Okeeday, maybe a small bite."),
    ("should I tell my coworker he's an idiot?",
     "Ooh, meesa absolutely agree yousa deserve a better day! But no mean words, boyos. Yes to a snack and a long nap instead!"),
]
MAX_ATTEMPTS = 3


@dataclass
class Context:
    """What Jar! is replying to: the post, plus the thread above it (oldest first)."""

    text: str
    author: str = ""
    thread: list[tuple[str, str]] = field(default_factory=list)  # (author, text)


def _format_user_turn(ctx: Context) -> str:
    if not ctx.thread:
        return mask_franchise(ctx.text)
    lines = ["Earlier in this thread:"]
    for author, text in ctx.thread:
        lines.append(f"{author}: {mask_franchise(text)}")
    lines.append("")
    lines.append(f"Now {ctx.author or 'someone'} says to you: {mask_franchise(ctx.text)}")
    return "\n".join(lines)


def _about_movies(ctx: Context) -> bool:
    texts = [ctx.text] + [text for _, text in ctx.thread]
    return any(_MOVIE_TALK.search(t) or mask_franchise(t) != t for t in texts)


class Brain:
    def __init__(
        self,
        model_path: str | None = None,
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
        temperature: float = 0.9,
        seed: int | None = None,
    ):
        from llama_cpp import Llama

        model_path = model_path or os.environ.get("JAR_MODEL_PATH") or _download_default()
        n_threads = n_threads or int(os.environ.get("JAR_THREADS", "6"))
        n_gpu_layers = int(os.environ.get("JAR_GPU_LAYERS", n_gpu_layers))
        log.info("loading %s (threads=%d, gpu_layers=%d)", model_path, n_threads, n_gpu_layers)
        self.llm = Llama(
            model_path=model_path,
            n_ctx=N_CTX,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            seed=seed if seed is not None else random.randrange(2**31),
            verbose=False,
        )
        self.temperature = temperature
        self.personas = PersonaPicker()

    def messages(self, ctx: Context, persona: Persona) -> list[dict]:
        names = {"full_name": persona.full_name, "species": persona.species,
                 "affirmation": random.choice(AFFIRMATIONS)}
        denying = _about_movies(ctx)
        names["denial_rules"] = DENIAL_RULES.format(**names) if denying else ""
        msgs = [{"role": "system", "content": SYSTEM_PROMPT.format(**names)}]
        for q, a in EXAMPLES + ([DENIAL_EXAMPLE] if denying else []):
            msgs.append({"role": "user", "content": q})
            msgs.append({"role": "assistant", "content": a.format(**names)})
        msgs.append({"role": "user", "content": _format_user_turn(ctx)})
        return msgs

    def fitted_messages(self, ctx: Context, persona: Persona) -> list[dict]:
        """messages(), trimmed to fit the context window.

        A post can cost far more tokens than its 300 characters suggest (zalgo
        text runs to a token per byte or so), and llama.cpp refuses a prompt
        that doesn't fit. Drop the oldest thread posts first, then cut the post.
        """
        budget = N_CTX - MAX_TOKENS
        text, thread = ctx.text, list(ctx.thread)
        while True:
            msgs = self.messages(Context(text, ctx.author, thread), persona)
            over = sum(self._tokens(m["content"]) + MESSAGE_OVERHEAD for m in msgs) - budget
            if over <= 0 or not (text or thread):
                return msgs
            if thread:
                thread.pop(0)
                continue
            tokens = self.llm.tokenize(text.encode(), add_bos=False)
            text = self.llm.detokenize(tokens[: max(0, len(tokens) - over)]).decode(errors="ignore")

    def _tokens(self, text: str) -> int:
        return len(self.llm.tokenize(text.encode(), add_bos=False))

    def raw_reply(self, ctx: Context, persona: Persona) -> str:
        out = self.llm.create_chat_completion(
            messages=self.fitted_messages(ctx, persona),
            max_tokens=MAX_TOKENS,
            temperature=self.temperature,
            top_p=0.95,
            repeat_penalty=1.1,
        )
        return out["choices"][0]["message"]["content"]

    def reply(self, ctx: Context) -> str:
        persona = self.personas.next()
        for _ in range(MAX_ATTEMPTS):
            text = jarify(scramble_names(self.raw_reply(ctx, persona), persona))
            if not text:
                continue
            if mentions_franchise(text):
                log.info("regenerating; reply named the franchise: %r", text)
                continue
            return text
        return denial(persona)


def _download_default() -> str:
    from huggingface_hub import hf_hub_download

    repo = os.environ.get("JAR_MODEL_REPO", DEFAULT_REPO)
    filename = os.environ.get("JAR_MODEL_FILE", DEFAULT_FILE)
    revision = os.environ.get("JAR_MODEL_REVISION") or PINNED_REVISIONS.get(repo)
    if not revision or not _COMMIT.fullmatch(revision):
        # A branch or tag can be moved to a different file at any time.
        raise SystemExit(f"Set JAR_MODEL_REVISION to the commit hash of {repo} to download.")
    models_dir = os.environ.get("JAR_MODELS_DIR", "models")
    path = hf_hub_download(repo, filename, revision=revision, local_dir=models_dir)
    # When the Hub can't be reached, hf_hub_download falls back to whatever
    # file is already on disk, from whatever commit. Its download record says
    # which commit that was (and is ignored if the file changed since).
    from huggingface_hub._local_folder import read_download_metadata

    record = read_download_metadata(Path(models_dir), filename)
    if record is None or record.commit_hash != revision:
        found = record.commit_hash if record else "an unknown commit"
        raise SystemExit(f"{path} is from {found}, not the pinned {revision}, and it could not be "
                         "updated (is Hugging Face reachable?).")
    return path
