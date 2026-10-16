# Jar!

A Bluesky bot in the spirit of [Jev!](https://bsky.app/profile/jevbot.bsky.social), except it
is a clumsy, relentlessly positive amphibian with its own way of talking. Tag it and it replies:

> **@someone:** what's nine plus ten?
> **Jar!:** Yes indeed! Nine plus ten is nineteen. Meesa count with long tongue, okeeday!

## How it works

1. **Notifications.** `jarbot run` polls the account's notifications every 20 s for mentions,
   quotes, and replies to Jar!'s own posts ([jarbot/bot.py](jarbot/bot.py)). On startup and
   every 10 minutes it also sweeps back 24 hours, to catch up on anything it missed while it was
   down. The sweep also searches post text for `@handle`, because a post that names Jar! without
   a real mention link (for example, a pasted handle) never produces a notification.
2. **Context.** It loads up to 6 posts above the mention in the thread, so replies follow the
   conversation.
3. **Generation.** A small open model from Hugging Face writes the reply locally through
   llama.cpp ([jarbot/brain.py](jarbot/brain.py)). The default is
   `unsloth/Qwen3-4B-Instruct-2507-GGUF`, Q4_K_M quantization, about 2.5 GB, running on the CPU.
   It works from a persona prompt plus a few original example exchanges.
   The persona is in [jarbot/persona.py](jarbot/persona.py). Jar!'s full name is "Jar Bot"
   plus a surname that changes on every reply (Bonks, Banks, Bunkles…), and its species
   changes too (Gangun, Gungian, Goongan…). It has never heard of any movie it might remind you
   of. Movie and studio names in incoming posts are replaced with stand-ins like "that space
   movie" before the model sees them. If a reply names one anyway, the bot generates it again,
   and after three tries it falls back to a canned denial.
4. **Speech pass.** A deterministic pass ([jarbot/speech.py](jarbot/speech.py)) enforces Jar!'s
   invented quirks. I/me becomes *meesa*, you becomes *yousa*, and *am/is/are* after them is
   dropped. Not and don't become *no*, okay becomes *okeeday*, and guys becomes *boyos*. Jar!
   never imitates a real-world accent: if the model drifts into one, the pass changes the
   spelling back, so *da* becomes *the*, *tink* becomes *think*, and *swimmin'* becomes
   *swimming*.
   It also strips @-mentions, links, hashtags and emoji, and fits the reply in one post.
5. **Reply.** Jar! posts the reply in the thread, likes the post it answered, and records the
   post in `state/jarbot.db`, so it never answers the same post twice.

Guardrails: Jar! doesn't reply to accounts labelled as bots, which prevents bot-to-bot loops. It
answers each author at most 10 times an hour and posts at most 60 replies an hour overall
(replies that fail to post count too, since they still cost a turn of the model). It
stops after 10 replies in one thread, skips threads whose reply settings shut it out, and ignores
mentions older than 24 hours. All of these can be changed in `.env`.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'      # builds llama-cpp-python (CPU) from source
cp .env.example .env                    # then fill in handle + app password
```

The model downloads into `./models` the first time it's used. It is pinned to a fixed commit of
the Hugging Face repo (`PINNED_REVISIONS` in [jarbot/brain.py](jarbot/brain.py)), so an upstream
change never reaches the bot until you update the pin.

Try it locally, no Bluesky needed:

```bash
.venv/bin/jarbot say "is pineapple on pizza ok?"
.venv/bin/jarbot chat
```

On Bluesky:

```bash
.venv/bin/jarbot profile                # display name "Jar!", bio, and the "bot" self-label
.venv/bin/jarbot post                   # optional intro post
.venv/bin/jarbot run --dry-run --once   # show what it would reply to, post nothing
.venv/bin/jarbot run                    # go live
```

To run it in the background with systemd:

```bash
mkdir -p ~/.config/systemd/user && cp deploy/jarbot.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now jarbot
journalctl --user -u jarbot -f
```

To keep it running while you're logged out, run `loginctl enable-linger $USER`.

## Notes

- A reply takes about 5–50 s on this iMac's CPU: the CPU throttles under load, and the GPU is busy
  with other work. To use the GPU, rebuild the wheel with
  `CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall --no-cache-dir llama-cpp-python`
  and set `JAR_GPU_LAYERS=99`.
- To try a different model, set `JAR_MODEL_REPO` and `JAR_MODEL_FILE` to any chat GGUF on
  Hugging Face, or `JAR_MODEL_PATH` to a local file. `unsloth/gemma-3-4b-it-GGUF` is pinned
  already; for any other repo, also set `JAR_MODEL_REVISION` to the commit hash to download.
- Tests: `.venv/bin/pytest`. They use a fake Bluesky client and a fake model.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
