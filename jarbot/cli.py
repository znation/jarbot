"""jarbot command line: talk to Jar! locally, or run it on Bluesky."""

import argparse
import logging
import sys

from .config import Settings

PROFILE_NAME = "Jar!"
PROFILE_BIO = (
    "Meesa Jar!, a small language model. Tag meesa and meesa reply, okeeday? Yes absolutely!\n\n"
    "Bot. Runs a small open model locally."
)
INTRO_POST = "Hello, boyos! Meesa Jar!, a small language model. Tag meesa and meesa reply. Yes indeed, absolutely!"


def _brain(args):
    from .brain import Brain

    return Brain(seed=args.seed)


def cmd_say(args) -> None:
    from .brain import Context

    brain = _brain(args)
    for _ in range(args.n):
        print(brain.reply(Context(text=" ".join(args.text))))


def cmd_chat(args) -> None:
    from .brain import Context

    brain = _brain(args)
    print("Talk to Jar! (Ctrl-D to quit)")
    thread: list[tuple[str, str]] = []
    while True:
        try:
            line = input("you> ").strip()
        except EOFError:
            print()
            return
        if not line:
            continue
        reply = brain.reply(Context(text=line, author="you", thread=thread[-6:]))
        print(f"Jar!> {reply}")
        thread += [("you", line), ("Jar! (you)", reply)]


def cmd_run(args) -> None:
    from .bot import Bot

    settings = Settings.from_env()
    settings.require_login()  # fail before spending time loading the model
    bot = Bot(settings, _brain(args), dry_run=args.dry_run)
    if args.once:
        bot.poll_once()
    else:
        bot.run_forever()


def cmd_profile(args) -> None:
    """Set display name and bio, and label the account as a bot (like Jev!)."""
    from .bot import login

    client = login(Settings.from_env(), reuse_saved=False, save=False)
    try:
        current = client.com.atproto.repo.get_record(
            {"repo": client.me.did, "collection": "app.bsky.actor.profile", "rkey": "self"}
        ).value
        profile = current.model_dump(by_alias=True, exclude_none=True) if hasattr(current, "model_dump") else dict(current)
    except Exception:
        profile = {}
    profile.update({
        "$type": "app.bsky.actor.profile",
        "displayName": args.name,
        "description": args.bio,
        "labels": {"$type": "com.atproto.label.defs#selfLabels", "values": [{"val": "bot"}]},
    })
    profile.pop("py_type", None)
    client.com.atproto.repo.put_record(
        {"repo": client.me.did, "collection": "app.bsky.actor.profile", "rkey": "self", "record": profile}
    )
    print(f"Updated profile of @{client.me.handle}: {args.name!r}, labelled as a bot.")


def cmd_post(args) -> None:
    from .bot import login

    client = login(Settings.from_env(), reuse_saved=False, save=False)
    post = client.send_post(text=args.text, langs=["en"])
    print(post.uri)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="jarbot", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--seed", type=int, default=None, help="sampling seed (default: random)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("say", help="generate a reply to some text, locally")
    s.add_argument("text", nargs="+")
    s.add_argument("-n", type=int, default=1, help="number of replies")
    s.set_defaults(func=cmd_say)

    s = sub.add_parser("chat", help="chat with Jar! in the terminal")
    s.set_defaults(func=cmd_chat)

    s = sub.add_parser("run", help="reply to mentions on Bluesky")
    s.add_argument("--dry-run", action="store_true", help="generate replies but don't post them")
    s.add_argument("--once", action="store_true", help="poll once and exit")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("profile", help="set the Bluesky profile and bot label")
    s.add_argument("--name", default=PROFILE_NAME)
    s.add_argument("--bio", default=PROFILE_BIO)
    s.set_defaults(func=cmd_profile)

    s = sub.add_parser("post", help="make a top-level post (e.g. an intro)")
    s.add_argument("text", nargs="?", default=INTRO_POST)
    s.set_defaults(func=cmd_post)

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args.func(args)


if __name__ == "__main__":
    main()
