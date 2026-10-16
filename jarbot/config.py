"""Settings, read from the environment and an optional .env file."""

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env reader: KEY=VALUE lines; real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


@dataclass
class Settings:
    handle: str
    app_password: str
    state_dir: Path
    poll_seconds: float
    max_age_hours: float
    sweep_minutes: float
    max_per_author_per_hour: int
    max_per_hour: int
    max_replies_per_thread: int
    reply_to_bots: bool

    def require_login(self) -> None:
        if not self.handle or not self.app_password:
            raise SystemExit("Set BSKY_HANDLE and BSKY_APP_PASSWORD (see .env.example).")

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        e = os.environ.get
        return cls(
            handle=e("BSKY_HANDLE", ""),
            app_password=e("BSKY_APP_PASSWORD", ""),
            state_dir=Path(e("JAR_STATE_DIR", "state")),
            poll_seconds=float(e("JAR_POLL_SECONDS", "20")),
            max_age_hours=float(e("JAR_MAX_AGE_HOURS", "24")),
            sweep_minutes=float(e("JAR_SWEEP_MINUTES", "10")),
            max_per_author_per_hour=int(e("JAR_MAX_PER_AUTHOR_PER_HOUR", "10")),
            max_per_hour=int(e("JAR_MAX_PER_HOUR", "60")),
            max_replies_per_thread=int(e("JAR_MAX_REPLIES_PER_THREAD", "10")),
            reply_to_bots=e("JAR_REPLY_TO_BOTS", "0") == "1",
        )
