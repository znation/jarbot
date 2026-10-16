"""Bookkeeping: which posts Jar! has handled, so it never replies twice."""

import os
import sqlite3
import time
from pathlib import Path


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # Private from the moment it exists: it holds other people's post text.
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        os.fchmod(fd, 0o600)  # an older file may have looser permissions
        os.close(fd)
        self.db = sqlite3.connect(path)
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS handled (
                uri TEXT PRIMARY KEY,
                author_did TEXT NOT NULL,
                reason TEXT NOT NULL,
                status TEXT NOT NULL,      -- replied | skipped:<why> | error:<why>
                prompt TEXT,
                reply TEXT,
                reply_uri TEXT,
                handled_at REAL NOT NULL
            )"""
        )
        self.db.commit()

    def seen(self, uri: str) -> bool:
        return self.db.execute("SELECT 1 FROM handled WHERE uri = ?", (uri,)).fetchone() is not None

    def record(self, uri: str, author_did: str, reason: str, status: str,
               prompt: str | None = None, reply: str | None = None, reply_uri: str | None = None) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO handled VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (uri, author_did, reason, status, prompt, reply, reply_uri, time.time()),
        )
        self.db.commit()

    def replies_since(self, seconds: float, author_did: str | None = None) -> int:
        """Replies generated recently, posted or not.

        A reply that failed to post still cost a turn of the model, so it
        counts: otherwise posts Jar! can't answer (in a thread with replies
        turned off, say) could keep the CPU busy without limit.
        """
        q = ("SELECT COUNT(*) FROM handled WHERE (status = 'replied' OR status LIKE 'error:%')"
             " AND handled_at > ?")
        args: list = [time.time() - seconds]
        if author_did:
            q += " AND author_did = ?"
            args.append(author_did)
        return self.db.execute(q, args).fetchone()[0]
