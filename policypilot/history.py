from __future__ import annotations

import sqlite3
from pathlib import Path


class ChatHistory:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY, conversation_id TEXT NOT NULL, role TEXT NOT NULL,
                content TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def add(self, conversation_id: str, role: str, content: str) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError("role must be 'user' or 'assistant'")
        with self._connect() as conn:
            conn.execute("INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                         (conversation_id, role, content))

    def recent(self, conversation_id: str, limit: int = 6) -> list[tuple[str, str]]:
        with self._connect() as conn:
            rows = conn.execute("""SELECT role, content FROM messages WHERE conversation_id = ?
                ORDER BY id DESC LIMIT ?""", (conversation_id, limit)).fetchall()
        return list(reversed(rows))

    def clear(self, conversation_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
