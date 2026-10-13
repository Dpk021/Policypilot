from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    pinecone_api_key: str | None
    pinecone_index_name: str | None
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-5-nano"
    pinecone_namespace: str = "policypilot"
    sqlite_path: Path = ROOT / "data" / "history.db"
    chunk_size: int = 1_000
    chunk_overlap: int = 150
    top_k: int = 5
    min_score: float = 0.45

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
            pinecone_index_name=os.getenv("PINECONE_INDEX_NAME"),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-5-nano"),
            pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "policypilot"),
            sqlite_path=Path(os.getenv("SQLITE_DB_PATH", str(ROOT / "data" / "history.db"))),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
            top_k=int(os.getenv("TOP_K", "5")),
            min_score=float(os.getenv("MIN_RETRIEVAL_SCORE", "0.45")),
        )

    def require_ingestion_credentials(self) -> None:
        missing = [name for name, value in {
            "OPENAI_API_KEY": self.openai_api_key,
            "PINECONE_API_KEY": self.pinecone_api_key,
            "PINECONE_INDEX_NAME": self.pinecone_index_name,
        }.items() if not value]
        if missing:
            raise ValueError("Missing required environment variable(s): " + ", ".join(missing))
