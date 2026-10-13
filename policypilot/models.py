from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentPage:
    document_name: str
    page_number: int
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    document_name: str
    page_number: int
    chunk_index: int

    @property
    def metadata(self) -> dict[str, str | int]:
        return {
            "text": self.text,
            "document_name": self.document_name,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
        }


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[RetrievedChunk]
    grounded: bool
