from __future__ import annotations

import hashlib
import re

from .models import Chunk, DocumentPage


def _windows(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("chunk_size must be positive and chunk_overlap must be smaller than chunk_size")
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    result: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + size, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(" ", start, end)
            if boundary > start + (size // 2):
                end = boundary
        piece = normalized[start:end].strip()
        if piece:
            result.append(piece)
        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)
    return result


def chunk_pages(pages: list[DocumentPage], chunk_size: int = 1_000, overlap: int = 150) -> list[Chunk]:
    """Custom, page-aware character chunking with a readable-word boundary and overlap."""
    chunks: list[Chunk] = []
    for page in pages:
        for local_index, text in enumerate(_windows(page.text, chunk_size, overlap)):
            digest = hashlib.sha256(
                f"{page.document_name}|{page.page_number}|{local_index}|{text}".encode()
            ).hexdigest()[:24]
            chunks.append(Chunk(digest, text, page.document_name, page.page_number, local_index))
    return chunks
