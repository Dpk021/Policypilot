from __future__ import annotations

from typing import Protocol, Sequence

from .config import Settings
from .models import Chunk, RetrievedChunk


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class Generator(Protocol):
    def generate(self, instructions: str, prompt: str) -> str: ...


class VectorStore(Protocol):
    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None: ...
    def query(self, vector: Sequence[float], top_k: int) -> list[RetrievedChunk]: ...


class OpenAIService:
    def __init__(self, settings: Settings):
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required.")
        # Lazy imports keep offline unit tests independent of paid-service SDKs.
        from openai import OpenAI
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.embedding_model = settings.embedding_model
        self.chat_model = settings.chat_model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.embedding_model, input=list(texts))
        return [item.embedding for item in response.data]

    def generate(self, instructions: str, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.chat_model,
            instructions=instructions,
            input=prompt,
        )
        return response.output_text.strip()


class PineconeStore:
    def __init__(self, settings: Settings):
        if not settings.pinecone_api_key or not settings.pinecone_index_name:
            raise ValueError("PINECONE_API_KEY and PINECONE_INDEX_NAME are required.")
        from pinecone import Pinecone
        self.index = Pinecone(api_key=settings.pinecone_api_key).Index(settings.pinecone_index_name)
        self.namespace = settings.pinecone_namespace

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        records = [
            {"id": chunk.id, "values": vector, "metadata": chunk.metadata}
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        if records:
            self.index.upsert(vectors=records, namespace=self.namespace)

    def query(self, vector: Sequence[float], top_k: int) -> list[RetrievedChunk]:
        response = self.index.query(vector=list(vector), top_k=top_k, include_metadata=True,
                                    namespace=self.namespace)
        results: list[RetrievedChunk] = []
        for match in response.matches:
            metadata = match.metadata or {}
            text = metadata.get("text")
            if not text:
                continue
            chunk = Chunk(
                id=str(match.id), text=str(text), document_name=str(metadata.get("document_name", "Unknown")),
                page_number=int(metadata.get("page_number", 0)), chunk_index=int(metadata.get("chunk_index", 0)),
            )
            results.append(RetrievedChunk(chunk, float(match.score)))
        return results
