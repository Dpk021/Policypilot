from __future__ import annotations

from itertools import islice
from typing import Iterable

from .chunking import chunk_pages
from .config import Settings
from .history import ChatHistory
from .models import Answer, Chunk, DocumentPage, RetrievedChunk
from .services import Embedder, Generator, VectorStore

NO_CONTEXT = "I don’t have enough information in the uploaded documents to answer that. Please upload a relevant policy or rephrase your question."


def _batches(items: list[Chunk], size: int = 96) -> Iterable[list[Chunk]]:
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch


class RAGPipeline:
    def __init__(self, settings: Settings, embedder: Embedder, generator: Generator,
                 store: VectorStore, history: ChatHistory):
        self.settings, self.embedder, self.generator, self.store, self.history = (
            settings, embedder, generator, store, history)

    def index_pages(self, pages: list[DocumentPage]) -> int:
        chunks = chunk_pages(pages, self.settings.chunk_size, self.settings.chunk_overlap)
        for batch in _batches(chunks):
            self.store.upsert(batch, self.embedder.embed([chunk.text for chunk in batch]))
        return len(chunks)

    def ask(self, conversation_id: str, question: str) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("Please enter a question.")
        matches = self.store.query(self.embedder.embed([question])[0], self.settings.top_k)
        citations = [match for match in matches if match.score >= self.settings.min_score]
        self.history.add(conversation_id, "user", question)
        if not citations:
            self.history.add(conversation_id, "assistant", NO_CONTEXT)
            return Answer(NO_CONTEXT, [], False)
        context = "\n\n".join(
            f"[Source {index}: {item.chunk.document_name}, page {item.chunk.page_number}]\n{item.chunk.text}"
            for index, item in enumerate(citations, start=1)
        )
        chat_history = "\n".join(f"{role.title()}: {content}" for role, content in self.history.recent(conversation_id))
        prompt = f"""Retrieved document context:\n{context}\n\nRecent conversation:\n{chat_history}\n\nUser question: {question}"""
        instructions = ("You are PolicyPilot, a company-policy assistant. Answer only from the retrieved "
                        "document context. If the context does not support the answer, say you do not know. "
                        "Do not invent policy details. Be concise and cite source labels like [Source 1].")
        answer_text = self.generator.generate(instructions, prompt)
        self.history.add(conversation_id, "assistant", answer_text)
        return Answer(answer_text, citations, True)
