from pathlib import Path

from policypilot.config import Settings
from policypilot.history import ChatHistory
from policypilot.models import Chunk, DocumentPage, RetrievedChunk
from policypilot.pipeline import NO_CONTEXT, RAGPipeline


class FakeEmbedder:
    def embed(self, texts):
        return [[float(len(text)), 1.0] for text in texts]


class FakeGenerator:
    def __init__(self):
        self.prompt = ""

    def generate(self, instructions, prompt):
        self.prompt = prompt
        return "Refunds are available within 30 days. [Source 1]"


class FakeStore:
    def __init__(self, results=None):
        self.results = results or []
        self.upserts = []

    def upsert(self, chunks, vectors):
        self.upserts.append((list(chunks), list(vectors)))

    def query(self, vector, top_k):
        return self.results[:top_k]


def make_pipeline(tmp_path: Path, results=None):
    settings = Settings(sqlite_path=tmp_path / "history.db", chunk_size=30, chunk_overlap=5,
                        top_k=3, min_score=0.5, openai_api_key=None, pinecone_api_key=None,
                        pinecone_index_name=None)
    generator = FakeGenerator()
    return RAGPipeline(settings, FakeEmbedder(), generator, FakeStore(results), ChatHistory(settings.sqlite_path)), generator


def test_index_pages_batches_chunks_without_live_services(tmp_path):
    pipeline, _ = make_pipeline(tmp_path)
    indexed = pipeline.index_pages([DocumentPage("policy.pdf", 1, "A refund policy sentence. " * 10)])

    assert indexed > 1
    assert sum(len(batch[0]) for batch in pipeline.store.upserts) == indexed


def test_missing_context_returns_safe_message_and_saves_history(tmp_path):
    pipeline, _ = make_pipeline(tmp_path, [])
    answer = pipeline.ask("chat-1", "What is the refund period?")

    assert answer.text == NO_CONTEXT
    assert not answer.grounded
    assert pipeline.history.recent("chat-1") == [("user", "What is the refund period?"), ("assistant", NO_CONTEXT)]


def test_grounded_answer_includes_context_and_citation(tmp_path):
    chunk = Chunk("x", "Refunds are available within 30 days.", "refund.pdf", 2, 0)
    pipeline, generator = make_pipeline(tmp_path, [RetrievedChunk(chunk, 0.93)])
    answer = pipeline.ask("chat-2", "Can I get a refund?")

    assert answer.grounded and answer.citations[0].chunk.page_number == 2
    assert "refund.pdf, page 2" in generator.prompt
