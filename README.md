# PolicyPilot — Company Policy RAG Chatbot

PolicyPilot is a small, production-minded **Retrieval-Augmented Generation (RAG)** chatbot for company policy PDFs (refund, return, payment, order policies, and similar). Upload PDFs in a Streamlit UI, and ask questions that are answered **only from the uploaded documents**, with document-and-page citations. If nothing relevant is found, the assistant says so instead of guessing.

**Stack:** Python 3.10+ · Streamlit · PyMuPDF · OpenAI (embeddings + Responses API) · Pinecone · SQLite · pytest

## How it works

```text
Ingestion
  PDF upload → PyMuPDF text extraction (per page)
            → custom word-boundary chunking with overlap (page metadata kept)
            → OpenAI embeddings (text-embedding-3-small, 1536 dims)
            → Pinecone upsert (stable content-hash IDs, namespace-isolated)

Question answering
  User question → OpenAI embedding → Pinecone top-k similarity search
               → drop matches below MIN_RETRIEVAL_SCORE
               → grounded answer from OpenAI chat model (+ recent chat history)
               → citations: "document.pdf, p. N"
               ↘ SQLite stores per-conversation history across app restarts
```

## Features

- **PDF-only upload flow** with clear errors for corrupt or scanned/no-text PDFs.
- **Custom chunking** (no LangChain): character windows that snap to word boundaries, with overlap and page-level metadata.
- **Grounded generation**: the model is instructed to answer only from retrieved context and to cite `[Source N]` labels.
- **No-context guard**: a safe fallback answer when no chunk clears the similarity threshold.
- **Citations** shown under every answer (document name + page number).
- **Conversation memory** via SQLite, keyed by a per-browser-session conversation ID.
- **Configurable, low-cost defaults**: models, chunk size, `top_k`, and threshold all come from `.env`.
- **Friendly error handling** for auth, missing index, rate-limit/credit, and connectivity failures — without leaking config values.
- **Offline unit tests** using fake embedder/generator/store, so no API keys or paid calls are needed to run the suite.
- **Secrets never in code**: credentials are read from a local `.env` that is git-ignored.

## Project structure

```text
.
├── app.py                    # Streamlit UI (upload, process, chat, citations)
├── policypilot/
│   ├── config.py             # Settings loaded from .env with sane defaults
│   ├── pdf_reader.py         # PyMuPDF page extraction + PDFExtractionError
│   ├── chunking.py           # Page-aware, overlapping, word-boundary chunker
│   ├── models.py             # DocumentPage, Chunk, RetrievedChunk, Answer dataclasses
│   ├── services.py           # Embedder/Generator/VectorStore protocols + OpenAI & Pinecone impls
│   ├── pipeline.py           # RAGPipeline: index_pages() and ask()
│   ├── history.py            # SQLite-backed chat history
│   └── errors.py             # friendly_service_error() mapping
├── tests/
│   ├── test_chunking.py
│   ├── test_errors.py
│   └── test_pipeline.py
├── data/                     # history.db is created here at runtime (git-ignored)
├── .env.example              # Copy to .env and fill in keys
├── requirements.txt
└── README.md
```

## Setup

### 1. Prerequisites

- Python 3.10 or newer
- An [OpenAI API key](https://platform.openai.com/)
- A [Pinecone](https://www.pinecone.io/) account and API key

### 2. Create the Pinecone index

Create a **serverless** index before uploading any documents:

| Setting   | Value                                              |
|-----------|----------------------------------------------------|
| Dimension | `1536` (matches `text-embedding-3-small`)          |
| Metric    | `cosine`                                           |
| Name      | anything — put it in `PINECONE_INDEX_NAME`         |

If you switch `OPENAI_EMBEDDING_MODEL`, create an index whose dimension matches the new model.

### 3. Install

```bash
git clone https://github.com/Dpk021/Policypilot.git
cd Policypilot
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Configure

```bash
cp .env.example .env
```

Then open `.env` and fill in:

```dotenv
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=policypilot
```

`.env` is listed in `.gitignore` — never commit it.

### 5. Run

```bash
streamlit run app.py
```

In the browser:

1. Use the sidebar to upload one or more **text-based** PDFs.
2. Click **Process uploaded PDFs** — you'll see how many chunks were indexed.
3. Ask questions in the chat box. Each answer shows its sources underneath.
4. **Start a new chat** resets the conversation (history for the old one stays in SQLite).

## Configuration reference

All values are read from `.env` by `policypilot/config.py`.

| Variable                 | Default                  | Purpose |
|--------------------------|--------------------------|---------|
| `OPENAI_API_KEY`         | — (required)             | OpenAI credentials. |
| `PINECONE_API_KEY`       | — (required)             | Pinecone credentials. |
| `PINECONE_INDEX_NAME`    | — (required)             | Name of the 1536-dim cosine index. |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model (dimension must match the index). |
| `OPENAI_CHAT_MODEL`      | `gpt-5-nano`             | Answer-generation model; swap without touching code. |
| `PINECONE_NAMESPACE`     | `policypilot`            | Logical partition inside the index. |
| `CHUNK_SIZE`             | `1000`                   | Max characters per chunk. |
| `CHUNK_OVERLAP`          | `150`                    | Characters shared between consecutive chunks. |
| `TOP_K`                  | `5`                      | Number of candidates retrieved from Pinecone. |
| `MIN_RETRIEVAL_SCORE`    | `0.45`                   | Cosine-score floor; below it the no-context answer is returned. |
| `SQLITE_DB_PATH`         | `data/history.db`        | Where chat history is stored. |

Tune `TOP_K` and `MIN_RETRIEVAL_SCORE` against your own policy documents — the defaults are deliberately conservative for a small API budget.

## Testing

```bash
pytest -q
```

The suite covers:

- chunk sizing, overlap, word-boundary snapping, and page metadata
- batched indexing through the pipeline
- grounded retrieval and the no-context guard
- SQLite history persistence
- error-message mapping for provider failures

Tests use in-memory fakes for the embedder, generator, and vector store, so they run fully offline and never call OpenAI or Pinecone.

## Design notes

- **Protocols over SDKs.** `Embedder`, `Generator`, and `VectorStore` are `typing.Protocol`s, so the pipeline is testable with fakes and the OpenAI/Pinecone SDKs are imported lazily only when real services are constructed.
- **Deterministic vector IDs.** Each chunk ID is a SHA-256 of `document|page|index|text`, so re-processing the same PDF overwrites rather than duplicates vectors.
- **Citations travel with vectors.** Document name, page number, chunk index, and text are stored as Pinecone metadata, which is what makes page-level citations possible at answer time.
- **Grounding is a threshold, not a guarantee.** The score floor cuts down unsupported answers but doesn't eliminate hallucination risk; a production deployment should add an evaluation set, observability, authentication, and access control.

## Roadmap / ideas

- OCR fallback for scanned PDFs
- Hybrid (BM25 + vector) retrieval and a reranker
- Per-document delete / re-index from the UI
- An evaluation harness with gold Q&A pairs
