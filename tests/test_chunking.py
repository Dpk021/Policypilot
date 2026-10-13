import pytest

from policypilot.chunking import chunk_pages
from policypilot.models import DocumentPage


def test_chunking_preserves_document_and_page_metadata_and_overlap():
    page = DocumentPage("refund.pdf", 3, "one two three four five six seven eight nine ten")
    chunks = chunk_pages([page], chunk_size=20, overlap=5)

    assert len(chunks) >= 2
    assert all(chunk.document_name == "refund.pdf" and chunk.page_number == 3 for chunk in chunks)
    assert chunks[0].id != chunks[1].id
    assert any(word in chunks[1].text.split() for word in chunks[0].text.split()[-2:])


def test_chunking_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_pages([DocumentPage("a.pdf", 1, "text")], chunk_size=10, overlap=10)
