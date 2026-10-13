from __future__ import annotations

import pymupdf

from .models import DocumentPage


class PDFExtractionError(ValueError):
    pass


def extract_pdf_pages(pdf_bytes: bytes, document_name: str) -> list[DocumentPage]:
    """Extract non-empty text from each PDF page while retaining its citation page."""
    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:  # PyMuPDF provides several parser exception types.
        raise PDFExtractionError(f"Could not read {document_name} as a PDF.") from exc
    try:
        pages = [
            DocumentPage(document_name, number, page.get_text("text").strip())
            for number, page in enumerate(document, start=1)
            if page.get_text("text").strip()
        ]
    finally:
        document.close()
    if not pages:
        raise PDFExtractionError(f"No selectable text was found in {document_name}. OCR may be required.")
    return pages
