"""Utility functions."""

import mimetypes
import tempfile
from datetime import datetime
from pathlib import Path


def generate_task_id() -> str:
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def load_image(path: str) -> tuple[bytes, str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    mime, _ = mimetypes.guess_type(str(p))
    return p.read_bytes(), mime or "image/jpeg"


def load_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return p.read_text(encoding="utf-8")


def load_document(path: Path) -> tuple[bytes, str, Path | None]:
    """
    Load document file. Converts DOCX to PDF if needed.
    Returns (bytes, mime_type, temp_file_to_delete).
    """
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return path.read_bytes(), "application/pdf", None

    if suffix == ".txt":
        return path.read_bytes(), "text/plain", None

    if suffix in {".docx", ".doc"}:
        temp_pdf = _convert_docx_to_pdf(path)
        return temp_pdf.read_bytes(), "application/pdf", temp_pdf

    return path.read_bytes(), "application/octet-stream", None


def _convert_docx_to_pdf(docx_path: Path) -> Path:
    """Convert DOCX to PDF, returns temp file path."""
    from docx import Document
    from fpdf import FPDF

    doc = Document(str(docx_path))
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Use built-in font that supports basic characters
    pdf.set_font("Helvetica", size=10)
    
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            # Encode to latin-1 compatible, replace unsupported chars
            safe_text = text.encode("latin-1", errors="replace").decode("latin-1")
            pdf.multi_cell(0, 5, safe_text)
            pdf.ln(2)

    # Create temp file
    temp_file = Path(tempfile.mktemp(suffix=".pdf"))
    pdf.output(str(temp_file))
    return temp_file
