from pathlib import Path
from typing import Tuple
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path: Path) -> Tuple[str, int]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    text = "\n".join(pages).strip()
    return text, len(reader.pages)
