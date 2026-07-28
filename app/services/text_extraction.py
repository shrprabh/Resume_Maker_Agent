"""Extract plain text from uploaded resume files.

Returns (text, status) where status is "ok" | "failed" | "unsupported".
Parsers are imported lazily so the app starts fast and only pays for the
formats that actually show up.
"""

import io

# Scanned/image-only PDFs extract as empty text — callers treat that as
# "failed" (there is no OCR in this pipeline).
MAX_EXTRACTED_CHARS = 50_000


def extract_text(filename: str, content: bytes) -> tuple[str, str]:
    name = (filename or "").lower()
    try:
        if name.endswith(".pdf"):
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif name.endswith(".docx"):
            from docx import Document

            doc = Document(io.BytesIO(content))
            text = "\n".join(p.text for p in doc.paragraphs)
        elif name.endswith((".txt", ".md")):
            text = content.decode("utf-8", errors="replace")
        else:
            return "", "unsupported"
    except Exception:
        return "", "failed"

    text = text.strip()
    if not text:
        return "", "failed"
    return text[:MAX_EXTRACTED_CHARS], "ok"
