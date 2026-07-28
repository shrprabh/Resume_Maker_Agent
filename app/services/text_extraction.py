"""Extract normalized, provenance-friendly text from uploaded source files.

``extract_text`` is the backwards-compatible API used by the resume router.
New callers can use ``extract_text_result`` to retain extraction metadata and
warnings that are useful when presenting a source/knowledge-base summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import io
import re
import unicodedata

# Scanned/image-only PDFs extract as empty text — callers treat that as
# "failed" (there is no OCR in this pipeline).
MAX_EXTRACTED_CHARS = 50_000
PDF_PAGE_SEPARATOR = "\n\n\f\n\n"
SPARSE_PDF_PAGE_CHARS = 40

_WORD_RE = re.compile(r"\b[\w]+(?:[’'-][\w]+)*\b", re.UNICODE)
_HEADING_CLEAN_RE = re.compile(r"[^a-z0-9&+ ]+")

_SECTION_HEADINGS = {
    "summary": "Summary",
    "professional summary": "Summary",
    "profile": "Summary",
    "objective": "Objective",
    "experience": "Experience",
    "professional experience": "Experience",
    "work experience": "Experience",
    "employment history": "Experience",
    "education": "Education",
    "academic background": "Education",
    "skills": "Skills",
    "technical skills": "Skills",
    "core competencies": "Skills",
    "projects": "Projects",
    "selected projects": "Projects",
    "certifications": "Certifications",
    "licenses & certifications": "Certifications",
    "licenses and certifications": "Certifications",
    "publications": "Publications",
    "research": "Research",
    "awards": "Awards",
    "honors": "Awards",
    "volunteer experience": "Volunteer Experience",
    "leadership": "Leadership",
    "professional affiliations": "Professional Affiliations",
}


@dataclass(slots=True)
class ExtractionResult:
    """Structured outcome for one uploaded source.

    ``pages`` is populated for PDFs. DOCX does not expose a reliable rendered
    page count, and plain-text formats have no page model, so it is ``None``
    for those formats.
    """

    filename: str
    kind: str
    status: str
    bytes: int
    pages: int | None
    extracted_chars: int
    words: int
    truncated: bool
    warnings: list[str] = field(default_factory=list)
    text: str = ""
    detected_sections: list[str] = field(default_factory=list)


def _file_kind(filename: str) -> str:
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        return "pdf"
    if name.endswith(".docx"):
        return "docx"
    if name.endswith(".txt"):
        return "txt"
    if name.endswith(".md"):
        return "md"
    return "unsupported"


def _normalize_text(text: str) -> tuple[str, int]:
    """Normalize extracted text without flattening its useful structure."""

    nul_count = text.count("\x00")
    text = text.replace("\x00", "")
    text = unicodedata.normalize("NFC", text)
    text = (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u2028", "\n")
        .replace("\u2029", "\n")
    )
    # A UTF-8 BOM is metadata rather than candidate content.
    text = text.lstrip("\ufeff")

    lines = [line.rstrip(" \t") for line in text.split("\n")]
    compacted: list[str] = []
    previous_blank = False
    for line in lines:
        blank = not line.strip()
        if blank and previous_blank:
            continue
        compacted.append("" if blank else line)
        previous_blank = blank

    return "\n".join(compacted).strip(" \t\n"), nul_count


def _detect_sections(text: str) -> list[str]:
    detected: list[str] = []
    for line in text.replace("\f", "\n").splitlines():
        heading = line.strip()
        # Resume headings are short and normally occupy their own line.
        if not heading or len(heading) > 45:
            continue
        heading = _HEADING_CLEAN_RE.sub("", heading.casefold().rstrip(":")).strip()
        canonical = _SECTION_HEADINGS.get(heading)
        if canonical and canonical not in detected:
            detected.append(canonical)
    return detected


def _result(
    *,
    filename: str,
    kind: str,
    byte_count: int,
    status: str,
    text: str = "",
    pages: int | None = None,
    truncated: bool = False,
    warnings: list[str] | None = None,
    extracted_chars: int | None = None,
) -> ExtractionResult:
    return ExtractionResult(
        filename=filename or "",
        kind=kind,
        status=status,
        bytes=byte_count,
        pages=pages,
        extracted_chars=(
            len(text) if extracted_chars is None else extracted_chars
        ),
        words=len(_WORD_RE.findall(text)),
        truncated=truncated,
        warnings=warnings or [],
        text=text,
        detected_sections=_detect_sections(text),
    )


def _extract_pdf(content: bytes) -> tuple[str, int, list[str], int]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(content))
    page_texts: list[str] = []
    sparse_pages: list[int] = []
    nul_count = 0

    for page_number, page in enumerate(reader.pages, start=1):
        normalized, page_nuls = _normalize_text(page.extract_text() or "")
        page_texts.append(normalized)
        nul_count += page_nuls
        visible_chars = len(re.sub(r"\s+", "", normalized))
        if visible_chars < SPARSE_PDF_PAGE_CHARS:
            sparse_pages.append(page_number)

    warnings: list[str] = []
    if sparse_pages:
        page_list = ", ".join(str(number) for number in sparse_pages)
        warnings.append(
            "Little or no selectable text was found on PDF "
            f"page(s) {page_list}; scanned pages may require OCR."
        )

    # Form feed is a conventional plain-text page boundary. Keeping it lets
    # downstream chunkers avoid merging the end of one page into the next.
    text = PDF_PAGE_SEPARATOR.join(page_texts).strip(" \t\n")
    return text, len(page_texts), warnings, nul_count


def _extract_docx(content: bytes) -> tuple[str, int]:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P

    document = Document(io.BytesIO(content))
    blocks: list[str] = []
    nul_count = 0

    # ``document.paragraphs`` omits table text and appending all tables later
    # destroys source order. Walking body XML preserves paragraphs and tables
    # in the same order the candidate sees in the DOCX.
    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            raw = Paragraph(child, document).text
            normalized, block_nuls = _normalize_text(raw)
            nul_count += block_nuls
            if normalized:
                blocks.append(normalized)
        elif isinstance(child, CT_Tbl):
            table = Table(child, document)
            table_lines: list[str] = []
            for row in table.rows:
                cells: list[str] = []
                seen_cells: set[int] = set()
                for cell in row.cells:
                    # Horizontally merged cells can appear more than once.
                    cell_id = id(cell._tc)
                    if cell_id in seen_cells:
                        continue
                    seen_cells.add(cell_id)
                    cell_text, cell_nuls = _normalize_text(cell.text)
                    nul_count += cell_nuls
                    cells.append(cell_text)
                row_text = "\t".join(cells).rstrip()
                if row_text:
                    table_lines.append(row_text)
            if table_lines:
                blocks.append("\n".join(table_lines))

    text, final_nuls = _normalize_text("\n\n".join(blocks))
    return text, nul_count + final_nuls


def _extract_utf8(content: bytes) -> tuple[str, list[str], int]:
    warnings: list[str] = []
    try:
        decoded = content.decode("utf-8")
    except UnicodeDecodeError:
        # Replacement is deliberately limited to malformed byte sequences;
        # we do not guess an arbitrary legacy encoding and silently corrupt
        # names, employers, or technical terms.
        decoded = content.decode("utf-8", errors="replace")
        warnings.append(
            "The file was not valid UTF-8; invalid byte sequences were "
            "replaced while extracting text."
        )
    text, nul_count = _normalize_text(decoded)
    return text, warnings, nul_count


def extract_text_result(filename: str, content: bytes) -> ExtractionResult:
    """Extract one source and return normalized text plus extraction metadata."""

    kind = _file_kind(filename)
    byte_count = len(content) if isinstance(content, (bytes, bytearray)) else 0
    if kind == "unsupported":
        return _result(
            filename=filename,
            kind=kind,
            byte_count=byte_count,
            status="unsupported",
            warnings=["Supported source formats are PDF, DOCX, TXT, and MD."],
        )

    if not isinstance(content, (bytes, bytearray)):
        return _result(
            filename=filename,
            kind=kind,
            byte_count=byte_count,
            status="failed",
            warnings=["The uploaded source did not contain readable bytes."],
        )

    content = bytes(content)
    pages: int | None = None
    warnings: list[str] = []
    nul_count = 0
    try:
        if kind == "pdf":
            text, pages, warnings, nul_count = _extract_pdf(content)
        elif kind == "docx":
            text, nul_count = _extract_docx(content)
        else:
            text, warnings, nul_count = _extract_utf8(content)
    except Exception as exc:
        return _result(
            filename=filename,
            kind=kind,
            byte_count=byte_count,
            status="failed",
            pages=pages,
            warnings=[f"Text extraction failed ({type(exc).__name__})."],
        )

    if nul_count:
        warnings.append(
            f"Removed {nul_count} NUL character(s) while normalizing the source."
        )

    if not text.strip(" \t\n\f"):
        if kind == "pdf" and not any("OCR" in warning for warning in warnings):
            warnings.append(
                "No selectable PDF text was found; the document may require OCR."
            )
        return _result(
            filename=filename,
            kind=kind,
            byte_count=byte_count,
            status="failed",
            pages=pages,
            warnings=warnings,
        )

    original_chars = len(text)
    truncated = original_chars > MAX_EXTRACTED_CHARS
    if truncated:
        text = text[:MAX_EXTRACTED_CHARS].rstrip()
        warnings.append(
            f"Extracted text was truncated from {original_chars:,} to "
            f"{MAX_EXTRACTED_CHARS:,} characters."
        )

    return _result(
        filename=filename,
        kind=kind,
        byte_count=byte_count,
        status="ok",
        text=text,
        pages=pages,
        truncated=truncated,
        warnings=warnings,
        extracted_chars=original_chars,
    )


def extract_text(filename: str, content: bytes) -> tuple[str, str]:
    """Return ``(text, status)`` for backwards compatibility."""

    result = extract_text_result(filename, content)
    return result.text, result.status


# Descriptive alias for callers that model uploads as knowledge-base sources.
extract_source = extract_text_result
