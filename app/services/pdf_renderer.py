"""Render a Markdown document (resume or cover letter) to a PDF on disk.

Markdown -> HTML via the `markdown` package, HTML -> PDF via xhtml2pdf
(pure Python — no system libraries needed). xhtml2pdf is synchronous and
CPU-bound, so callers should wrap render_pdf in asyncio.to_thread.
"""

import re
from pathlib import Path

import markdown as md_lib
from xhtml2pdf import pisa

PDF_DIR = Path(__file__).resolve().parents[2] / "generated_pdfs"

_CSS = """
@page { size: letter; margin: 0.56in 0.64in 0.58in; }
body {
  font-family: Helvetica, Arial, sans-serif;
  font-size: 10.15pt;
  line-height: 1.34;
  color: #202522;
  background-color: #ffffff;
}
body.compact {
  font-size: 9.5pt;
  line-height: 1.25;
}
body.dense {
  font-size: 8.85pt;
  line-height: 1.18;
}
h1 {
  font-size: 20pt;
  line-height: 1.1;
  margin: 0 0 3pt 0;
  color: #183f2d;
  page-break-after: avoid;
}
h2 {
  font-size: 11.25pt;
  line-height: 1.1;
  color: #183f2d;
  border-bottom: 0.8pt solid #86a393;
  margin: 11pt 0 5pt 0;
  padding-bottom: 2.4pt;
  page-break-after: avoid;
}
h3 {
  font-size: 10.35pt;
  line-height: 1.22;
  color: #1c2922;
  margin: 7pt 0 2.5pt 0;
  page-break-after: avoid;
  -pdf-keep-with-next: true;
}
p {
  margin: 1.5pt 0 3.5pt 0;
  orphans: 2;
  widows: 2;
}
.contact {
  margin: 0 0 6pt 0;
  color: #48534d;
  font-size: 9.25pt;
  line-height: 1.25;
}
.skill-line {
  margin: 0 0 2.4pt 0;
  line-height: 1.28;
}
.resume-bullet {
  margin: 0 0 3.1pt 12pt;
  text-indent: -9pt;
  line-height: 1.31;
  orphans: 2;
  widows: 2;
}
.document-entry {
  page-break-inside: auto;
}
.keep-with-next {
  -pdf-keep-with-next: true;
}
body.compact h1 { font-size: 18.5pt; }
body.compact h2 { margin-top: 8pt; margin-bottom: 3.5pt; }
body.compact h3 { margin-top: 5pt; }
body.compact p { margin-bottom: 2.5pt; }
body.compact .resume-bullet { margin-bottom: 2pt; line-height: 1.25; }
body.dense h1 { font-size: 17.5pt; }
body.dense h2 { margin-top: 6pt; margin-bottom: 3pt; }
body.dense h3 { margin-top: 4pt; }
body.dense p { margin-bottom: 1.8pt; }
body.dense .contact { margin-bottom: 4pt; font-size: 8.4pt; }
body.dense .skill-line { margin-bottom: 1.5pt; line-height: 1.18; }
body.dense .resume-bullet { margin-bottom: 1.5pt; line-height: 1.18; }
"""


def normalize_resume_markdown(markdown_text: str) -> str:
    """Repair common LLM Markdown mistakes before rendering.

    Gemini occasionally emits role bullets as inline separators:
    ``**Role** | Dates * Did X * Did Y``. Markdown treats that as a paragraph,
    so the PDF has literal asterisks and no hanging indent. Normalize those
    separators into real list items and promote experience/project role lines
    to level-three headings. This also protects output created before the
    stricter writer prompt was added.
    """
    text = markdown_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    # Split inline bullet separators without touching bold ``**text**``.
    text = re.sub(r"[ \t]+\*[ \t]+(?=\S)", "\n- ", text)

    lines = text.splitlines()
    normalized: list[str] = []
    section = ""
    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith("## "):
            section = stripped[3:].strip().lower()

        # Accept either Markdown bullet marker, but output one canonical form.
        if re.match(r"^\s*\*\s+\S", line):
            line = re.sub(r"^\s*\*\s+", "- ", line)
            stripped = line.strip()

        # Older drafts used bold paragraphs for role headings. Semantic h3
        # headings keep the role with the bullets and create consistent space.
        if (
            section in {"experience", "projects"}
            and stripped.startswith("**")
            and "**" in stripped[2:]
            and not stripped.startswith("### ")
        ):
            line = "### " + stripped.replace("**", "")
            stripped = line

        # Models occasionally put every bold Skills category on one source
        # line. Split at category markers before Markdown can collapse the
        # entire section into one unreadable paragraph.
        skill_categories = (
            re.split(r"(?=\*\*[^*\n]+:\*\*)", stripped)
            if section == "skills"
            else []
        )
        skill_categories = [part.strip() for part in skill_categories if part.strip()]
        if len(skill_categories) > 1:
            for category in skill_categories:
                if normalized and normalized[-1].strip():
                    normalized.append("")
                normalized.append(category)
            continue

        # Python-Markdown requires a blank boundary before a list in several
        # contexts. Add one only when starting a new list.
        if stripped.startswith("- ") and normalized:
            previous = normalized[-1].strip()
            if previous and not previous.startswith("- "):
                normalized.append("")

        # A single newline inside Markdown is only visual source formatting;
        # it collapses to a space in HTML. Skills, education, and
        # certifications are intentionally one item per line, so make each
        # source line a real paragraph.
        if (
            section in {"skills", "education", "certifications"}
            and stripped
            and not stripped.startswith(("#", "- "))
            and normalized
            and normalized[-1].strip()
        ):
            normalized.append("")

        normalized.append(line)

    return "\n".join(normalized).strip() + "\n"


def _decorate_resume_html(body: str) -> str:
    """Apply semantic classes and ATS-readable hanging list markers."""

    # The first paragraph after the candidate name is the contact line.
    body = re.sub(
        r"(<h1>.*?</h1>)\s*<p>",
        r'\1<p class="contact">',
        body,
        count=1,
        flags=re.DOTALL,
    )

    # Convert list markup to literal hyphen paragraphs. xhtml2pdf's default
    # bullet glyph extracts as a control character in some PDF parsers; a
    # literal hyphen remains readable to both ATS software and copy/paste.
    def convert_list(match: re.Match) -> str:
        items = re.findall(r"<li>(.*?)</li>", match.group(1), flags=re.DOTALL)
        return "\n".join(
            f'<p class="resume-bullet">-&nbsp;{item.strip()}</p>'
            for item in items
        )

    body = re.sub(
        r"<ul>\s*(.*?)\s*</ul>",
        convert_list,
        body,
        flags=re.DOTALL,
    )

    # Skills stay single-column but each category receives its own visual row.
    def decorate_skills(match: re.Match) -> str:
        section = re.sub(
            r"<p>",
            '<p class="skill-line">',
            match.group(2),
        )
        return match.group(1) + section

    body = re.sub(
        r"(<h2>Skills</h2>)(.*?)(?=<h2>|\Z)",
        decorate_skills,
        body,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Group role/project markup semantically. The h3 rule keeps the heading
    # with its first bullet, while the role itself remains page-breakable so a
    # long entry does not create a large blank area or an unnecessary page.
    return re.sub(
        r"(<h3>.*?</h3>.*?)(?=<h3>|<h2>|\Z)",
        r'<div class="document-entry">\1</div>',
        body,
        flags=re.DOTALL,
    )


def markdown_to_html(markdown_text: str) -> str:
    normalized = normalize_resume_markdown(markdown_text)
    body = md_lib.markdown(normalized, extensions=["sane_lists"])
    body = _decorate_resume_html(body)
    word_count = len(re.findall(r"\b[\w+#./-]+\b", normalized))
    if len(normalized) > 7_200 or word_count > 850:
        density = "dense"
    elif len(normalized) > 6_500 or word_count > 750:
        density = "compact"
    else:
        density = "standard"
    return (
        f"<html><head><style>{_CSS}</style></head>"
        f'<body class="{density}">{body}</body></html>'
    )


def render_pdf(markdown_text: str, out_name: str) -> Path:
    PDF_DIR.mkdir(exist_ok=True)
    html = markdown_to_html(markdown_text)
    out_path = PDF_DIR / out_name
    with open(out_path, "wb") as fh:
        status = pisa.CreatePDF(html, dest=fh)
    if status.err:
        raise RuntimeError(f"PDF rendering failed for {out_name}")
    return out_path
