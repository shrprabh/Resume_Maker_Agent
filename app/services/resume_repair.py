"""Conservative, deterministic repairs applied before resume publication.

The model pipeline is still responsible for writing the resume.  This module
only fixes two mechanical failure modes that should not discard an otherwise
usable draft:

* normalize or restore a verified Education section; and
* shorten an overlong draft by removing complete, low-priority bullets.

No prose is synthesized here.  Education is copied from the candidate profile
verbatim, and compaction never rewrites or partially truncates a bullet.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re

from app.services.resume_scoring import (
    canonical_markdown_heading,
    extract_ats_keywords,
    extract_claimable_keywords,
    extract_unsupported_keywords,
    extract_user_attested_keywords,
)


_WORD_PATTERN = re.compile(r"\b[\w+#./-]+\b")
_HEADING_PATTERN = re.compile(
    r"^(?P<level>#{2,3})\s+(?P<title>.+?)\s*$",
    re.MULTILINE,
)
_BULLET_PATTERN = re.compile(r"^\s*[-*]\s+\S")
_SAFE_EDUCATION_ALIASES = {
    "education certification",
    "education certifications",
    "education and certification",
    "education and certifications",
    "education credentials",
}
_EDUCATION_SIGNALS = re.compile(
    r"\b(?:"
    r"university|college|school|institute|academy|"
    r"bachelor(?:'s)?|master(?:'s)?|doctorate|doctoral|ph\.?d\.?|"
    r"associate(?:'s)?|degree|diploma|b\.?\s*[as]\.?|m\.?\s*[as]\.?"
    r")\b",
    re.IGNORECASE,
)
_PLACEHOLDER_EDUCATION = re.compile(
    r"(?:"
    r"^\s*(?:[-*]\s*)?(?:n/?a|none|unknown|not\s+(?:provided|found|"
    r"available|listed|supplied)|missing)\s*[.!]?\s*$"
    r"|"
    r"\b(?:no|without)\s+(?:verified\s+)?(?:education|degree|academic)"
    r"(?:\s+(?:details?|information|history|evidence|records?))?"
    r"(?:\s+(?:was|were|is|are))?\s*(?:provided|found|available|listed|"
    r"supplied|included)?\b"
    r"|"
    r"\b(?:education|degree|academic\s+(?:history|background))\b"
    r"[^\n.]{0,40}\b(?:not\s+(?:provided|found|available|listed|supplied)|"
    r"unknown|missing|n/?a|none)\b"
    r")",
    re.IGNORECASE | re.MULTILINE,
)
_GENERIC_BULLET_MARKERS = (
    "collaborated",
    "participated",
    "responsible for",
    "helped",
    "assisted",
    "supported",
    "worked with",
    "contributed to",
)


@dataclass(frozen=True)
class PublicationRepairResult:
    """The repaired document plus an auditable summary of safe changes."""

    markdown: str
    repair_notes: tuple[str, ...]
    original_word_count: int
    final_word_count: int


@dataclass(frozen=True)
class _BulletSpan:
    start: int
    end: int
    text: str
    section: str
    role: str
    document_index: int


def _word_count(text: str) -> int:
    return len(_WORD_PATTERN.findall(text or ""))


def _section_matches(text: str) -> list[re.Match[str]]:
    return list(
        re.finditer(
            r"^##\s+(.+?)\s*$\n?(.*?)(?=^##\s+|\Z)",
            text or "",
            flags=re.MULTILINE | re.DOTALL,
        )
    )


def _section_body_verbatim(text: str, accepted_headings: set[str]) -> str:
    for match in _section_matches(text):
        if canonical_markdown_heading(match.group(1)) in accepted_headings:
            # Removing boundary newlines keeps the source's internal text,
            # punctuation, bullets, and line wrapping byte-for-byte intact.
            return match.group(2).strip("\r\n")
    return ""


def _has_education_section(text: str) -> bool:
    return any(
        canonical_markdown_heading(match.group("title")) == "education"
        for match in _HEADING_PATTERN.finditer(text or "")
        if match.group("level") == "##"
    )


def _normalize_education_heading(text: str) -> tuple[str, bool]:
    headings = [
        match
        for match in _HEADING_PATTERN.finditer(text or "")
        if match.group("level") == "##"
    ]
    exact = next(
        (
            match
            for match in headings
            if canonical_markdown_heading(match.group("title")) == "education"
        ),
        None,
    )
    selected = exact or next(
        (
            match
            for match in headings
            if canonical_markdown_heading(match.group("title"))
            in _SAFE_EDUCATION_ALIASES
        ),
        None,
    )
    if selected is None or selected.group(0) == "## Education":
        return text, False
    return (
        text[: selected.start()] + "## Education" + text[selected.end() :],
        True,
    )


def _verified_education_body(candidate_profile: str) -> str:
    exact = _section_body_verbatim(candidate_profile, {"education"})
    combined = False
    body = exact
    if not body:
        body = _section_body_verbatim(
            candidate_profile,
            _SAFE_EDUCATION_ALIASES,
        )
        combined = bool(body)
    if not body or not re.search(r"[A-Za-z0-9]", body):
        return ""
    if _PLACEHOLDER_EDUCATION.search(body):
        return ""
    # A combined Education & Certifications section containing only a
    # certification must not be used to manufacture an Education section.
    if combined and not _EDUCATION_SIGNALS.search(body):
        return ""
    return body


def _restore_education(
    resume_markdown: str,
    candidate_profile: str,
) -> tuple[str, bool]:
    if _has_education_section(resume_markdown):
        return resume_markdown, False
    body = _verified_education_body(candidate_profile)
    if not body:
        return resume_markdown, False
    return (
        resume_markdown.rstrip()
        + "\n\n## Education\n"
        + body
        + "\n",
        True,
    )


def _bullet_spans(text: str) -> list[_BulletSpan]:
    lines = (text or "").splitlines(keepends=True)
    offsets: list[int] = []
    cursor = 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line)

    section = ""
    role = ""
    role_ordinal = 0
    metadata: dict[int, tuple[str, str]] = {}
    starts: list[int] = []
    for index, line in enumerate(lines):
        heading = re.match(r"^(#{2,3})\s+(.+?)\s*$", line.rstrip("\r\n"))
        if heading:
            canonical = canonical_markdown_heading(heading.group(2))
            if heading.group(1) == "##":
                section = canonical
                role = ""
            else:
                role_ordinal += 1
                role = f"{canonical}@{role_ordinal}"
        if _BULLET_PATTERN.match(line):
            starts.append(index)
            metadata[index] = (section, role or f"{section}:unheaded")

    spans: list[_BulletSpan] = []
    for document_index, line_index in enumerate(starts):
        end_line = len(lines)
        for candidate in range(line_index + 1, len(lines)):
            line = lines[candidate]
            if _BULLET_PATTERN.match(line) or re.match(r"^#{2,3}\s+\S", line):
                end_line = candidate
                break
        start = offsets[line_index]
        end = offsets[end_line] if end_line < len(lines) else len(text)
        section_name, role_name = metadata[line_index]
        spans.append(
            _BulletSpan(
                start=start,
                end=end,
                text=text[start:end],
                section=section_name,
                role=role_name,
                document_index=document_index,
            )
        )
    return spans


def _is_narrative_bullet(span: _BulletSpan) -> bool:
    return span.section in {
        "experience",
        "professional experience",
        "work experience",
        "projects",
        "selected projects",
        "project experience",
    }


def _normalized_bullet(text: str) -> str:
    value = re.sub(r"^\s*[-*]\s+", "", text.strip())
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[*_`]", "", value)
    return value.casefold().strip(" .;")


def _keyword_present(keyword: str, document: str) -> bool:
    needle = re.sub(r"\s+", " ", keyword).casefold().strip()
    haystack = re.sub(r"\s+", " ", document).casefold()
    if not needle:
        return False
    return (
        re.search(
            rf"(?<![a-z0-9+#]){re.escape(needle)}(?![a-z0-9+#])",
            haystack,
        )
        is not None
    )


def _unique_casefolded(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = re.sub(r"\s+", " ", value).casefold().strip()
        if value.strip() and key not in seen:
            result.append(value.strip())
            seen.add(key)
    return result


def _placed_claimable_keywords(
    resume_markdown: str,
    jd_analysis: str,
    match_strategy: str,
) -> tuple[str, ...]:
    ats_keywords = extract_ats_keywords(jd_analysis)
    claimable = _unique_casefolded(
        extract_claimable_keywords(match_strategy, ats_keywords)
        + extract_user_attested_keywords(match_strategy)
    )
    if not claimable and ats_keywords:
        unsupported = {
            canonical_markdown_heading(item)
            for item in extract_unsupported_keywords(match_strategy)
        }
        claimable = [
            keyword
            for keyword in ats_keywords
            if canonical_markdown_heading(keyword) not in unsupported
        ]
    return tuple(
        keyword
        for keyword in claimable
        if _keyword_present(keyword, resume_markdown)
    )


def _delete_span(text: str, span: _BulletSpan) -> str:
    return text[: span.start] + text[span.end :]


def _deletion_is_safe(
    current: str,
    span: _BulletSpan,
    protected_keywords: tuple[str, ...],
    *,
    minimum_per_role: int,
) -> bool:
    spans = _bullet_spans(current)
    if len(spans) <= 4:
        return False
    narrative = [item for item in spans if _is_narrative_bullet(item)]
    role_counts = Counter(item.role for item in narrative)
    if role_counts[span.role] <= minimum_per_role:
        return False
    candidate = _delete_span(current, span)
    return all(
        _keyword_present(keyword, candidate)
        for keyword in protected_keywords
    )


def _remove_duplicate_bullets(
    text: str,
    protected_keywords: tuple[str, ...],
) -> tuple[str, int]:
    removed = 0
    while True:
        spans = [span for span in _bullet_spans(text) if _is_narrative_bullet(span)]
        seen: set[str] = set()
        duplicate: _BulletSpan | None = None
        for span in spans:
            normalized = _normalized_bullet(span.text)
            if normalized and normalized in seen:
                if _deletion_is_safe(
                    text,
                    span,
                    protected_keywords,
                    minimum_per_role=1,
                ):
                    duplicate = span
                    break
            seen.add(normalized)
        if duplicate is None:
            break
        text = _delete_span(text, duplicate)
        removed += 1
    return text, removed


def _removal_priority(span: _BulletSpan) -> tuple[int, int, int, int, int]:
    lowered = span.text.casefold()
    generic = sum(marker in lowered for marker in _GENERIC_BULLET_MARKERS)
    has_metric = bool(re.search(r"(?:\$|\b\d+(?:[.,]\d+)?%?\b)", span.text))
    is_project = "project" in span.section
    return (
        generic,
        int(not has_metric),
        int(is_project),
        span.document_index,
        _word_count(span.text),
    )


def _compact_bullets(
    text: str,
    protected_keywords: tuple[str, ...],
    target_words: int,
) -> tuple[str, int, int]:
    text, duplicate_count = _remove_duplicate_bullets(
        text,
        protected_keywords,
    )
    removed = duplicate_count
    if _word_count(text) <= target_words:
        return text, duplicate_count, removed

    # First retain two bullets per role where possible.  If that cannot meet
    # the limit, a second pass may reduce older/less relevant roles to one.
    for minimum_per_role in (2, 1):
        while _word_count(text) > target_words:
            spans = [
                span
                for span in _bullet_spans(text)
                if _is_narrative_bullet(span)
            ]
            candidates = sorted(
                spans,
                key=_removal_priority,
                reverse=True,
            )
            selected = next(
                (
                    span
                    for span in candidates
                    if _deletion_is_safe(
                        text,
                        span,
                        protected_keywords,
                        minimum_per_role=minimum_per_role,
                    )
                ),
                None,
            )
            if selected is None:
                break
            text = _delete_span(text, selected)
            removed += 1
    return text, duplicate_count, removed


def repair_resume_for_publication(
    resume_markdown: str,
    *,
    candidate_profile: str = "",
    jd_analysis: str = "",
    match_strategy: str = "",
    target_words: int = 900,
) -> PublicationRepairResult:
    """Apply safe publication repairs without inventing candidate claims.

    The function is deliberately idempotent: once a heading is canonical and
    the draft is within the target, running it again leaves the Markdown
    unchanged.
    """
    original = resume_markdown or ""
    original_words = _word_count(original)
    repaired, heading_changed = _normalize_education_heading(original)
    repaired, education_restored = _restore_education(
        repaired,
        candidate_profile,
    )
    notes: list[str] = []
    if heading_changed:
        notes.append("Normalized the verified Education section heading.")
    if education_restored:
        notes.append(
            "Restored the Education section verbatim from the candidate profile."
        )

    protected_keywords = _placed_claimable_keywords(
        repaired,
        jd_analysis,
        match_strategy,
    )
    duplicate_count = 0
    removed_count = 0
    if _word_count(repaired) > target_words:
        repaired, duplicate_count, removed_count = _compact_bullets(
            repaired,
            protected_keywords,
            target_words,
        )
    if removed_count:
        notes.append(
            f"Removed {removed_count} complete low-priority bullet"
            f"{'s' if removed_count != 1 else ''}"
            + (
                f", including {duplicate_count} duplicate"
                f"{'s' if duplicate_count != 1 else ''}"
                if duplicate_count
                else ""
            )
            + "."
        )
    final_words = _word_count(repaired)
    if final_words > target_words and original_words > target_words:
        notes.append(
            f"Draft remains {final_words} words because further whole-bullet "
            "removal would drop protected ATS evidence or minimum role detail."
        )

    return PublicationRepairResult(
        markdown=repaired,
        repair_notes=tuple(notes),
        original_word_count=original_words,
        final_word_count=final_words,
    )
