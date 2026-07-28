"""Deterministic resume coverage scoring and reviewer-response validation.

Language models remain useful judges of writing and claim fidelity, but exact
keyword presence is a parsing problem. Keeping those responsibilities
separate prevents malformed reviewer output from becoming a misleading 0/100.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator


class ReviewerDecision(BaseModel):
    score: int = Field(ge=0, le=100)
    ats_coverage: int = Field(ge=0, le=100)
    fabrication_count: int = Field(ge=0)
    approved: bool = False
    feedback: list[str] = Field(default_factory=list)

    @field_validator("feedback", mode="before")
    @classmethod
    def normalize_feedback(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        return [str(item) for item in value]


class ResumeScorecard(BaseModel):
    supported_ats_coverage: int | None = Field(default=None, ge=0, le=100)
    overall_requirement_match: int | None = Field(default=None, ge=0, le=100)
    evidence_integrity: int | None = Field(default=None, ge=0, le=100)
    quality_score: int | None = Field(default=None, ge=0, le=100)
    score_status: Literal["valid", "partial", "unavailable"]
    structure_valid: bool
    structure_issues: list[str] = Field(default_factory=list)
    word_count: int = Field(ge=0)
    claimable_keywords: list[str] = Field(default_factory=list)
    placed_keywords: list[str] = Field(default_factory=list)
    missing_supported_keywords: list[str] = Field(default_factory=list)
    unsupported_keywords: list[str] = Field(default_factory=list)


class ResumeStructureAudit(BaseModel):
    valid: bool
    issues: list[str] = Field(default_factory=list)
    word_count: int = Field(ge=0)
    bullet_count: int = Field(ge=0)
    role_count: int = Field(ge=0)


def canonical_markdown_heading(value: str) -> str:
    """Normalize harmless heading variations without weakening section checks."""
    value = re.sub(r"\([^)]*\)", "", value or "")
    value = re.sub(r"[^a-z0-9]+", " ", value.casefold())
    return " ".join(value.split())


_MONTH_NUMBERS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def _role_end_date(role_heading: str) -> tuple[int, int] | None:
    if re.search(r"\b(?:present|current)\b", role_heading, re.IGNORECASE):
        return (9999, 12)
    month_pattern = "|".join(_MONTH_NUMBERS)
    month_years = re.findall(
        rf"\b({month_pattern})\s+((?:19|20)\d{{2}})\b",
        role_heading,
        flags=re.IGNORECASE,
    )
    if month_years:
        month, year = month_years[-1]
        return (int(year), _MONTH_NUMBERS[month.casefold()])
    years = re.findall(r"\b(?:19|20)\d{2}\b", role_heading)
    if years:
        return (int(years[-1]), 12)
    return None


def normalize_experience_chronology(markdown_text: str) -> str:
    """Put parseable Experience entries in conventional newest-first order.

    Reordering role blocks is a lossless formatting correction: headings and
    bullets remain byte-for-byte unchanged. If any role lacks a parseable end
    date, preserve the model's order rather than guessing.
    """
    pattern = r"(^##\s+Experience\s*$\n)(.*?)(?=^##\s+|\Z)"
    match = re.search(pattern, markdown_text or "", re.MULTILINE | re.DOTALL | re.I)
    if not match:
        return markdown_text
    body = match.group(2)
    parts = re.split(r"(?=^###\s+)", body, flags=re.MULTILINE)
    prefix = parts[0]
    entries = [part for part in parts[1:] if part.strip()]
    if len(entries) < 2:
        return markdown_text

    dated_entries: list[tuple[tuple[int, int], int, str]] = []
    for index, entry in enumerate(entries):
        heading = entry.splitlines()[0]
        end_date = _role_end_date(heading)
        if end_date is None:
            return markdown_text
        dated_entries.append((end_date, index, entry))
    ordered = sorted(
        dated_entries,
        key=lambda item: (item[0][0], item[0][1], -item[1]),
        reverse=True,
    )
    new_body = prefix + "".join(entry for _, _, entry in ordered)
    return (
        markdown_text[: match.start(2)]
        + new_body
        + markdown_text[match.end(2) :]
    )


def normalize_skill_category_markdown(markdown_text: str) -> str:
    """Standardize model-emitted Skills labels without changing their claims.

    Models commonly return valid category lines such as ``Languages: C#`` or
    ``- Frameworks: React`` even when asked for ``**Languages:** C#``.  The
    content is already usable and evidence-grounded; this lossless pass only
    removes an optional list marker and bolds the existing label so the
    renderer and deterministic structure audit interpret it consistently.
    """
    pattern = r"(^##\s+Skills\s*$\n)(.*?)(?=^##\s+|\Z)"
    match = re.search(pattern, markdown_text or "", re.MULTILINE | re.DOTALL | re.I)
    if not match:
        return markdown_text

    normalized_lines: list[str] = []
    for line in match.group(2).splitlines(keepends=True):
        newline = "\n" if line.endswith("\n") else ""
        content = line[:-1] if newline else line
        candidate = re.sub(r"^\s*[-+*]\s+", "", content).strip()
        category = re.fullmatch(
            r"(?:\*\*)?([A-Za-z][A-Za-z0-9 &/+.,()-]{1,49})"
            r"(?:\*\*)?\s*:\s*(?:\*\*)?\s*(\S.*)",
            candidate,
        )
        if category and not candidate.startswith("#"):
            label = re.sub(r"\s+", " ", category.group(1)).strip()
            values = category.group(2).strip()
            normalized_lines.append(f"**{label}:** {values}{newline}")
        else:
            normalized_lines.append(line)

    normalized_body = "".join(normalized_lines)
    return (
        markdown_text[: match.start(2)]
        + normalized_body
        + markdown_text[match.end(2) :]
    )


def extract_markdown_section(text: str, heading: str) -> str:
    """Return a level-two section body, or an empty string when absent."""
    target = canonical_markdown_heading(heading)
    pattern = r"^##\s+(.+?)\s*$\n?(.*?)(?=^##\s+|\Z)"
    for match in re.finditer(
        pattern,
        text or "",
        flags=re.MULTILINE | re.DOTALL,
    ):
        if canonical_markdown_heading(match.group(1)) == target:
            return match.group(2).strip()
    return ""


def _clean_markdown_label(value: str) -> str:
    value = value.strip().strip("|").strip()
    value = re.sub(r"^[*-]\s+", "", value)
    value = re.sub(r"^\d+[.)]\s+", "", value)
    value = value.replace("**", "").replace("`", "")
    value = value.strip().strip("\"'“”‘’").strip()
    return re.sub(r"\s+", " ", value)


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = _clean_markdown_label(value)
        key = cleaned.casefold()
        if cleaned and key not in seen:
            result.append(cleaned)
            seen.add(key)
    return result


def _list_items(section: str) -> list[str]:
    items: list[str] = []
    for line in section.splitlines():
        stripped = line.strip()
        if re.match(r"^(?:[-*]|\d+[.)])\s+\S", stripped):
            items.append(_clean_markdown_label(stripped))
    return _unique(items)


def _expand_keyword_item(item: str) -> list[str]:
    """Split analyzer output such as '"CI/CD" AND "Continuous Integration"'."""
    quoted = re.findall(r"[\"“]([^\"”]+)[\"”]", item)
    if len(quoted) >= 2:
        return _unique(quoted)
    parts = re.split(r"\s+AND\s+|[,;]\s*", item)
    return _unique(parts if len(parts) > 1 else [item])


def extract_ats_keywords(jd_analysis: str) -> list[str]:
    """Read ATS keywords from lists, tables, or plain model-output lines."""
    section = extract_markdown_section(jd_analysis, "ATS Keywords (verbatim)")
    items = _list_items(section)

    for line in section.splitlines():
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not cells:
            continue
        first = _clean_markdown_label(cells[0])
        if (
            not first
            or canonical_markdown_heading(first)
            in {"ats keyword", "ats keywords", "keyword", "keywords"}
            or re.fullmatch(r":?-{3,}:?", first)
        ):
            continue
        items.append(first)

    # Some otherwise capable models omit list markers. The section is
    # contractually keyword-only, so each non-table line remains safe to parse.
    if not items:
        for line in section.splitlines():
            stripped = _clean_markdown_label(line)
            if not stripped or stripped.startswith("|"):
                continue
            label_match = re.match(
                r"^(?:ATS\s+)?(?:keywords?|skills?|technologies)\s*:\s*(.+)$",
                stripped,
                re.I,
            )
            items.append(label_match.group(1) if label_match else stripped)

    expanded: list[str] = []
    for item in items:
        expanded.extend(_expand_keyword_item(item))
    return _unique(expanded)


def extract_claimable_keywords(
    match_strategy: str,
    ats_keywords: list[str] | None = None,
) -> list[str]:
    """Read the first column of the strategist's keyword-placement table."""
    section = extract_markdown_section(match_strategy, "Keyword Placement Plan")
    values: list[str] = []
    positive_lines: list[str] = []
    for line in section.splitlines():
        stripped = line.strip()
        lowered = stripped.casefold()
        negative = any(
            marker in lowered
            for marker in (
                "do not place",
                "do-not-claim",
                "do not claim",
                "no evidence",
                "unsupported",
            )
        )
        if stripped and not negative:
            positive_lines.append(stripped)
        if not (stripped.startswith("|") and stripped.endswith("|")):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 2:
            continue
        first = _clean_markdown_label(cells[0])
        if not first or canonical_markdown_heading(first) in {
            "ats keyword",
            "keyword",
            "requirement",
        }:
            continue
        if re.fullmatch(r":?-{3,}:?", first):
            continue
        placement = " ".join(cells[1:]).casefold()
        if negative:
            continue
        values.extend(_expand_keyword_item(first))

    # Reconcile nonstandard bullets/arrows against the exact JD keyword list.
    # This prevents strings like "C# — Skills" from becoming fake keywords.
    if ats_keywords:
        positive_text = "\n".join(positive_lines)
        reconciled = [
            keyword
            for keyword in ats_keywords
            if _keyword_present(keyword, positive_text)
        ]
        if reconciled:
            return _unique(reconciled)

    # Fallback for callers that do not have the JD analysis available.
    if not values:
        for item in _list_items(section):
            label = re.split(
                r"\s*(?::|→|->|—|\|\s*)\s*|\s+-\s+",
                item,
                maxsplit=1,
            )[0].strip()
            label = label.replace("**", "").strip()
            values.extend(_expand_keyword_item(label))
    return _unique(values)


def extract_user_attested_keywords(match_strategy: str) -> list[str]:
    """Return gap keywords explicitly resolved by candidate-supplied evidence."""
    section = extract_markdown_section(
        match_strategy,
        "User-Attested Gap Resolutions",
    )
    values: list[str] = []
    for line in section.splitlines():
        match = re.match(r"^\s*[-*]\s+\*\*(.+?)\*\*", line)
        if match:
            values.extend(_expand_keyword_item(match.group(1)))
    return _unique(values)


def extract_unsupported_keywords(match_strategy: str) -> list[str]:
    section = extract_markdown_section(match_strategy, "Do-Not-Claim List")
    values = _list_items(section)
    for line in section.splitlines():
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not cells:
            continue
        first = _clean_markdown_label(cells[0])
        if (
            not first
            or canonical_markdown_heading(first)
            in {"requirement", "keyword", "requirement keyword"}
            or re.fullmatch(r":?-{3,}:?", first)
        ):
            continue
        values.extend(_expand_keyword_item(first))
    return _unique(values)


def _keyword_present(keyword: str, document: str) -> bool:
    """Match an exact ATS term without short-keyword substring collisions."""
    needle = re.sub(r"\s+", " ", keyword).casefold().strip()
    haystack = re.sub(r"\s+", " ", document).casefold()
    if not needle:
        return False
    # Keep language punctuation significant: C, C#, and C++ are separate
    # terms, and SQL must not be counted merely because PostgreSQL appears.
    pattern = (
        rf"(?<![a-z0-9+#]){re.escape(needle)}(?![a-z0-9+#])"
    )
    return re.search(pattern, haystack) is not None


def _keyword_is_unsupported(keyword: str, unsupported: list[str]) -> bool:
    key = keyword.casefold().strip()
    for item in unsupported:
        candidate = item.casefold().strip()
        if key == candidate:
            return True
        # Do-not-claim bullets often add a short explanation after punctuation.
        if re.match(rf"^{re.escape(key)}(?:\s*[-—:;(]|\s+experience\b)", candidate):
            return True
    return False


def _requirement_match(
    match_strategy: str,
    resolved_keywords: list[str] | None = None,
) -> int | None:
    section = extract_markdown_section(match_strategy, "Requirement-to-Evidence Map")
    if not section:
        return None

    # Prefer the explicit Must-Haves subsection. Stop at the next level-three
    # heading so nice-to-haves do not dilute the role-match score.
    must_match = re.search(
        r"^###\s+Must-Haves?\s*$\n?(.*?)(?=^###\s+|\Z)",
        section,
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    relevant = must_match.group(1).strip() if must_match else section
    items = _list_items(relevant)
    if not items:
        return None
    resolved = [
        canonical_markdown_heading(keyword)
        for keyword in (resolved_keywords or [])
    ]
    supported = 0
    for item in items:
        lacks_evidence = re.search(
            r"\b(?:NO EVIDENCE|UNSUPPORTED|NOT FOUND)\b",
            item,
            re.I,
        )
        item_key = canonical_markdown_heading(item)
        resolved_here = any(
            keyword and keyword in item_key for keyword in resolved
        )
        if not lacks_evidence or resolved_here:
            supported += 1
    return round((supported / len(items)) * 100)


def parse_reviewer_decision(text: str) -> ReviewerDecision | None:
    """Parse either the LangGraph JSON contract or the ADK verdict format."""
    cleaned = (text or "").strip()
    if not cleaned:
        return None

    json_text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    candidates = [json_text]
    embedded = re.search(r"\{.*\}", json_text, flags=re.DOTALL)
    if embedded and embedded.group(0) != json_text:
        candidates.append(embedded.group(0))
    for candidate in candidates:
        try:
            return ReviewerDecision.model_validate(json.loads(candidate))
        except (json.JSONDecodeError, ValidationError, TypeError):
            pass

    score_match = re.search(r"(?:SCORE:|score)\s*(\d{1,3})\s*/\s*100", cleaned, re.I)
    coverage_match = re.search(
        r"ATS\s+coverage:?\s*(\d{1,3})\s*%", cleaned, re.I
    )
    fabrication_match = re.search(
        r"Fabrications?:?\s*(\d+)", cleaned, re.I
    )
    if not score_match or not coverage_match:
        return None
    feedback = [
        match.group(1).strip()
        for match in re.finditer(r"^\s*\d+[.)]\s+(.+)$", cleaned, re.MULTILINE)
    ]
    approved = cleaned.casefold().startswith("approved")
    fabrications = int(fabrication_match.group(1)) if fabrication_match else 0
    try:
        return ReviewerDecision(
            score=int(score_match.group(1)),
            ats_coverage=int(coverage_match.group(1)),
            fabrication_count=fabrications,
            approved=approved and fabrications == 0,
            feedback=feedback,
        )
    except ValidationError:
        return None


def audit_resume_structure(resume_markdown: str) -> ResumeStructureAudit:
    """Reject fragments and underdeveloped drafts before model approval."""
    text = (resume_markdown or "").strip()
    headings = {
        match.group(1).strip().casefold()
        for match in re.finditer(r"^##\s+(.+?)\s*$", text, re.MULTILINE)
    }
    word_count = len(re.findall(r"\b[\w+#./-]+\b", text))
    bullet_count = len(re.findall(r"^\s*[-*]\s+\S", text, re.MULTILINE))
    role_count = len(re.findall(r"^###\s+\S", text, re.MULTILINE))
    issues: list[str] = []

    if not re.search(r"^#\s+\S", text, re.MULTILINE):
        issues.append("Missing candidate-name heading.")
    for required in ("summary", "skills", "education"):
        if required not in headings:
            issues.append(f"Missing required {required.title()} section.")
    if "experience" not in headings and "projects" not in headings:
        issues.append("Missing Experience or Projects section.")
    if word_count < 220:
        issues.append(
            f"Draft is underdeveloped at {word_count} words; target at least 220."
        )
    if word_count > 950:
        issues.append(
            f"Draft is overlong at {word_count} words; target at most 950 "
            "by removing repetition and low-relevance detail."
        )
    if ("experience" in headings or "projects" in headings) and bullet_count < 4:
        issues.append(
            f"Only {bullet_count} achievement bullets were produced; target at least 4."
        )
    if "experience" in headings and role_count < 1:
        issues.append("Experience contains no parseable role heading.")
    summary_section = extract_markdown_section(text, "Summary")
    summary_words = len(re.findall(r"\b[\w+#./-]+\b", summary_section))
    if "summary" in headings and not 40 <= summary_words <= 80:
        issues.append(
            f"Summary is {summary_words} words; target 40-80 specific words."
        )
    skill_section = extract_markdown_section(text, "Skills")
    skill_categories = re.findall(r"\*\*[^*\n]+:\*\*", skill_section)
    if "skills" in headings and len(skill_categories) < 2:
        issues.append("Skills must contain at least two labeled category lines.")
    if "skills" in headings and len(skill_categories) > 5:
        issues.append(
            f"Skills has {len(skill_categories)} category lines; consolidate "
            "related skills into no more than 5 focused categories."
        )

    return ResumeStructureAudit(
        valid=not issues,
        issues=issues,
        word_count=word_count,
        bullet_count=bullet_count,
        role_count=role_count,
    )


def build_scorecard(
    *,
    resume_markdown: str,
    jd_analysis: str,
    match_strategy: str,
    reviewer: ReviewerDecision | None,
) -> ResumeScorecard:
    structure = audit_resume_structure(resume_markdown)
    ats_keywords = extract_ats_keywords(jd_analysis)
    resolved = extract_user_attested_keywords(match_strategy)
    resolved_keys = {
        canonical_markdown_heading(keyword)
        for keyword in resolved
    }
    unsupported = [
        keyword
        for keyword in extract_unsupported_keywords(match_strategy)
        if canonical_markdown_heading(keyword) not in resolved_keys
    ]
    claimable = _unique(
        extract_claimable_keywords(match_strategy, ats_keywords) + resolved
    )
    if not claimable:
        claimable = [
            keyword
            for keyword in ats_keywords
            if not _keyword_is_unsupported(keyword, unsupported)
        ]

    placed = [
        keyword for keyword in claimable if _keyword_present(keyword, resume_markdown)
    ]
    missing = [
        keyword for keyword in claimable if not _keyword_present(keyword, resume_markdown)
    ]
    coverage = round((len(placed) / len(claimable)) * 100) if claimable else None
    requirement_match = _requirement_match(match_strategy, resolved)
    integrity = (
        max(0, 100 - reviewer.fabrication_count * 25)
        if reviewer is not None
        else None
    )
    available = [
        value
        for value in (coverage, requirement_match, integrity, reviewer.score if reviewer else None)
        if value is not None
    ]
    status: Literal["valid", "partial", "unavailable"]
    if reviewer is not None and coverage is not None:
        status = "valid"
    elif available:
        status = "partial"
    else:
        status = "unavailable"

    return ResumeScorecard(
        supported_ats_coverage=coverage,
        overall_requirement_match=requirement_match,
        evidence_integrity=integrity,
        quality_score=(
            min(reviewer.score, 50)
            if reviewer is not None and not structure.valid
            else reviewer.score if reviewer else None
        ),
        score_status=status,
        structure_valid=structure.valid,
        structure_issues=structure.issues,
        word_count=structure.word_count,
        claimable_keywords=claimable,
        placed_keywords=placed,
        missing_supported_keywords=missing,
        unsupported_keywords=unsupported,
    )


def build_maximum_match_insights(
    scorecard: ResumeScorecard,
    match_strategy: str,
    review_feedback: str,
) -> str:
    """Produce a transparent, UI-ready audit without another model call."""

    def joined(values: list[str], empty: str) -> str:
        return ", ".join(values) if values else empty

    gaps = extract_markdown_section(match_strategy, "Genuine Gaps (do not paper over)")
    if not gaps:
        gaps = "No separate genuine-gap section was returned by the strategist."
    resolved = extract_markdown_section(
        match_strategy,
        "User-Attested Gap Resolutions",
    )
    if not resolved:
        resolved = "No additional gap evidence was submitted."

    score_value = (
        f"{scorecard.supported_ats_coverage}%"
        if scorecard.supported_ats_coverage is not None
        else "Unavailable"
    )
    requirement_value = (
        f"{scorecard.overall_requirement_match}%"
        if scorecard.overall_requirement_match is not None
        else "Unavailable"
    )
    integrity_value = (
        f"{scorecard.evidence_integrity}%"
        if scorecard.evidence_integrity is not None
        else "Unavailable"
    )
    return f"""\
## Score interpretation

- **Supported ATS coverage:** {score_value}
- **Overall requirement match:** {requirement_value}
- **Evidence integrity:** {integrity_value}
- **Document structure:** {"Passed" if scorecard.structure_valid else "Needs revision"}
- **Score status:** {scorecard.score_status}

Supported ATS coverage measures only keywords the strategy mapped to candidate
evidence. It is not a promise of an employer ATS outcome.

## Verified keyword placement

{joined(scorecard.placed_keywords, "No claimable keyword placements were detected.")}

## Missing but supported keywords

{joined(scorecard.missing_supported_keywords, "None. Every supported keyword was placed.")}

## Structure audit

{joined(scorecard.structure_issues, "Passed. Required sections and minimum substance are present.")}

## Protected exclusions

{joined(scorecard.unsupported_keywords, "No explicit Do-Not-Claim terms were returned.")}

## User-attested gap resolutions

{resolved}

## Original genuine gaps

{gaps}

## Maximum-match auditor

{review_feedback or "The reviewer response was unavailable; deterministic coverage remains shown above."}
""".strip()
