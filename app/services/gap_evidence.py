"""Structured, session-bound evidence upgrades for Maximum Verified Match."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from app.schemas.resume import (
    GapEvidenceSubmission,
    GapEvidenceValidation,
    MaximumMatchGap,
)

from .resume_scoring import (
    canonical_markdown_heading,
    extract_ats_keywords,
    extract_markdown_section,
)


@dataclass(frozen=True)
class ValidatedGapEvidence:
    gap: MaximumMatchGap
    submission: GapEvidenceSubmission


def _gap_id(skill: str) -> str:
    key = canonical_markdown_heading(skill)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _clean_cell(value: str) -> str:
    value = value.strip().strip("|").strip()
    value = re.sub(r"^[-*]\s+", "", value)
    value = value.replace("**", "").replace("`", "")
    return " ".join(value.strip(" \"'“”‘’").split())


def _label_and_reason(value: str) -> tuple[str, str]:
    cleaned = _clean_cell(value)
    bold = re.match(r"^\*\*(.+?)\*\*\s*(?::|[-—–])?\s*(.*)$", value.strip())
    if bold:
        return _clean_cell(bold.group(1)), _clean_cell(bold.group(2))
    parts = re.split(r"\s+[—–]\s+|:\s+", cleaned, maxsplit=1)
    return parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""


def _section_rows(section: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for raw_line in (section or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("|") and line.endswith("|"):
            cells = [_clean_cell(cell) for cell in line.strip("|").split("|")]
            if not cells:
                continue
            heading = canonical_markdown_heading(cells[0])
            if (
                heading in {
                    "requirement",
                    "keyword",
                    "requirement keyword",
                    "gap",
                    "skill",
                }
                or re.fullmatch(r":?-{3,}:?", cells[0])
            ):
                continue
            rows.append((cells[0], cells[1] if len(cells) > 1 else ""))
            continue
        if re.match(r"^[-*]\s+\S", line):
            rows.append(_label_and_reason(line))
    return rows


def extract_maximum_match_gaps(
    match_strategy: str,
    jd_analysis: str,
) -> list[MaximumMatchGap]:
    """Return stable, deduplicated gap cards from strategist artifacts."""
    ats_keywords = extract_ats_keywords(jd_analysis)
    ats_keys = {
        canonical_markdown_heading(keyword)
        for keyword in ats_keywords
    }
    collected: dict[str, MaximumMatchGap] = {}
    sections = (
        (
            "Do-Not-Claim List",
            "do_not_claim",
            "The strategist found no supporting evidence in the uploaded material.",
        ),
        (
            "Genuine Gaps (do not paper over)",
            "genuine_gap",
            "The requirement was identified as a genuine evidence gap.",
        ),
    )
    for heading, origin, fallback_reason in sections:
        section = extract_markdown_section(match_strategy, heading)
        for skill, reason in _section_rows(section):
            key = canonical_markdown_heading(skill)
            if not key or key in {"none", "no genuine gaps", "no gaps"}:
                continue
            is_ats_keyword = (
                key in ats_keys
                or any(
                    (len(key) >= 4 and len(ats_key) >= 4)
                    and (key in ats_key or ats_key in key)
                    for ats_key in ats_keys
                    if key and ats_key
                )
            )
            item = MaximumMatchGap(
                id=_gap_id(skill),
                skill=skill,
                reason=reason or fallback_reason,
                origin=origin,
                ats_keyword=is_ats_keyword,
            )
            # The Do-Not-Claim list may also contain general writing advice.
            # Show only JD-keyword exclusions there; genuine-gap entries remain
            # visible even when their label is not a verbatim ATS keyword.
            if origin == "do_not_claim" and not is_ats_keyword:
                continue
            # The Do-Not-Claim explanation is normally the most specific.
            if key not in collected or origin == "do_not_claim":
                collected[key] = item
    return list(collected.values())


def validate_gap_evidence(
    gaps: list[MaximumMatchGap],
    submissions: list[GapEvidenceSubmission],
) -> list[ValidatedGapEvidence]:
    known = {gap.id: gap for gap in gaps}
    seen: set[str] = set()
    validated: list[ValidatedGapEvidence] = []
    for submission in submissions:
        gap = known.get(submission.gap_id)
        if gap is None:
            raise ValueError(
                "One evidence item refers to an unknown or expired gap. "
                "Reload the gap list and try again."
            )
        if submission.gap_id in seen:
            raise ValueError(
                f"Submit only one evidence record for {gap.skill}; combine "
                "multiple examples into its evidence description."
            )
        evidence_payload = " ".join(
            (
                submission.evidence_text,
                submission.outcome,
                submission.role_or_contribution,
            )
        )
        if re.search(
            r"\b(?:ignore\s+(?:all\s+|any\s+|the\s+|previous\s+)?"
            r"instructions?|system\s+message|developer\s+message|"
            r"reveal\s+(?:the\s+)?prompt|do\s+not\s+follow)\b",
            evidence_payload,
            flags=re.IGNORECASE,
        ):
            raise ValueError(
                f"Evidence for {gap.skill} contains instruction-like text. "
                "Describe only the work, product, technology, and outcome."
            )
        seen.add(submission.gap_id)
        validated.append(
            ValidatedGapEvidence(gap=gap, submission=submission)
        )
    return validated


def evidence_validation_rows(
    evidence: list[ValidatedGapEvidence],
) -> list[GapEvidenceValidation]:
    return [
        GapEvidenceValidation(
            gap_id=item.gap.id,
            skill=item.gap.skill,
            source_type=item.submission.source_type,
            source_label=item.submission.source_name,
        )
        for item in evidence
    ]


def evidence_signature(evidence: list[ValidatedGapEvidence]) -> str:
    payload = sorted(
        [
            {
                "skill": item.gap.skill,
                **item.submission.model_dump(mode="json"),
            }
            for item in evidence
        ],
        key=lambda item: item["gap_id"],
    )
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def augment_profile_with_gap_evidence(
    candidate_profile: str,
    evidence: list[ValidatedGapEvidence],
) -> str:
    if not evidence:
        return candidate_profile
    blocks = [
        "\n## User-Attested Gap Evidence",
        (
            "The candidate supplied and explicitly attested to the following "
            "additional facts after reviewing the detected gaps. Treat these "
            "as authorized facts, preserve their exact scope, and never infer "
            "greater ownership, scale, or outcomes."
        ),
    ]
    for item in evidence:
        submission = item.submission
        source_label = (
            "Work experience"
            if submission.source_type.value == "work_experience"
            else "Product or project"
        )
        blocks.extend(
            [
                f"\n### {item.gap.skill}",
                f"- **Evidence type:** {source_label}",
                f"- **Employer or product:** {submission.source_name}",
                (
                    "- **Role or personal contribution:** "
                    f"{submission.role_or_contribution}"
                ),
                f"- **Dates:** {submission.dates}",
                f"- **Candidate-attested evidence:** {submission.evidence_text}",
            ]
        )
        if submission.outcome:
            blocks.append(f"- **Outcome:** {submission.outcome}")
        if submission.reference_url:
            blocks.append(f"- **Reference:** {submission.reference_url}")
    return candidate_profile.rstrip() + "\n\n" + "\n".join(blocks).strip()


def augment_strategy_with_gap_evidence(
    match_strategy: str,
    evidence: list[ValidatedGapEvidence],
) -> str:
    if not evidence:
        return match_strategy
    lines = [
        "\n## User-Attested Gap Resolutions",
        (
            "These entries supersede the matching item in Genuine Gaps and "
            "the Do-Not-Claim List only for the exact scope stated below. "
            "They do not authorize adjacent technologies or broader claims."
        ),
    ]
    for item in evidence:
        submission = item.submission
        placement = (
            "the named Experience role"
            if submission.source_type.value == "work_experience"
            else "Projects and Skills"
        )
        lines.append(
            f"- **{item.gap.skill}** — Resolved by candidate-attested "
            f"{submission.source_type.value.replace('_', ' ')} evidence from "
            f"{submission.source_name}; place it in {placement} using only "
            "the submitted facts."
        )
    return match_strategy.rstrip() + "\n\n" + "\n".join(lines).strip()


def resolved_gap_names(evidence: list[ValidatedGapEvidence]) -> list[str]:
    return [item.gap.skill for item in evidence]
