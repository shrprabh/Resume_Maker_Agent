"""Deterministic input checks that run before any model tokens are spent."""

from __future__ import annotations

import re


_JOB_DESCRIPTION_SIGNALS = (
    r"\b(?:full\s+)?job description\b",
    r"\bjob requisition(?:\s+id)?\b",
    r"\bminimum qualifications?\b",
    r"\bpreferred qualifications?\b",
    r"(?im)^\s*you will\s*:",
    r"(?im)^\s*you have\s*:",
    r"\bequal employment opportunities?\b",
    r"\bnot eligible for .{0,40}(?:sponsorship|immigration)\b",
    r"\bwe(?:'re| are) looking for\b",
)


def looks_like_job_description(text: str) -> bool:
    """Identify a likely hiring post without flagging incidental phrases."""
    normalized = (text or "").strip()
    if len(normalized) < 600:
        return False
    hits = [
        index
        for index, pattern in enumerate(_JOB_DESCRIPTION_SIGNALS)
        if re.search(pattern, normalized, re.IGNORECASE)
    ]
    explicit_label = 0 in hits or 1 in hits
    return len(hits) >= 3 or (explicit_label and len(hits) >= 2)


def candidate_context_error(source_name: str) -> str:
    display_name = (
        "Candidate context"
        if source_name == "pasted_text"
        else f"'{source_name}'"
    )
    return (
        f"{display_name} appears to contain a job description rather than "
        "candidate evidence. Keep exactly one target posting in the Job "
        "Description field. Candidate sources should contain only real work "
        "history, projects, skills, education, certifications, or achievements."
    )
