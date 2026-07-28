"""Typed control-flow tools used by the Google ADK review agents.

The review decision must be one atomic operation.  Asking a model to call an
exit tool and then emit separately parsed prose creates two sources of truth:
the loop may exit even though the prose is missing, or stale prose from an
earlier pass may remain in session state.  These tools accept the complete
decision as typed arguments, persist canonical writer-compatible feedback, and
only then decide whether the enclosing loop may stop.
"""

from google.adk.tools.tool_context import ToolContext

from . import config


def _validated_percent(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a whole number from 0 through 100")
    if not 0 <= value <= 100:
        raise ValueError(f"{name} must be from 0 through 100")
    return value


def _validated_fabrication_count(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("fabrication_count must be a non-negative whole number")
    return value


def _clean_feedback(feedback: list[str]) -> list[str]:
    if not isinstance(feedback, list):
        raise ValueError("feedback must be a list of actionable corrections")
    return [
        " ".join(str(item).split())
        for item in feedback
        if " ".join(str(item).split())
    ]


def _validated_approval(value: bool) -> bool:
    if not isinstance(value, bool):
        raise ValueError("approved must be true or false")
    return value


def _submit_review(
    *,
    score: int,
    ats_coverage: int,
    fabrication_count: int,
    approved: bool,
    feedback: list[str],
    threshold: int,
    minimum_coverage: int,
    state_key: str,
    tool_context: ToolContext,
) -> dict[str, object]:
    score = _validated_percent("score", score)
    ats_coverage = _validated_percent("ats_coverage", ats_coverage)
    fabrication_count = _validated_fabrication_count(fabrication_count)
    approved = _validated_approval(approved)
    corrections = _clean_feedback(feedback)

    gate_issues: list[str] = []
    if score < threshold:
        gate_issues.append(
            f"Resolve the remaining audit findings; the review score is "
            f"{score}/100 and the approval minimum is {threshold}/100."
        )
    if ats_coverage < minimum_coverage:
        gate_issues.append(
            f"Place the remaining evidence-supported ATS terms naturally; "
            f"coverage is {ats_coverage}% and the minimum is "
            f"{minimum_coverage}%."
        )
    if fabrication_count:
        gate_issues.append(
            f"Remove or narrow all {fabrication_count} unsupported "
            "claim(s) identified by the audit."
        )

    if not approved and not corrections:
        raise ValueError(
            "feedback must include at least one actionable correction "
            "when approved is false"
        )

    accepted = approved and not gate_issues
    if accepted and corrections:
        raise ValueError("feedback must be empty when the review is approved")
    if not accepted:
        corrections.extend(
            issue for issue in gate_issues if issue not in corrections
        )
        if not corrections:
            raise ValueError(
                "feedback must include at least one actionable correction "
                "when the review is not approved"
            )

    if accepted:
        canonical = (
            f"APPROVED — score {score}/100, ATS coverage {ats_coverage}%, "
            "Fabrications: 0."
        )
    else:
        numbered = "\n".join(
            f"{index}. {item}"
            for index, item in enumerate(corrections, start=1)
        )
        canonical = (
            f"SCORE: {score}/100 | ATS coverage: {ats_coverage}% | "
            f"Fabrications: {fabrication_count}\n{numbered}"
        )

    # Persist the canonical decision before emitting either control-flow action.
    # skip_summarization makes this tool response the reviewer's terminal event,
    # avoiding a second model call that could overwrite or malform the result.
    tool_context.state[state_key] = canonical
    tool_context.actions.skip_summarization = True
    tool_context.actions.escalate = accepted
    return {
        "status": "approved" if accepted else "revision_required",
        "score": score,
        "ats_coverage": ats_coverage,
        "fabrication_count": fabrication_count,
        "approved": accepted,
        "feedback_count": 0 if accepted else len(corrections),
    }


def submit_quality_review(
    score: int,
    ats_coverage: int,
    fabrication_count: int,
    approved: bool,
    feedback: list[str],
    tool_context: ToolContext,
) -> dict[str, object]:
    """Submit the complete authentic-resume review exactly once.

    Args:
        score: Overall review score from 0 through 100.
        ats_coverage: Percentage of supported ATS terms present, from 0 to 100.
        fabrication_count: Number of unsupported or inflated claims found.
        approved: True only when every approval gate is satisfied.
        feedback: Actionable corrections; empty only for a valid approval.
        tool_context: Google ADK invocation context.
    """

    return _submit_review(
        score=score,
        ats_coverage=ats_coverage,
        fabrication_count=fabrication_count,
        approved=approved,
        feedback=feedback,
        threshold=config.QUALITY_THRESHOLD,
        minimum_coverage=config.MIN_SUPPORTED_ATS_COVERAGE,
        state_key=config.STATE_REVIEW_FEEDBACK,
        tool_context=tool_context,
    )


def submit_maximum_match_review(
    score: int,
    ats_coverage: int,
    fabrication_count: int,
    approved: bool,
    feedback: list[str],
    tool_context: ToolContext,
) -> dict[str, object]:
    """Submit the complete Maximum Verified Match review exactly once.

    Args:
        score: Overall review score from 0 through 100.
        ats_coverage: Percentage of supported ATS terms present, from 0 to 100.
        fabrication_count: Number of unsupported or inflated claims found.
        approved: True only when every approval gate is satisfied.
        feedback: Actionable corrections; empty only for a valid approval.
        tool_context: Google ADK invocation context.
    """

    return _submit_review(
        score=score,
        ats_coverage=ats_coverage,
        fabrication_count=fabrication_count,
        approved=approved,
        feedback=feedback,
        threshold=config.MAXIMUM_MATCH_THRESHOLD,
        minimum_coverage=95,
        state_key=config.STATE_MAXIMUM_MATCH_FEEDBACK,
        tool_context=tool_context,
    )
