"""Bridges FastAPI to the ADK pipeline.

One module-level Runner + InMemorySessionService serve every request; each
request gets its own session, so per-request state never mixes. This is the
programmatic equivalent of what `adk web` does interactively.
"""

import uuid

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from resume_agent import config
from resume_agent.agent import root_agent
from resume_agent.maximum_agent import maximum_match_root_agent
from .resume_scoring import (
    build_maximum_match_insights,
    build_scorecard,
    normalize_experience_chronology,
    normalize_skill_category_markdown,
    parse_reviewer_decision,
)

USER_ID = "api_user"

_session_service = InMemorySessionService()
_runner = Runner(
    agent=root_agent,
    app_name=config.APP_NAME,
    session_service=_session_service,
)
_maximum_match_runner = Runner(
    agent=maximum_match_root_agent,
    app_name=config.APP_NAME,
    session_service=_session_service,
)


def _unverified_review_note(scorecard) -> str:
    coverage = scorecard.supported_ats_coverage
    coverage_text = f"{coverage}%" if coverage is not None else "not measurable"
    if scorecard.structure_valid:
        structure_text = "document structure passed"
    else:
        issues = "; ".join(scorecard.structure_issues[:3])
        structure_text = f"document structure needs attention ({issues})"
    return (
        "MODEL REVIEW NOT VERIFIED — Google ADK did not commit a complete "
        "typed decision, so the draft was preserved without inventing a "
        "quality or fabrication score. Deterministic checks: supported ATS "
        f"coverage {coverage_text}; {structure_text}. Inspect the evidence "
        "audit before using the resume."
    )


async def run_pipeline(job_description: str, candidate_text: str) -> dict:
    session_id = uuid.uuid4().hex

    # Seed the raw inputs into session state — the same channel the agents
    # use to hand work to each other ({placeholder} templating reads it).
    # review_feedback is seeded empty so the writer's first pass sees a
    # resolved placeholder instead of a missing key.
    await _session_service.create_session(
        app_name=config.APP_NAME,
        user_id=USER_ID,
        session_id=session_id,
        state={
            config.STATE_JD_TEXT: job_description,
            config.STATE_CANDIDATE_TEXT: candidate_text,
            config.STATE_REVIEW_FEEDBACK: "",
        },
    )

    # The message is only the trigger; the real inputs are already in state.
    trigger = types.Content(
        role="user",
        parts=[types.Part(text="Generate the tailored resume and cover letter now.")],
    )

    tool_approved = False
    revision_count = 0
    async for event in _runner.run_async(
        user_id=USER_ID, session_id=session_id, new_message=trigger
    ):
        # The typed review tool is the single source of truth. Count only its
        # successfully committed state update, not an attempted/invalid call.
        if (
            event.author == "quality_reviewer"
            and config.STATE_REVIEW_FEEDBACK
            in event.actions.state_delta
        ):
            revision_count += 1
        if (
            event.author == "quality_reviewer"
            and event.actions.escalate is True
        ):
            tool_approved = True

    # Read results from session state — NOT from the last event. After the
    # loop, the final event text is the reviewer's verdict (or the cover
    # letter), never the resume itself.
    session = await _session_service.get_session(
        app_name=config.APP_NAME, user_id=USER_ID, session_id=session_id
    )
    state = session.state
    draft_resume = normalize_skill_category_markdown(
        normalize_experience_chronology(
            state.get(config.STATE_DRAFT_RESUME, "")
        )
    )
    review_feedback = state.get(config.STATE_REVIEW_FEEDBACK, "")
    reviewer = parse_reviewer_decision(review_feedback)
    scorecard = build_scorecard(
        resume_markdown=draft_resume,
        jd_analysis=state.get(config.STATE_JD_ANALYSIS, ""),
        match_strategy=state.get(config.STATE_MATCH_STRATEGY, ""),
        reviewer=reviewer,
    )
    coverage = scorecard.supported_ats_coverage
    approved = (
        tool_approved
        and reviewer is not None
        and reviewer.approved
        and reviewer.fabrication_count == 0
        and scorecard.structure_valid
        and (
            scorecard.quality_score is not None
            and scorecard.quality_score >= config.QUALITY_THRESHOLD
        )
        and (
            coverage is None
            or coverage >= config.MIN_SUPPORTED_ATS_COVERAGE
        )
    )
    if reviewer is None:
        review_feedback = _unverified_review_note(scorecard)
    elif not approved:
        gate_issues = list(scorecard.structure_issues)
        if (
            coverage is not None
            and coverage < config.MIN_SUPPORTED_ATS_COVERAGE
        ):
            gate_issues.append(
                f"Supported ATS coverage is {coverage}%; "
                f"minimum is {config.MIN_SUPPORTED_ATS_COVERAGE}%."
            )
        if gate_issues:
            review_feedback = (
                review_feedback
                + "\nPOST-REVIEW VALIDATION:\n"
                + "\n".join(
                    f"{index}. {issue}"
                    for index, issue in enumerate(gate_issues, start=1)
                )
            ).strip()

    return {
        "session_id": session_id,
        "approved": approved,
        "resume_markdown": draft_resume,
        "cover_letter_markdown": state.get(config.STATE_COVER_LETTER, ""),
        "cover_letter_error": "",
        "jd_analysis": state.get(config.STATE_JD_ANALYSIS, ""),
        "candidate_profile": state.get(config.STATE_CANDIDATE_PROFILE, ""),
        "match_strategy": state.get(config.STATE_MATCH_STRATEGY, ""),
        "review_feedback": review_feedback,
        "review_score": scorecard.quality_score,
        "ats_coverage": coverage,
        "review_valid": reviewer is not None,
        "revision_count": revision_count,
        "usage": {},
        "engine": "google_adk",
        "model_name": config.MODEL,
        "langsmith_enabled": False,
        "langsmith_project": None,
        "trace_content": False,
    }


async def run_maximum_match(
    *,
    jd_analysis: str,
    candidate_profile: str,
    match_strategy: str,
) -> dict:
    """Run the optional ADK maximum-match writer/reviewer loop."""
    session_id = uuid.uuid4().hex
    await _session_service.create_session(
        app_name=config.APP_NAME,
        user_id=USER_ID,
        session_id=session_id,
        state={
            config.STATE_JD_ANALYSIS: jd_analysis,
            config.STATE_CANDIDATE_PROFILE: candidate_profile,
            config.STATE_MATCH_STRATEGY: match_strategy,
            config.STATE_MAXIMUM_MATCH_FEEDBACK: "",
        },
    )
    trigger = types.Content(
        role="user",
        parts=[
            types.Part(
                text="Generate and audit the Maximum Verified Match resume now."
            )
        ],
    )

    tool_approved = False
    revision_count = 0
    async for event in _maximum_match_runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=trigger,
    ):
        if (
            event.author == "maximum_match_reviewer"
            and config.STATE_MAXIMUM_MATCH_FEEDBACK
            in event.actions.state_delta
        ):
            revision_count += 1
        if (
            event.author == "maximum_match_reviewer"
            and event.actions.escalate is True
        ):
            tool_approved = True

    session = await _session_service.get_session(
        app_name=config.APP_NAME,
        user_id=USER_ID,
        session_id=session_id,
    )
    state = session.state
    maximum_resume = normalize_skill_category_markdown(
        normalize_experience_chronology(
            state.get(config.STATE_MAXIMUM_MATCH_RESUME, "")
        )
    )
    review_feedback = state.get(config.STATE_MAXIMUM_MATCH_FEEDBACK, "")
    reviewer = parse_reviewer_decision(review_feedback)
    scorecard = build_scorecard(
        resume_markdown=maximum_resume,
        jd_analysis=jd_analysis,
        match_strategy=match_strategy,
        reviewer=reviewer,
    )
    approved = (
        tool_approved
        and reviewer is not None
        and reviewer.approved
        and reviewer.fabrication_count == 0
        and scorecard.structure_valid
        and (
            scorecard.quality_score is not None
            and scorecard.quality_score >= config.MAXIMUM_MATCH_THRESHOLD
        )
        and (
            scorecard.supported_ats_coverage is None
            or scorecard.supported_ats_coverage >= 95
        )
    )
    if reviewer is None:
        review_feedback = _unverified_review_note(scorecard)
    elif not approved:
        gate_issues = list(scorecard.structure_issues)
        if (
            scorecard.supported_ats_coverage is not None
            and scorecard.supported_ats_coverage < 95
        ):
            gate_issues.append(
                "Supported ATS coverage is "
                f"{scorecard.supported_ats_coverage}%; minimum is 95%."
            )
        if gate_issues:
            review_feedback = (
                review_feedback
                + "\nPOST-REVIEW VALIDATION:\n"
                + "\n".join(
                    f"{index}. {issue}"
                    for index, issue in enumerate(gate_issues, start=1)
                )
            ).strip()
    return {
        "session_id": session_id,
        "approved": approved,
        "resume_markdown": maximum_resume,
        "review_feedback": review_feedback,
        "scores": scorecard.model_dump(),
        "insights_markdown": build_maximum_match_insights(
            scorecard,
            match_strategy,
            review_feedback,
        ),
        "revision_count": revision_count,
        "usage": {},
        "engine": "google_adk",
        "model_name": config.MODEL,
        "langsmith_enabled": False,
        "langsmith_project": None,
        "trace_content": False,
    }
