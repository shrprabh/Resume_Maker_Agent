"""LangGraph + OpenRouter implementation of the resume multi-agent pipeline.

This is intentionally independent of the Google ADK runner. Both engines use
the same specialist prompts and return the same result contract, making the
FastAPI route, PDF renderer, and frontend provider-agnostic.
"""

import asyncio
import logging
import re
import uuid
from typing import Annotated, Any, Literal

import langsmith as ls
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter
from langgraph.graph import END, START, StateGraph
from langsmith import Client
from typing_extensions import TypedDict

from resume_agent import config, prompts
from .resume_scoring import (
    ReviewerDecision,
    build_maximum_match_insights,
    build_scorecard,
    canonical_markdown_heading,
    normalize_experience_chronology,
    normalize_skill_category_markdown,
    parse_reviewer_decision,
)

DEFAULT_OPENROUTER_MODEL = "openai/gpt-4.1-mini"
logger = logging.getLogger(__name__)

# Per-agent output ceilings prevent verbose reasoning models from consuming
# thousands of unnecessary completion tokens. They are output limits only;
# each model still receives the full resume/JD context.
_OUTPUT_TOKEN_BUDGETS = {
    "jd_analyzer": 2_400,
    "profile_analyzer": 7_500,
    "match_strategist": 9_000,
    "resume_writer": 7_000,
    "quality_reviewer": 1_500,
    "cover_letter_writer": 3_200,
    "maximum_match_writer": 7_000,
    "maximum_match_reviewer": 1_500,
}

# OpenRouter counts hidden reasoning against max_tokens. Analytical stages
# benefit from a small reasoning allowance; document writers need that budget
# for visible, complete content. Per-call settings also avoid one model-wide
# policy being wasteful or truncating output for a different specialist.
_REASONING_EFFORTS = {
    "jd_analyzer": "minimal",
    "profile_analyzer": "minimal",
    "match_strategist": "minimal",
    "resume_writer": "none",
    "quality_reviewer": "none",
    "cover_letter_writer": "none",
    "maximum_match_writer": "none",
    "maximum_match_reviewer": "none",
}


def _merge_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    keys = set(left) | set(right)
    return {key: int(left.get(key, 0)) + int(right.get(key, 0)) for key in keys}


class PipelineState(TypedDict, total=False):
    job_description: str
    candidate_text: str
    jd_analysis: str
    candidate_profile: str
    match_strategy: str
    draft_resume: str
    review_feedback: str
    cover_letter: str
    cover_letter_error: str
    review_score: int | None
    ats_coverage: int | None
    fabrication_count: int
    review_valid: bool
    approved: bool
    revision_count: int
    maximum_match_resume: str
    maximum_match_feedback: str
    usage: Annotated[dict[str, int], _merge_usage]


def _fill_instruction(template: str, state: PipelineState) -> str:
    """Resolve ADK state placeholders without changing the canonical prompts."""
    values = {
        "jd_text": state.get("job_description", ""),
        "candidate_text": state.get("candidate_text", ""),
        "jd_analysis": state.get("jd_analysis", ""),
        "candidate_profile": state.get("candidate_profile", ""),
        "match_strategy": state.get("match_strategy", ""),
        "draft_resume": state.get("draft_resume", ""),
        "review_feedback": state.get("review_feedback", ""),
        "maximum_match_resume": state.get("maximum_match_resume", ""),
        "maximum_match_feedback": state.get("maximum_match_feedback", ""),
    }
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"{{{key}?}}", value)
        rendered = rendered.replace(f"{{{key}}}", value)
    return rendered


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        ).strip()
    return str(content).strip()


def _missing_sections(markdown_text: str, required: tuple[str, ...]) -> list[str]:
    headings = {
        canonical_markdown_heading(match)
        for match in re.findall(
            r"^##\s+(.+?)\s*$",
            markdown_text or "",
            flags=re.MULTILINE,
        )
    }
    return [
        heading
        for heading in required
        if canonical_markdown_heading(heading) not in headings
    ]


def _require_sections(
    artifact_name: str,
    markdown_text: str,
    required: tuple[str, ...],
) -> None:
    missing = _missing_sections(markdown_text, required)
    if missing:
        raise ValueError(
            f"{artifact_name} returned incomplete output; missing sections: "
            + ", ".join(missing)
            + ". Choose a model with reliable long-form output."
        )


def _usage(message: Any) -> dict[str, int]:
    raw = getattr(message, "usage_metadata", None) or {}
    if not raw:
        metadata = getattr(message, "response_metadata", None) or {}
        raw = metadata.get("token_usage") or metadata.get("usage") or {}
    input_tokens = raw.get("input_tokens", raw.get("prompt_tokens", 0))
    output_tokens = raw.get("output_tokens", raw.get("completion_tokens", 0))
    total_tokens = raw.get("total_tokens", 0)
    if not total_tokens:
        total_tokens = int(input_tokens or 0) + int(output_tokens or 0)
    return {
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "total_tokens": int(total_tokens or 0),
    }


async def run_pipeline(
    *,
    job_description: str,
    candidate_text: str,
    openrouter_api_key: str,
    model_name: str,
    langsmith_enabled: bool = False,
    langsmith_api_key: str | None = None,
    langsmith_project: str = "rolefit-resume-agent",
    trace_content: bool = False,
) -> dict[str, Any]:
    session_id = uuid.uuid4().hex
    model = ChatOpenRouter(
        model=model_name,
        api_key=openrouter_api_key,
        temperature=0.1,
        max_tokens=max(_OUTPUT_TOKEN_BUDGETS.values()),
        reasoning={"effort": "minimal"},
        # langchain-openrouter uses milliseconds here. The old value of 300
        # timed out after 0.3 seconds, then retried paid requests whose
        # responses could no longer be consumed by this application.
        max_retries=0,
        timeout=300_000,
        app_title="RoleFit Resume Agent",
        app_url="http://127.0.0.1:8080",
    )

    async def invoke(
        state: PipelineState, instruction: str, run_name: str
    ) -> tuple[str, dict[str, int]]:
        rendered_instruction = _fill_instruction(instruction, state)
        output_budget = _OUTPUT_TOKEN_BUDGETS.get(run_name, 2_600)
        response = await model.bind(
            max_tokens=output_budget,
            reasoning={
                "effort": _REASONING_EFFORTS.get(run_name, "minimal")
            },
        ).ainvoke(
            [
                SystemMessage(content=rendered_instruction),
                HumanMessage(
                    content="Execute your assigned specialist task now. "
                    "Return only the requested output."
                ),
            ],
            config={
                "run_name": run_name,
                "tags": ["rolefit", "resume-agent", model_name],
                "metadata": {
                    "engine": "langgraph_openrouter",
                    "prompt_chars": len(rendered_instruction),
                    "max_output_tokens": output_budget,
                },
            },
        )
        return _message_text(response), _usage(response)

    async def invoke_reviewer(
        state: PipelineState,
        instruction: str,
        run_name: str,
    ) -> tuple[ReviewerDecision | None, dict[str, int]]:
        """Request portable JSON text and enforce the schema locally.

        A single OpenRouter slug can be routed to providers with different
        JSON-schema/tool capabilities. Local Pydantic validation keeps the
        strict contract without letting a provider reject a completed resume.
        """
        rendered_instruction = _fill_instruction(instruction, state)
        output_budget = _OUTPUT_TOKEN_BUDGETS.get(run_name, 1_200)
        response = await model.bind(
            max_tokens=output_budget,
            reasoning={
                "effort": _REASONING_EFFORTS.get(run_name, "none")
            },
        ).ainvoke(
            [
                SystemMessage(content=rendered_instruction),
                HumanMessage(
                    content=(
                        "Execute the audit now. Return only one valid JSON "
                        "object matching the requested schema, with no fence "
                        "or commentary."
                    )
                ),
            ],
            config={
                "run_name": run_name,
                "tags": ["rolefit", "resume-agent", model_name, "review"],
                "metadata": {
                    "engine": "langgraph_openrouter",
                    "prompt_chars": len(rendered_instruction),
                    "max_output_tokens": output_budget,
                    "structured_output": False,
                    "local_schema_validation": True,
                },
            },
        )
        return parse_reviewer_decision(_message_text(response)), _usage(response)

    async def jd_analyzer(state: PipelineState) -> dict:
        text, usage = await invoke(
            state, prompts.JD_ANALYZER_INSTRUCTION, "jd_analyzer"
        )
        _require_sections(
            "JD analyzer",
            text,
            (
                "Target Company",
                "Target Role",
                "Role Summary",
                "Must-Have Requirements",
                "ATS Keywords (verbatim)",
                "Key Responsibilities",
            ),
        )
        return {"jd_analysis": text, "usage": usage}

    async def profile_analyzer(state: PipelineState) -> dict:
        text, usage = await invoke(
            state, prompts.PROFILE_ANALYZER_INSTRUCTION, "profile_analyzer"
        )
        _require_sections(
            "Profile analyzer",
            text,
            (
                "Contact",
                "Work History",
                "Skills Inventory",
                "Education & Certifications",
                "Conflicts & Gaps",
            ),
        )
        return {"candidate_profile": text, "usage": usage}

    async def match_strategist(state: PipelineState) -> dict:
        text, usage = await invoke(
            state, prompts.MATCH_STRATEGIST_INSTRUCTION, "match_strategist"
        )
        _require_sections(
            "Match strategist",
            text,
            (
                "Requirement-to-Evidence Map",
                "Genuine Gaps (do not paper over)",
                "Positioning Strategy",
                "Keyword Placement Plan",
                "Do-Not-Claim List",
            ),
        )
        return {"match_strategy": text, "usage": usage}

    async def resume_writer(state: PipelineState) -> dict:
        text, usage = await invoke(
            state, prompts.RESUME_WRITER_INSTRUCTION, "resume_writer"
        )
        text = normalize_skill_category_markdown(
            normalize_experience_chronology(text)
        )
        return {
            "draft_resume": text,
            "revision_count": int(state.get("revision_count", 0)) + 1,
            "usage": usage,
        }

    async def quality_reviewer(state: PipelineState) -> dict:
        reviewer_instruction = (
            prompts.QUALITY_REVIEWER_INSTRUCTION
            + """

LANGGRAPH OUTPUT OVERRIDE
This runtime uses a conditional graph edge instead of the exit_loop tool.
Do NOT call a tool and ignore the tool-call response format above. Return ONLY
one valid JSON object with this exact shape:
{
  "score": 0,
  "ats_coverage": 0,
  "fabrication_count": 0,
  "approved": false,
  "feedback": ["specific edit 1", "specific edit 2"]
}
`approved` may be true only when score is at least 85 and fabrication_count
is zero. When approved, feedback must be an empty list.
"""
        )
        decision, usage = await invoke_reviewer(
            state, reviewer_instruction, "quality_reviewer"
        )
        if decision is None:
            # A formatting failure is not a real zero. Stop the loop so an
            # unusable review cannot trigger more paid rewrites or overwrite
            # an otherwise useful draft.
            scorecard = build_scorecard(
                resume_markdown=state.get("draft_resume", ""),
                jd_analysis=state.get("jd_analysis", ""),
                match_strategy=state.get("match_strategy", ""),
                reviewer=None,
            )
            return {
                "review_score": None,
                "ats_coverage": scorecard.supported_ats_coverage,
                "fabrication_count": 0,
                "review_valid": False,
                "approved": False,
                "review_feedback": (
                    "REVIEW UNAVAILABLE — the model returned an invalid "
                    "review format. The draft was preserved and no false "
                    "0/100 score was assigned."
                ),
                "usage": usage,
            }

        scorecard = build_scorecard(
            resume_markdown=state.get("draft_resume", ""),
            jd_analysis=state.get("jd_analysis", ""),
            match_strategy=state.get("match_strategy", ""),
            reviewer=decision,
        )
        score = (
            scorecard.quality_score
            if scorecard.quality_score is not None
            else decision.score
        )
        coverage = (
            scorecard.supported_ats_coverage
            if scorecard.supported_ats_coverage is not None
            else decision.ats_coverage
        )
        fabrications = decision.fabrication_count
        feedback_items = list(decision.feedback)
        feedback_items.extend(scorecard.structure_issues)
        if (
            coverage is not None
            and coverage < config.MIN_SUPPORTED_ATS_COVERAGE
        ):
            if scorecard.missing_supported_keywords:
                feedback_items.append(
                    "Place these evidence-supported ATS terms naturally: "
                    + ", ".join(scorecard.missing_supported_keywords)
                )
            else:
                feedback_items.append(
                    "Rebuild the resume with the strategy-approved ATS terms; "
                    f"deterministic supported coverage is only {coverage}%."
                )
        approved = decision.approved and (
            score >= config.QUALITY_THRESHOLD
            and fabrications == 0
            and scorecard.structure_valid
            and (
                coverage is None
                or coverage >= config.MIN_SUPPORTED_ATS_COVERAGE
            )
        )
        if approved:
            feedback = (
                f"APPROVED — score {score}/100, ATS coverage {coverage}%."
            )
        else:
            numbered = "\n".join(
                f"{index}. {item}"
                for index, item in enumerate(feedback_items, start=1)
            )
            feedback = (
                f"SCORE: {score}/100 | ATS coverage: {coverage}% | "
                f"Fabrications: {fabrications}\n{numbered}"
            ).strip()
        return {
            "review_score": score,
            "ats_coverage": coverage,
            "fabrication_count": fabrications,
            "review_valid": True,
            "approved": approved,
            "review_feedback": feedback,
            "usage": usage,
        }

    def review_route(
        state: PipelineState,
    ) -> Literal["resume_writer", "__end__"]:
        if state.get("review_valid") is False:
            return END
        if state.get("approved") or int(state.get("revision_count", 0)) >= (
            config.MAX_REVISION_LOOPS
        ):
            return END
        return "resume_writer"

    refinement_builder = StateGraph(PipelineState)
    refinement_builder.add_node("resume_writer", resume_writer)
    refinement_builder.add_node("quality_reviewer", quality_reviewer)
    refinement_builder.add_edge(START, "resume_writer")
    refinement_builder.add_edge("resume_writer", "quality_reviewer")
    refinement_builder.add_conditional_edges("quality_reviewer", review_route)
    refinement_graph = refinement_builder.compile()

    async def refinement_loop(state: PipelineState) -> dict:
        child_state: PipelineState = {**state, "usage": {}}
        result = await refinement_graph.ainvoke(
            child_state, {"recursion_limit": 12, "run_name": "refinement_loop"}
        )
        return {
            "draft_resume": result.get("draft_resume", ""),
            "review_feedback": result.get("review_feedback", ""),
            "review_score": result.get("review_score", 0),
            "ats_coverage": result.get("ats_coverage", 0),
            "fabrication_count": result.get("fabrication_count", 0),
            "review_valid": result.get("review_valid", False),
            "approved": result.get("approved", False),
            "revision_count": result.get("revision_count", 0),
            # The child graph has already aggregated all writer/reviewer calls.
            "usage": result.get("usage", {}),
        }

    async def cover_letter_writer(state: PipelineState) -> dict:
        try:
            text, usage = await invoke(
                state,
                prompts.COVER_LETTER_WRITER_INSTRUCTION,
                "cover_letter_writer",
            )
            return {
                "cover_letter": text,
                "cover_letter_error": "",
                "usage": usage,
            }
        except Exception as exc:
            # The cover letter is valuable but must not discard an already
            # successful resume/refinement branch during provider instability.
            logger.warning(
                "Cover-letter branch failed for model %s: %s",
                model_name,
                type(exc).__name__,
            )
            return {
                "cover_letter": "",
                "cover_letter_error": (
                    "The cover-letter model call failed. The resume was "
                    "preserved; retry with a reliable model to generate the letter."
                ),
            }

    async def package_results(_: PipelineState) -> dict:
        return {}

    builder = StateGraph(PipelineState)
    builder.add_node("jd_analyzer", jd_analyzer)
    builder.add_node("profile_analyzer", profile_analyzer)
    builder.add_node("match_strategist", match_strategist)
    builder.add_node("refinement_loop", refinement_loop)
    builder.add_node("cover_letter_writer", cover_letter_writer)
    builder.add_node("package_results", package_results)
    builder.add_edge(START, "jd_analyzer")
    builder.add_edge(START, "profile_analyzer")
    builder.add_edge(["jd_analyzer", "profile_analyzer"], "match_strategist")
    builder.add_edge("match_strategist", "refinement_loop")
    builder.add_edge("match_strategist", "cover_letter_writer")
    builder.add_edge(
        ["refinement_loop", "cover_letter_writer"], "package_results"
    )
    builder.add_edge("package_results", END)
    graph = builder.compile()

    langsmith_client: Client | None = None
    if langsmith_enabled:
        if not langsmith_api_key:
            raise ValueError(
                "A LangSmith API key is required when tracing is enabled"
            )
        langsmith_client = Client(
            api_key=langsmith_api_key,
            hide_inputs=not trace_content,
            hide_outputs=not trace_content,
            hide_metadata=False,
        )
        tracing = ls.tracing_context(
            enabled=True,
            project_name=langsmith_project,
            tags=["rolefit", "resume-generation", "langgraph-openrouter"],
            metadata={
                "engine": "langgraph_openrouter",
                "model": model_name,
                "session_id": session_id,
                "content_redacted": not trace_content,
            },
            client=langsmith_client,
        )
    else:
        # An environment-level LANGSMITH_TRACING=true must not override the
        # per-generation UI switch.
        tracing = ls.tracing_context(enabled=False)

    initial_state: PipelineState = {
        "job_description": job_description,
        "candidate_text": candidate_text,
        "review_feedback": "",
        "revision_count": 0,
        "approved": False,
        "usage": {},
    }
    try:
        with tracing:
            result = await graph.ainvoke(
                initial_state,
                {
                    "recursion_limit": 20,
                    "run_name": "rolefit_resume_pipeline",
                    "tags": ["rolefit", model_name],
                },
            )
        if langsmith_client:
            await asyncio.to_thread(langsmith_client.flush, 5)
    finally:
        if langsmith_client:
            await asyncio.to_thread(langsmith_client.close, 2)

    return {
        "session_id": session_id,
        "approved": bool(result.get("approved")),
        "resume_markdown": result.get("draft_resume", ""),
        "cover_letter_markdown": result.get("cover_letter", ""),
        "cover_letter_error": result.get("cover_letter_error", ""),
        "jd_analysis": result.get("jd_analysis", ""),
        "candidate_profile": result.get("candidate_profile", ""),
        "match_strategy": result.get("match_strategy", ""),
        "review_feedback": result.get("review_feedback", ""),
        "review_score": result.get("review_score"),
        "ats_coverage": result.get("ats_coverage"),
        "review_valid": result.get("review_valid", False),
        "revision_count": int(result.get("revision_count", 0) or 0),
        "usage": result.get("usage", {}),
        "engine": "langgraph_openrouter",
        "model_name": model_name,
        "langsmith_enabled": langsmith_enabled,
        "langsmith_project": langsmith_project if langsmith_enabled else None,
        "trace_content": trace_content if langsmith_enabled else False,
    }


async def run_maximum_match(
    *,
    jd_analysis: str,
    candidate_profile: str,
    match_strategy: str,
    openrouter_api_key: str,
    model_name: str,
    langsmith_enabled: bool = False,
    langsmith_api_key: str | None = None,
    langsmith_project: str = "rolefit-resume-agent",
    trace_content: bool = False,
) -> dict[str, Any]:
    """Generate the optional evidence-maximized resume without rerunning analysis."""
    session_id = uuid.uuid4().hex
    model = ChatOpenRouter(
        model=model_name,
        api_key=openrouter_api_key,
        temperature=0.1,
        max_tokens=max(_OUTPUT_TOKEN_BUDGETS.values()),
        reasoning={"effort": "minimal"},
        max_retries=0,
        timeout=300_000,
        app_title="RoleFit Resume Agent",
        app_url="http://127.0.0.1:8080",
    )

    async def invoke(
        state: PipelineState, instruction: str, run_name: str
    ) -> tuple[str, dict[str, int]]:
        rendered_instruction = _fill_instruction(instruction, state)
        output_budget = _OUTPUT_TOKEN_BUDGETS.get(run_name, 2_600)
        response = await model.bind(
            max_tokens=output_budget,
            reasoning={
                "effort": _REASONING_EFFORTS.get(run_name, "minimal")
            },
        ).ainvoke(
            [
                SystemMessage(content=rendered_instruction),
                HumanMessage(
                    content="Execute the assigned specialist task now. "
                    "Return only the requested output."
                ),
            ],
            config={
                "run_name": run_name,
                "tags": [
                    "rolefit",
                    "maximum-verified-match",
                    model_name,
                ],
                "metadata": {
                    "engine": "langgraph_openrouter",
                    "prompt_chars": len(rendered_instruction),
                    "max_output_tokens": output_budget,
                    "branch": "maximum_verified_match",
                },
            },
        )
        return _message_text(response), _usage(response)

    async def invoke_reviewer(
        state: PipelineState,
        instruction: str,
        run_name: str,
    ) -> tuple[ReviewerDecision | None, dict[str, int]]:
        rendered_instruction = _fill_instruction(instruction, state)
        output_budget = _OUTPUT_TOKEN_BUDGETS.get(run_name, 1_000)
        response = await model.bind(
            max_tokens=output_budget,
            reasoning={
                "effort": _REASONING_EFFORTS.get(run_name, "none")
            },
        ).ainvoke(
            [
                SystemMessage(content=rendered_instruction),
                HumanMessage(
                    content=(
                        "Execute the audit now. Return only one valid JSON "
                        "object matching the requested schema, with no fence "
                        "or commentary."
                    )
                ),
            ],
            config={
                "run_name": run_name,
                "tags": [
                    "rolefit",
                    "maximum-verified-match",
                    model_name,
                    "review",
                ],
                "metadata": {
                    "engine": "langgraph_openrouter",
                    "prompt_chars": len(rendered_instruction),
                    "max_output_tokens": output_budget,
                    "branch": "maximum_verified_match",
                    "structured_output": False,
                    "local_schema_validation": True,
                },
            },
        )
        return parse_reviewer_decision(_message_text(response)), _usage(response)

    langsmith_client: Client | None = None
    if langsmith_enabled:
        if not langsmith_api_key:
            raise ValueError(
                "A LangSmith API key is required when tracing is enabled"
            )
        langsmith_client = Client(
            api_key=langsmith_api_key,
            hide_inputs=not trace_content,
            hide_outputs=not trace_content,
            hide_metadata=False,
        )
        tracing = ls.tracing_context(
            enabled=True,
            project_name=langsmith_project,
            tags=[
                "rolefit",
                "maximum-verified-match",
                "langgraph-openrouter",
            ],
            metadata={
                "engine": "langgraph_openrouter",
                "model": model_name,
                "session_id": session_id,
                "branch": "maximum_verified_match",
                "content_redacted": not trace_content,
            },
            client=langsmith_client,
        )
    else:
        tracing = ls.tracing_context(enabled=False)

    usage: dict[str, int] = {}
    maximum_resume = ""
    feedback = ""
    reviewer: ReviewerDecision | None = None
    scorecard = None
    revision_count = 0
    approved = False

    reviewer_instruction = (
        prompts.MAXIMUM_MATCH_REVIEWER_INSTRUCTION
        + """

LANGGRAPH OUTPUT OVERRIDE
This runtime uses Python routing instead of the exit_loop tool. Do not call a
tool. Return ONLY one valid JSON object:
{
  "score": 0,
  "ats_coverage": 0,
  "fabrication_count": 0,
  "approved": false,
  "feedback": ["specific correction 1"]
}
Set approved=true only when the score is at least 90 and there are zero
unsupported claims. Use an empty feedback list when approved.
"""
    )

    try:
        with tracing:
            for revision_count in range(
                1, config.MAXIMUM_MATCH_REVISION_LOOPS + 1
            ):
                state: PipelineState = {
                    "jd_analysis": jd_analysis,
                    "candidate_profile": candidate_profile,
                    "match_strategy": match_strategy,
                    "maximum_match_resume": maximum_resume,
                    "maximum_match_feedback": feedback,
                }
                maximum_resume, writer_usage = await invoke(
                    state,
                    prompts.MAXIMUM_MATCH_WRITER_INSTRUCTION,
                    "maximum_match_writer",
                )
                maximum_resume = normalize_skill_category_markdown(
                    normalize_experience_chronology(maximum_resume)
                )
                usage = _merge_usage(usage, writer_usage)
                state["maximum_match_resume"] = maximum_resume

                reviewer, reviewer_usage = await invoke_reviewer(
                    state,
                    reviewer_instruction,
                    "maximum_match_reviewer",
                )
                usage = _merge_usage(usage, reviewer_usage)
                scorecard = build_scorecard(
                    resume_markdown=maximum_resume,
                    jd_analysis=jd_analysis,
                    match_strategy=match_strategy,
                    reviewer=reviewer,
                )

                if reviewer is None:
                    feedback = (
                        "REVIEW UNAVAILABLE — the reviewer returned an invalid "
                        "format. The generated resume was preserved."
                    )
                    break

                approved = (
                    reviewer.fabrication_count == 0
                    and (
                        scorecard.quality_score is not None
                        and scorecard.quality_score
                        >= config.MAXIMUM_MATCH_THRESHOLD
                    )
                    and scorecard.structure_valid
                    and (
                        scorecard.supported_ats_coverage is None
                        or scorecard.supported_ats_coverage >= 95
                    )
                )
                if approved:
                    feedback = (
                        f"APPROVED — score {reviewer.score}/100, ATS coverage "
                        f"{scorecard.supported_ats_coverage}%, "
                        "Fabrications: 0."
                    )
                    break

                corrections = list(reviewer.feedback)
                corrections.extend(scorecard.structure_issues)
                if scorecard.missing_supported_keywords:
                    corrections.append(
                        "Place these supported terms naturally in their "
                        "strategy-approved locations: "
                        + ", ".join(scorecard.missing_supported_keywords)
                    )
                feedback = "\n".join(
                    f"{index}. {item}"
                    for index, item in enumerate(corrections, start=1)
                ) or (
                    "Keep every supported claim, remove any unsupported "
                    "language, and improve requirement positioning."
                )

        if langsmith_client:
            await asyncio.to_thread(langsmith_client.flush, 5)
    finally:
        if langsmith_client:
            await asyncio.to_thread(langsmith_client.close, 2)

    if scorecard is None:
        scorecard = build_scorecard(
            resume_markdown=maximum_resume,
            jd_analysis=jd_analysis,
            match_strategy=match_strategy,
            reviewer=reviewer,
        )
    insights = build_maximum_match_insights(
        scorecard,
        match_strategy,
        feedback,
    )
    return {
        "session_id": session_id,
        "approved": approved,
        "resume_markdown": maximum_resume,
        "review_feedback": feedback,
        "scores": scorecard.model_dump(),
        "insights_markdown": insights,
        "revision_count": revision_count,
        "usage": usage,
        "engine": "langgraph_openrouter",
        "model_name": model_name,
        "langsmith_enabled": langsmith_enabled,
        "langsmith_project": langsmith_project if langsmith_enabled else None,
        "trace_content": trace_content if langsmith_enabled else False,
    }
