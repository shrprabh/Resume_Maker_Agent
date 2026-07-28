"""The multi-agent resume pipeline — 6 LlmAgents wired with ADK workflow agents.

Tree (demonstrates all three ADK workflow agent types):

    resume_pipeline (SequentialAgent)          # stages run in order
    |
    |-- analysis_stage (ParallelAgent)         # independent -> run concurrently
    |     |-- jd_analyzer          -> state["jd_analysis"]
    |     `-- profile_analyzer     -> state["candidate_profile"]
    |
    |-- match_strategist           -> state["match_strategy"]
    |
    `-- production_stage (ParallelAgent)       # letter drafts WHILE resume loops
          |-- refinement_loop (LoopAgent, max 3 passes)
          |     |-- resume_writer  -> state["draft_resume"]
          |     `-- quality_reviewer -> typed review submission tool
          |           (writes review_feedback; stops only when approved)
          `-- cover_letter_writer  -> state["cover_letter"]

How data flows (the core ADK concept):
- `output_key` saves an agent's final text into session state.
- `{key}` placeholders in an instruction inject state back in at call time.
- All agents in one run share ONE session state, so parallel branches are
  safe only because they write DIFFERENT keys.
"""

from google.adk.agents import LlmAgent, LoopAgent, ParallelAgent, SequentialAgent
from google.genai import types

from . import config, prompts
from .tools import submit_quality_review

# --------------------------------------------------------------------------
# Stage 1 — two independent analyses, run in parallel to save wall-clock time
# --------------------------------------------------------------------------
jd_analyzer = LlmAgent(
    name="jd_analyzer",
    model=config.MODEL,
    description="Dissects the job description into requirements and verbatim ATS keywords.",
    instruction=prompts.JD_ANALYZER_INSTRUCTION,
    output_key=config.STATE_JD_ANALYSIS,
)

profile_analyzer = LlmAgent(
    name="profile_analyzer",
    model=config.MODEL,
    description="Merges all candidate documents into a deduplicated fact inventory.",
    instruction=prompts.PROFILE_ANALYZER_INSTRUCTION,
    output_key=config.STATE_CANDIDATE_PROFILE,
)

analysis_stage = ParallelAgent(
    name="analysis_stage",
    sub_agents=[jd_analyzer, profile_analyzer],
)

# --------------------------------------------------------------------------
# Stage 2 — strategy: consumes BOTH analyses, so it must run after stage 1
# --------------------------------------------------------------------------
match_strategist = LlmAgent(
    name="match_strategist",
    model=config.MODEL,
    description="Maps evidence to requirements and plans positioning, keywords, and the Do-Not-Claim list.",
    instruction=prompts.MATCH_STRATEGIST_INSTRUCTION,
    output_key=config.STATE_MATCH_STRATEGY,
)

# --------------------------------------------------------------------------
# Stage 3a — write/review loop: writer drafts, reviewer critiques or approves.
# The reviewer must submit one typed, complete decision. The tool persists
# canonical feedback and stops the LoopAgent only after every approval gate
# passes; otherwise the writer receives that feedback on the next pass.
# --------------------------------------------------------------------------
resume_writer = LlmAgent(
    name="resume_writer",
    model=config.MODEL,
    description="Writes (and on later passes revises) the tailored ATS-safe resume.",
    instruction=prompts.RESUME_WRITER_INSTRUCTION,
    output_key=config.STATE_DRAFT_RESUME,
)

quality_reviewer = LlmAgent(
    name="quality_reviewer",
    model=config.MODEL,
    description="Audits the draft for fabrication, ATS coverage, and craft; approves or returns numbered edits.",
    instruction=prompts.QUALITY_REVIEWER_INSTRUCTION,
    tools=[submit_quality_review],
    generate_content_config=types.GenerateContentConfig(
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode=types.FunctionCallingConfigMode.ANY,
                allowed_function_names=["submit_quality_review"],
            )
        )
    ),
)

refinement_loop = LoopAgent(
    name="refinement_loop",
    sub_agents=[resume_writer, quality_reviewer],
    max_iterations=config.MAX_REVISION_LOOPS,
)

# --------------------------------------------------------------------------
# Stage 3b — cover letter needs only the strategy, not the finished resume,
# so it runs in parallel with the refinement loop (free performance win).
# --------------------------------------------------------------------------
cover_letter_writer = LlmAgent(
    name="cover_letter_writer",
    model=config.MODEL,
    description="Writes a one-page cover letter from the strategy and fact inventory.",
    instruction=prompts.COVER_LETTER_WRITER_INSTRUCTION,
    output_key=config.STATE_COVER_LETTER,
)

production_stage = ParallelAgent(
    name="production_stage",
    sub_agents=[refinement_loop, cover_letter_writer],
)

# --------------------------------------------------------------------------
# Root — `adk web` and the FastAPI Runner both look for this exact name.
# --------------------------------------------------------------------------
root_agent = SequentialAgent(
    name="resume_pipeline",
    description="Generates an ATS-optimized resume and cover letter tailored to a job description.",
    sub_agents=[analysis_stage, match_strategist, production_stage],
)
