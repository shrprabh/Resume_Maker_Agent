"""Optional Google ADK branch for an evidence-maximized resume.

It is a separate root so opening the Maximum Verified Match tab is the only
action that spends its model calls. The canonical six-agent application
pipeline remains unchanged.
"""

from google.adk.agents import LlmAgent, LoopAgent

from . import config, prompts
from .tools import exit_loop


maximum_match_writer = LlmAgent(
    name="maximum_match_writer",
    model=config.MODEL,
    description=(
        "Produces the strongest ATS match possible inside the candidate's "
        "verified evidence boundary."
    ),
    instruction=prompts.MAXIMUM_MATCH_WRITER_INSTRUCTION,
    output_key=config.STATE_MAXIMUM_MATCH_RESUME,
)

maximum_match_reviewer = LlmAgent(
    name="maximum_match_reviewer",
    model=config.MODEL,
    description=(
        "Traces every maximum-match claim to evidence and audits supported "
        "keyword placement."
    ),
    instruction=prompts.MAXIMUM_MATCH_REVIEWER_INSTRUCTION,
    tools=[exit_loop],
    output_key=config.STATE_MAXIMUM_MATCH_FEEDBACK,
)

maximum_match_root_agent = LoopAgent(
    name="maximum_match_pipeline",
    description="Writes and audits an evidence-maximized ATS resume.",
    sub_agents=[maximum_match_writer, maximum_match_reviewer],
    max_iterations=config.MAXIMUM_MATCH_REVISION_LOOPS,
)
