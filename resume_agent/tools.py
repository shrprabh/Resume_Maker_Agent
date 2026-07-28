"""Tools available to the agents.

`exit_loop` is the canonical ADK pattern for breaking out of a LoopAgent:
a sub-agent calls this tool, the tool flips `escalate` on the invocation
context, and the LoopAgent stops iterating instead of running until
max_iterations.
"""

from google.adk.tools.tool_context import ToolContext


def exit_loop(tool_context: ToolContext) -> dict:
    """Approve the resume and stop the revision loop.

    Call this ONLY when the resume scores at or above the quality bar AND
    contains zero fabricated claims. Do not call it otherwise.
    """
    # escalate=True tells the enclosing LoopAgent to stop iterating.
    tool_context.actions.escalate = True
    return {"status": "approved"}
