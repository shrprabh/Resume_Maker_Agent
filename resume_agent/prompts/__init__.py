"""Instruction prompts for the six agents, one module per agent.

Every prompt follows the same 5-block skeleton — ROLE / INPUTS / TASK /
OUTPUT FORMAT / CONSTRAINTS — so they are easy to compare, tune, and later
port to LangChain prompt templates.

`{placeholder}` tokens are ADK instruction templating: at call time ADK
replaces them with the value stored in session state under that key.
A trailing `?` (e.g. `{jd_text?}`) makes the key optional — it renders as
empty instead of raising when the key is missing, which keeps the agents
usable in `adk web` where state is not pre-seeded.
"""

from .jd_analyzer import JD_ANALYZER_INSTRUCTION
from .profile_analyzer import PROFILE_ANALYZER_INSTRUCTION
from .match_strategist import MATCH_STRATEGIST_INSTRUCTION
from .resume_writer import RESUME_WRITER_INSTRUCTION
from .quality_reviewer import QUALITY_REVIEWER_INSTRUCTION
from .cover_letter_writer import COVER_LETTER_WRITER_INSTRUCTION
from .maximum_match_reviewer import MAXIMUM_MATCH_REVIEWER_INSTRUCTION
from .maximum_match_writer import MAXIMUM_MATCH_WRITER_INSTRUCTION

__all__ = [
    "JD_ANALYZER_INSTRUCTION",
    "PROFILE_ANALYZER_INSTRUCTION",
    "MATCH_STRATEGIST_INSTRUCTION",
    "RESUME_WRITER_INSTRUCTION",
    "QUALITY_REVIEWER_INSTRUCTION",
    "COVER_LETTER_WRITER_INSTRUCTION",
    "MAXIMUM_MATCH_WRITER_INSTRUCTION",
    "MAXIMUM_MATCH_REVIEWER_INSTRUCTION",
]
