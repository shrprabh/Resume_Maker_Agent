"""Central knobs for the resume pipeline.

Everything tunable lives here so switching models (or porting to LangChain
later) touches one file.
"""

# Pin the model so an overloaded or behavior-changing `*-latest` alias cannot
# break the whole multi-agent pipeline without a code change. This model was
# verified against the configured Gemini account during setup.
# Swap to "gemini-pro-latest" for the writer/strategist if you have paid
# quota (pro's free-tier RPM is usually much lower).
MODEL = "gemini-3.5-flash"

APP_NAME = "resume_maker"

# Reviewer approves (calls exit_loop) at or above this score.
QUALITY_THRESHOLD = 85
MIN_SUPPORTED_ATS_COVERAGE = 85

# The optional evidence-maximized branch has a slightly higher publication
# bar because its language is intentionally more assertive.
MAXIMUM_MATCH_THRESHOLD = 90

# Hard cap on writer -> reviewer revision cycles. Each cycle costs 2 LLM
# calls; free tier is ~10 requests/min, so keep this small.
MAX_REVISION_LOOPS = 3
MAXIMUM_MATCH_REVISION_LOOPS = 2

# ---------------------------------------------------------------------------
# Session-state keys. These are the "wires" between agents: each agent writes
# its result to state via output_key and later agents read it via {placeholder}
# templating in their instruction. The FastAPI layer seeds the first two.
# ---------------------------------------------------------------------------
STATE_JD_TEXT = "jd_text"                    # seeded: raw job description
STATE_CANDIDATE_TEXT = "candidate_text"      # seeded: all candidate docs, concatenated
STATE_JD_ANALYSIS = "jd_analysis"            # written by jd_analyzer
STATE_CANDIDATE_PROFILE = "candidate_profile"  # written by profile_analyzer
STATE_MATCH_STRATEGY = "match_strategy"      # written by match_strategist
STATE_DRAFT_RESUME = "draft_resume"          # written by resume_writer (overwritten each loop pass)
STATE_REVIEW_FEEDBACK = "review_feedback"    # written by quality_reviewer
STATE_COVER_LETTER = "cover_letter"          # written by cover_letter_writer
STATE_MAXIMUM_MATCH_RESUME = "maximum_match_resume"
STATE_MAXIMUM_MATCH_FEEDBACK = "maximum_match_feedback"
