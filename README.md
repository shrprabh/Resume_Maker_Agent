# Resume Maker Agent

A dual-engine multi-agent platform that tailors a candidate's resume and
cover letter to a job description. Choose the existing **Google ADK + Gemini**
pipeline or an equivalent **LangGraph + OpenRouter** pipeline, with optional
privacy-safe **LangSmith** tracing. FastAPI accepts resume/context uploads and
returns Markdown plus rendered PDFs. A lazy **Maximum Verified Match** branch
can then generate a second, more assertive resume using every strategy-approved
keyword while preserving the same evidence boundary.

The pipeline is deliberately structured to demonstrate every core ADK multi-agent concept: `SequentialAgent`, `ParallelAgent`, `LoopAgent`, session-state wiring via `output_key` + `{placeholder}` templating, and early loop exit via tool escalation.

The LangGraph engine mirrors the same six specialist roles using graph state,
parallel edges, a conditional writer/reviewer loop, and a shared result
contract. OpenRouter provides the selected chat model; LangSmith is optional
observability rather than a model provider.

## Architecture

```
resume_pipeline (SequentialAgent)              stages run in order
│
├── analysis_stage (ParallelAgent)             independent → run concurrently
│   ├── 1. jd_analyzer        → state["jd_analysis"]        requirements + verbatim ATS keywords
│   └── 2. profile_analyzer   → state["candidate_profile"]  fact inventory (single source of truth)
│
├── 3. match_strategist       → state["match_strategy"]     evidence map, gaps, Do-Not-Claim list
│
└── production_stage (ParallelAgent)           letter drafts WHILE the resume loops
    ├── refinement_loop (LoopAgent, max 3 passes)
    │   ├── 4. resume_writer  → state["draft_resume"]       drafts, then revises per feedback
    │   └── 5. quality_reviewer → state["review_feedback"]  audits; calls exit_loop to approve
    └── 6. cover_letter_writer → state["cover_letter"]

maximum_match_pipeline (on demand; reuses completed analyses)
└── refinement_loop (maximum 2 passes)
    ├── maximum_match_writer
    └── maximum_match_reviewer
```

**The anti-fabrication chain** is the key prompt-design idea: the profile analyzer builds a facts-only inventory → the strategist emits an explicit **Do-Not-Claim list** (JD keywords with no supporting evidence) → the writer may only use inventory facts → the reviewer traces every claim back to the inventory and any fabrication caps the score at 40, forcing a revision pass.

## How data flows (the core ADK lesson)

- Every agent declares an `output_key`; ADK saves the agent's final text into shared **session state** under that key.
- Later agents pull values into their instructions with `{key}` placeholders (`{key?}` = optional, renders empty instead of raising — that's what keeps the agents usable in `adk web`, where state isn't pre-seeded).
- The FastAPI layer seeds `jd_text`, `candidate_text`, and an empty `review_feedback` at `create_session(state=...)`; the user message is just a trigger.
- The reviewer approves by calling the `exit_loop` tool, which sets `tool_context.actions.escalate = True` — the canonical ADK pattern for stopping a `LoopAgent` early.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# put your Google AI Studio key (https://aistudio.google.com/app/apikey) in:
#   resume_agent/.env
# GOOGLE_API_KEY=...
# GOOGLE_GENAI_USE_VERTEXAI=FALSE
```

## Run

**Interactive dev UI** (great for watching the pipeline — Events tab shows the parallel fan-out and loop passes, State tab shows every output_key):

```bash
adk web            # from the project root → http://localhost:8000
```

In the chat, paste one message containing both the JD and resume text (the stage-1 prompts fall back to the user message when state isn't seeded).

**API server** (port 8080 on purpose, so it can run alongside `adk web`):

```bash
uvicorn app.main:app --reload --port 8080
```

Open **http://127.0.0.1:8080** for the integrated frontend. It supports
drag-and-drop multi-document uploads, pasted candidate context, live pipeline
progress, resume and cover-letter review tabs, agent insights, and direct PDF
downloads. Before model tokens are spent, a source preflight shows the exact
normalized text read from each PDF, DOCX, TXT, Markdown knowledge file, and
pasted note, including detected sections, page/word counts, truncation, and
OCR warnings. Generation reuses that exact reviewed text through a one-hour
opaque source bundle. A request accepts up to 20 files and 15 MB per source.
The results also include an on-demand **Maximum Verified Match**
tab with separate supported-keyword coverage, overall requirement match, and
evidence-integrity scores. The engine selector keeps Google ADK as the default or reveals
OpenRouter model/key and LangSmith tracing controls. Provider keys are sent in
request headers, held only in the browser tab, and are never returned by the
API. Swagger remains available at **http://127.0.0.1:8080/docs**.

### Run with Docker

No source checkout or Python installation is required:

```bash
docker pull ghcr.io/shrprabh/resume-maker-agent:latest

docker run --rm -p 8080:8080 \
  -e OPENROUTER="sk-or-..." \
  -e LANGSMITH_API_KEY="lsv2_..." \
  -e LANGSMITH_PROJECT="ResumeMaker" \
  ghcr.io/shrprabh/resume-maker-agent:latest
```

Then open **http://127.0.0.1:8080**. Google ADK is optional; add
`-e GOOGLE_API_KEY="..." -e GOOGLE_GENAI_USE_VERTEXAI=FALSE` when you want to
use the Gemini engine.

For safer local secret handling, create a `.env` file and pass it without
putting keys in shell history:

```bash
docker run --rm -p 8080:8080 --env-file .env \
  ghcr.io/shrprabh/resume-maker-agent:latest
```

Or use the included Compose file:

```bash
docker compose up -d
```

The container runs as a non-root user, exposes port `8080`, includes a health
check, and never embeds local `.env` files in the image.

### LangGraph + OpenRouter

1. Select **LangGraph** in the frontend.
2. Paste an OpenRouter API key.
3. The quality-first default is
   `openai/gpt-4.1-mini`; replace it or click
   **Load models** to choose another catalog model.
4. Optionally enable LangSmith and enter its separate API key/project.
5. Keep the default metadata-only privacy mode unless you explicitly want
   resume and JD content recorded in traces.

The existing endpoint remains backward-compatible:

```text
POST /api/resume/sources/preflight
POST /api/resume/generate
X-OpenRouter-Api-Key: <key>       # LangGraph engine only
X-LangSmith-Api-Key: <key>        # only when tracing is enabled

engine=google_adk | langgraph_openrouter
model_name=<openrouter model slug>
langsmith_enabled=false
langsmith_project=rolefit-resume-agent
trace_content=false
source_bundle_id=<id returned by source preflight>

# returned by the generate response; valid for one hour in the local server
POST /api/resume/maximum-match/<session_id>
```

The maximum-match endpoint reuses the completed JD analysis, fact inventory,
and match strategy. It does not rerun the first three agents, does not rerun
the cover letter, locks duplicate clicks to one model run, and caches the
result for the exact evidence submission. Before generation, the Maximum
Verified Match tab lists genuine gaps and JD-keyword exclusions. A candidate
can leave each item protected or add attested work/product evidence with its
source, role, dates, contribution, outcome, and optional link. Evidence is
validated without a model call, added to a provenance-labeled fact section,
and may override only the matching gap. Editing evidence changes the cache
signature and triggers a fresh audited result; the authentic resume is never
modified. The commercial persistence plan replaces this bounded in-memory
context with a per-user job record.

Gap-evidence endpoints:

```text
GET  /api/resume/maximum-match/<session_id>/gaps
POST /api/resume/maximum-match/<session_id>/evidence/validate
POST /api/resume/maximum-match/<session_id>
```

The two POST requests accept:

```json
{
  "evidence": [{
    "gap_id": "<id from the gaps endpoint>",
    "source_type": "work_experience",
    "source_name": "Example Company",
    "role_or_contribution": "Software Engineer",
    "dates": "January 2025 - May 2025",
    "evidence_text": "Built and deployed...",
    "outcome": "Reduced deployment setup time...",
    "reference_url": "https://example.com",
    "candidate_attested": true
  }]
}
```

Provider helpers:

```text
GET /api/providers
GET /api/providers/openrouter/models
GET /api/providers/openrouter/validate
GET /api/providers/langsmith/validate
```

```bash
# generate — upload ANY mix of documents: existing resumes, project notes,
# a knowledge library. Repeat -F 'files=@...' per document; they are merged
# and deduplicated into one fact inventory by the profile_analyzer agent.
# (Single-file `resume_file=@...` and plain `resume_text=...` also work.)
curl -X POST http://localhost:8080/api/resume/generate \
  -F 'files=@samples/sample_resume.txt' \
  -F 'files=@samples/sample_knowledge_library.md' \
  -F "job_description=$(cat samples/sample_jd.txt)"

# download the rendered PDFs (URLs also returned in the JSON response)
curl -O http://localhost:8080/api/resume/download/<session_id>/resume
curl -O http://localhost:8080/api/resume/download/<session_id>/cover_letter
```

Response JSON: `resume_markdown`, `cover_letter_markdown`, `approved` (did the reviewer call exit_loop?), `artifacts` (every intermediate agent output — great for demos), and the two PDF URLs. Errors: 400 no resume input, 422 unsupported/unextractable file or blank JD, 503 Gemini rate limit.

Reviewer output is validated before it becomes a displayed score. Invalid
reviewer formatting is shown as **Unavailable**, never silently converted to
0/100. Exact supported-keyword coverage is calculated deterministically:

- **Supported ATS coverage:** claimable strategy keywords present in the draft.
- **Overall requirement match:** must-have requirements with supporting evidence.
- **Evidence integrity:** claim-auditor result; unavailable if its response
  cannot be validated.

The application does not claim that a numerical score guarantees an employer
ATS result.

Before any model call, candidate sources that strongly resemble a second job
description are rejected so hiring requirements cannot be mistaken for
candidate evidence. PDF pages retain boundaries, DOCX table content is read
in document order, and aggregate limits use visible fair head-and-tail
allocation instead of silently cutting off a later Education section.
Before any PDF is created, a deterministic publication pass canonicalizes
Education headings, restores missing Education verbatim from the verified
candidate profile, and removes only complete low-priority bullets when a
draft exceeds the 900-word publication target. Already placed,
evidence-supported ATS terms are protected. The final document gate still
rejects true fragments, missing required sections with no verified source,
fewer than four achievement bullets, drafts over 950 words after safe repair,
and unfocused Skills sections. Model outputs with more than five valid Skills categories are
losslessly consolidated into five professional ATS-safe rows before scoring
and PDF rendering. OpenRouter
reasoning is disabled for document-writing calls so hidden reasoning tokens
cannot truncate the visible resume. Experience blocks are normalized into
reverse chronological order without changing their text.

PDF downloads use an ATS-safe single-column layout, readable density-aware
type, hanging-indented hyphen bullets, headings kept with their first bullet,
and page-breakable long role blocks that avoid large blank areas. Downloads use
descriptive filenames:
`Company_Name_Candidate_Name_Resume.pdf` and
`Company_Name_Candidate_Name_Cover_Letter.pdf`. The optional variant uses
`Company_Name_Candidate_Name_Maximum_Match_Resume.pdf`.

## Project layout

```
resume_agent/        the ADK package — self-contained, no FastAPI imports;
                     this is the unit you'd port to LangChain
  config.py          model name, state keys, quality threshold, loop cap
  tools.py           exit_loop (escalate=True)
  agent.py           6 LlmAgents + workflow tree → root_agent
  maximum_agent.py   optional ADK writer/reviewer loop
  prompts/           one instruction module per agent
                     (ROLE / INPUTS / TASK / OUTPUT FORMAT / CONSTRAINTS skeleton)
app/                 FastAPI layer
  static/                      responsive browser frontend (HTML/CSS/JS)
  services/adk_runner.py       Runner + InMemorySessionService bridge
  services/langgraph_runner.py LangGraph state, parallel agents, review loop,
                               maximum-match branch, OpenRouter model, and
                               LangSmith tracing
  services/resume_repair.py    grounded Education restoration + whole-bullet
                               publication compaction
  services/resume_scoring.py   deterministic coverage + validated reviews
  services/text_extraction.py  structured PDF/DOCX/TXT/MD text + metadata
  services/pdf_renderer.py     Markdown → HTML → PDF (xhtml2pdf, pure Python)
  routers/resume.py            source preflight, generate, download, health
  routers/providers.py         OpenRouter catalog and credential validation
samples/             sample resume + knowledge library + JD for testing
                     (the JD demands Kubernetes, which no candidate document
                     has — a built-in anti-fabrication test; the knowledge
                     library adds Grafana/Prometheus facts the resume lacks,
                     testing the multi-document merge)
tests/               score-contract and maximum-match endpoint tests
```

## ADK → LangChain mapping (for the planned port)

| ADK concept | LangChain / LangGraph equivalent |
|---|---|
| `LlmAgent(instruction=..., output_key=...)` | An LCEL chain: `prompt \| llm`, result written into graph state |
| Session state + `{key}` templating | LangGraph `State` dict + prompt input variables |
| `SequentialAgent` | LangGraph nodes connected by ordered edges |
| `ParallelAgent` | LangGraph fan-out (multiple edges from one node) |
| `LoopAgent` + `escalate` | LangGraph conditional edge looping back until a flag/count |
| `exit_loop` tool | Conditional-edge function reading a "approved" state flag |
| `Runner` + `InMemorySessionService` | `graph.compile(checkpointer=MemorySaver())` + `invoke` |
| `adk web` | LangSmith / LangGraph Studio tracing |

## Gotchas worth knowing

- **The last event is not the resume.** After the loop, the final agent text is the reviewer's verdict (or cover letter). Always read `session.state["draft_resume"]` — see `adk_runner.py`.
- **Model availability changes.** The config pins `gemini-3.5-flash`, verified
  with the configured account, so a moving `*-latest` alias cannot silently
  switch the pipeline to an overloaded or behavior-changing model.
- **Parallel branches share one session state** — safe only because each writes a *different* output_key.
- **Free-tier rate limits:** one request costs up to ~9 Gemini calls (~10 RPM allowed). Keep `MAX_REVISION_LOOPS` small; the API maps 429s to a 503 response.
- **Scanned/image-only PDFs** extract as empty text and are rejected with 422 (no OCR in the pipeline).
- **`InMemorySessionService` is ephemeral** — fine for a stateless API; swap in `DatabaseSessionService` for persistence in production.
