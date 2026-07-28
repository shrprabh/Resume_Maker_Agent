import asyncio
import logging
import math
import os
import re
import time
import unicodedata
import uuid
from dataclasses import dataclass

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse

from ..schemas.resume import (
    HealthResponse,
    MaximumMatchEvidenceRequest,
    MaximumMatchEvidenceValidationResponse,
    MaximumMatchGapsResponse,
    MatchScorecard,
    MaximumMatchResponse,
    PipelineArtifacts,
    ResumeGenerateResponse,
    SourceManifestItem,
    SourcePreflightResponse,
)
from ..services import adk_runner, langgraph_runner, pdf_renderer
from ..services.credentials import normalize_api_key, normalize_env_text
from ..services.gap_evidence import (
    augment_profile_with_gap_evidence,
    augment_strategy_with_gap_evidence,
    evidence_signature,
    evidence_validation_rows,
    extract_maximum_match_gaps,
    resolved_gap_names,
    validate_gap_evidence,
)
from ..services.input_validation import (
    candidate_context_error,
    looks_like_job_description,
)
from ..services.resume_repair import repair_resume_for_publication
from ..services.resume_scoring import (
    build_maximum_match_insights,
    build_scorecard,
    normalize_experience_chronology,
    normalize_skill_category_markdown,
    parse_reviewer_decision,
)
from ..services.text_extraction import ExtractionResult, extract_text_result

router = APIRouter(prefix="/api/resume", tags=["resume"])
logger = logging.getLogger(__name__)

_DOC_FILES = {
    "resume": "resume_{sid}.pdf",
    "maximum_match": "maximum_match_{sid}.pdf",
    "cover_letter": "cover_{sid}.pdf",
}
_DOWNLOAD_NAMES: dict[str, dict[str, str]] = {}


@dataclass
class _MaximumMatchContext:
    jd_analysis: str
    candidate_profile: str
    match_strategy: str
    candidate_name: str
    company_name: str
    engine: str
    model_name: str
    langsmith_enabled: bool
    langsmith_project: str | None
    trace_content: bool
    created_at: float
    response: MaximumMatchResponse | None = None
    response_signature: str | None = None


@dataclass
class _SourceBundle:
    """Exact, candidate-reviewed source context prepared without model calls."""

    candidate_text: str
    source_manifest: list[SourceManifestItem]
    created_at: float


@dataclass
class _PreparedSources:
    candidate_text: str
    source_manifest: list[SourceManifestItem]
    ready: bool
    warnings: list[str]


# Current local deployment is intentionally ephemeral. Keeping only compact
# analyses (not raw uploads) lets the optional tab reuse completed work without
# paying for the first three agents again. The commercial persistence plan
# replaces this with a per-user job record.
_MAXIMUM_CONTEXT_TTL_SECONDS = 60 * 60
_MAXIMUM_CONTEXT_LIMIT = 64
_MAXIMUM_CONTEXTS: dict[str, _MaximumMatchContext] = {}
_MAXIMUM_LOCKS: dict[str, asyncio.Lock] = {}

_SOURCE_BUNDLE_TTL_SECONDS = 60 * 60
_SOURCE_BUNDLE_LIMIT = 64
_SOURCE_BUNDLES: dict[str, _SourceBundle] = {}


def _prune_source_bundles() -> None:
    cutoff = time.monotonic() - _SOURCE_BUNDLE_TTL_SECONDS
    expired = [
        bundle_id
        for bundle_id, bundle in _SOURCE_BUNDLES.items()
        if bundle.created_at < cutoff
    ]
    for bundle_id in expired:
        _SOURCE_BUNDLES.pop(bundle_id, None)
    while len(_SOURCE_BUNDLES) > _SOURCE_BUNDLE_LIMIT:
        oldest_id = min(
            _SOURCE_BUNDLES,
            key=lambda bundle_id: _SOURCE_BUNDLES[bundle_id].created_at,
        )
        _SOURCE_BUNDLES.pop(oldest_id, None)


def _source_bundle(bundle_id: str) -> _SourceBundle:
    """Resolve an opaque source token without revealing whether it was valid."""

    _prune_source_bundles()
    bundle = _SOURCE_BUNDLES.get(bundle_id.strip())
    if bundle is None:
        raise HTTPException(
            status_code=410,
            detail=(
                "This source preview expired or is unavailable. Check the "
                "uploaded source text again before generating."
            ),
        )
    return bundle


def _prune_maximum_contexts() -> None:
    cutoff = time.monotonic() - _MAXIMUM_CONTEXT_TTL_SECONDS
    expired = [
        sid
        for sid, context in _MAXIMUM_CONTEXTS.items()
        if context.created_at < cutoff
    ]
    for sid in expired:
        _MAXIMUM_CONTEXTS.pop(sid, None)
        _MAXIMUM_LOCKS.pop(sid, None)
    while len(_MAXIMUM_CONTEXTS) > _MAXIMUM_CONTEXT_LIMIT:
        oldest_sid = min(
            _MAXIMUM_CONTEXTS,
            key=lambda sid: _MAXIMUM_CONTEXTS[sid].created_at,
        )
        _MAXIMUM_CONTEXTS.pop(oldest_sid, None)
        _MAXIMUM_LOCKS.pop(oldest_sid, None)


def _maximum_context(session_id: str) -> _MaximumMatchContext:
    if not session_id.isalnum():
        raise HTTPException(status_code=404, detail="Unknown generation")
    _prune_maximum_contexts()
    context = _MAXIMUM_CONTEXTS.get(session_id)
    if context is None:
        raise HTTPException(
            status_code=410,
            detail=(
                "This generation context expired. Generate the authentic "
                "resume again, then reopen Maximum Verified Match."
            ),
        )
    return context


def _validated_evidence(
    context: _MaximumMatchContext,
    payload: MaximumMatchEvidenceRequest | None,
):
    gaps = extract_maximum_match_gaps(
        context.match_strategy,
        context.jd_analysis,
    )
    try:
        evidence = validate_gap_evidence(
            gaps,
            payload.evidence if payload else [],
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return gaps, evidence


def _extract_section(markdown_text: str, heading: str) -> str | None:
    pattern = rf"^##\s+{re.escape(heading)}\s*$\n+([^\n#]+)"
    match = re.search(pattern, markdown_text, flags=re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else None


def _candidate_name(resume_markdown: str) -> str:
    match = re.search(r"^#\s+(.+?)\s*$", resume_markdown, flags=re.MULTILINE)
    return match.group(1).strip() if match else "Candidate"


def _company_name(jd_analysis: str) -> str:
    explicit = _extract_section(jd_analysis, "Target Company")
    if explicit:
        return explicit
    # Backward-compatible fallback for analyses created before Target Company
    # became a required section.
    summary = _extract_section(jd_analysis, "Role Summary") or ""
    match = re.search(
        r"\bat\s+([A-Z][A-Za-z0-9&.,'’ -]+?)(?=\s+(?:is|seeks|needs|hiring|will)\b|[,.])",
        summary,
    )
    return match.group(1).strip(" .,") if match else "Target Company"


def _filename_part(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", ascii_value).strip("_")
    return (cleaned or "Unknown")[:80]


def _openrouter_key(header_value: str | None) -> str:
    key = normalize_api_key(header_value, "OPENROUTER_API_KEY")
    return key or normalize_api_key(
        os.getenv("OPENROUTER") or os.getenv("OPENROUTER_API_KEY"),
        "OPENROUTER_API_KEY",
    )


def _langsmith_key(header_value: str | None) -> str:
    key = normalize_api_key(header_value, "LANGSMITH_API_KEY")
    return key or normalize_api_key(
        os.getenv("LANGSMITH_API_KEY"),
        "LANGSMITH_API_KEY",
    )


def _validate_openrouter_credentials(
    *,
    openrouter_api_key: str,
    langsmith_enabled: bool,
    langsmith_api_key: str,
) -> None:
    if not openrouter_api_key:
        raise HTTPException(
            status_code=422,
            detail="OpenRouter API key is required for the LangGraph engine",
        )
    if not openrouter_api_key.startswith("sk-or-"):
        raise HTTPException(
            status_code=422,
            detail=(
                "OpenRouter inference keys normally start with 'sk-or-'. "
                "Copy the key from openrouter.ai/keys."
            ),
        )
    if langsmith_enabled and not langsmith_api_key:
        raise HTTPException(
            status_code=422,
            detail="LangSmith API key is required when tracing is enabled",
        )


def _pipeline_http_exception(exc: Exception) -> HTTPException:
    message = str(exc)
    lowered = message.lower()
    if "401" in message or "authentication" in lowered:
        return HTTPException(
            status_code=401,
            detail="OpenRouter or LangSmith rejected the supplied API key",
        )
    if "402" in message or "insufficient credits" in lowered:
        return HTTPException(
            status_code=402,
            detail="OpenRouter account has insufficient credits",
        )
    if "429" in message or "RESOURCE_EXHAUSTED" in message:
        return HTTPException(
            status_code=503,
            detail="Model provider rate limit hit — wait and retry",
        )
    if "503" in message or "UNAVAILABLE" in message:
        return HTTPException(
            status_code=503,
            detail="Model provider is temporarily unavailable — retry shortly",
        )
    if (
        "unavailable for free" in lowered
        or "notfoundresponseerror" in lowered
        or "model not found" in lowered
    ):
        paid_match = re.search(r"use this slug instead:\s*([^\s]+)", message)
        suggestion = (
            f" Use '{paid_match.group(1)}' instead." if paid_match else ""
        )
        return HTTPException(
            status_code=422,
            detail=f"The selected OpenRouter model is unavailable.{suggestion}",
        )
    if "readtimeout" in lowered or "timed out" in lowered:
        return HTTPException(
            status_code=504,
            detail=(
                "The selected OpenRouter model did not respond within five "
                "minutes. The request was not automatically retried, which "
                "prevents duplicate paid calls. Retry once or choose a "
                "faster available model."
            ),
        )
    safe_message = re.sub(
        r"(?:sk-or-|lsv2_)[A-Za-z0-9_-]+",
        "[REDACTED]",
        message,
    )
    safe_message = " ".join(safe_message.split())[:320]
    return HTTPException(
        status_code=502,
        detail=(
            "The model pipeline failed before producing a result"
            + (f": {safe_message}" if safe_message else ".")
        ),
    )


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


# More context = better fact inventory, but keep a ceiling so a huge document
# dump can't blow up latency/cost. Per-file cap lives in text_extraction.
MAX_TOTAL_CHARS = 150_000
MAX_SOURCE_FILES = 20
MAX_SOURCE_BYTES = 15 * 1024 * 1024
_WORD_RE = re.compile(r"\b[\w]+(?:[’'-][\w]+)*\b", re.UNICODE)
_OMISSION_MARKER = "\n\n[... middle source text omitted ...]\n\n"


def _source_name(value: str) -> str:
    """Make an upload name safe for both display and document separators."""

    cleaned = " ".join(
        unicodedata.normalize("NFC", value or "document")
        .replace("\x00", " ")
        .replace("=", " ")
        .split()
    )
    return (cleaned or "document")[:180]


def _fair_allocations(lengths: list[int], budget: int) -> list[int]:
    """Share a total character budget without letting one source crowd out others."""

    if not lengths:
        return []
    if sum(lengths) <= budget:
        return lengths.copy()
    allocations = [0] * len(lengths)
    active = set(range(len(lengths)))
    remaining = max(0, budget)
    while active and remaining:
        share = remaining // len(active)
        small = [index for index in active if lengths[index] <= share]
        if small:
            for index in small:
                allocations[index] = lengths[index]
                remaining -= lengths[index]
                active.remove(index)
            continue
        for index in sorted(active):
            allocations[index] = share
        remainder = remaining - share * len(active)
        for index in sorted(active)[:remainder]:
            allocations[index] += 1
        remaining = 0
    return allocations


def _head_tail_text(text: str, limit: int) -> str:
    """Keep both identity/summary and later Education sections when compacting."""

    if len(text) <= limit:
        return text
    if limit <= 0:
        return ""
    if limit <= len(_OMISSION_MARKER) + 40:
        return text[:limit].rstrip()
    available = limit - len(_OMISSION_MARKER)
    head_chars = (available * 2) // 3
    tail_chars = available - head_chars
    head = text[:head_chars]
    head_boundary = max(head.rfind("\n"), head.rfind(". "))
    if head_boundary >= int(head_chars * 0.8):
        head = head[: head_boundary + 1]
    head = head.rstrip()

    tail = text[-tail_chars:]
    newline_boundary = tail.find("\n")
    sentence_boundary = tail.find(". ")
    boundaries = [
        boundary
        for boundary in (newline_boundary, sentence_boundary)
        if 0 <= boundary <= int(tail_chars * 0.2)
    ]
    if boundaries:
        tail = tail[min(boundaries) + 1 :]
    tail = tail.lstrip()
    return f"{head}{_OMISSION_MARKER}{tail}"[:limit].rstrip()


async def _read_source_results(
    *,
    files: list[UploadFile],
    resume_file: UploadFile | None,
    resume_text: str | None,
) -> list[ExtractionResult]:
    results: list[ExtractionResult] = []
    uploads = [upload for upload in files if upload is not None]
    if resume_file is not None:
        uploads.append(resume_file)
    if len(uploads) > MAX_SOURCE_FILES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Upload at most {MAX_SOURCE_FILES} source documents per "
                "generation."
            ),
        )
    for upload in uploads:
        name = _source_name(upload.filename or "document")
        content = await upload.read(MAX_SOURCE_BYTES + 1)
        if len(content) > MAX_SOURCE_BYTES:
            suffix = name.rsplit(".", 1)[-1].casefold() if "." in name else ""
            results.append(
                ExtractionResult(
                    filename=name,
                    kind=(
                        suffix
                        if suffix in {"pdf", "docx", "txt", "md"}
                        else "unsupported"
                    ),
                    status="failed",
                    bytes=len(content),
                    pages=None,
                    extracted_chars=0,
                    words=0,
                    truncated=False,
                    warnings=[
                        f"File exceeds the {MAX_SOURCE_BYTES // (1024 * 1024)} MB "
                        "per-source upload limit."
                    ],
                    text="",
                    detected_sections=[],
                )
            )
            continue
        results.append(extract_text_result(name, content))

    if resume_text and resume_text.strip():
        results.append(
            extract_text_result(
                "Pasted candidate context.txt",
                resume_text.strip().encode("utf-8"),
            )
        )
    return results


def _prepare_sources(results: list[ExtractionResult]) -> _PreparedSources:
    if not results:
        return _PreparedSources(
            candidate_text="",
            source_manifest=[],
            ready=False,
            warnings=[
                "Provide at least one PDF, DOCX, TXT, MD, or pasted candidate context."
            ],
        )

    statuses: list[str] = []
    per_source_warnings: list[list[str]] = []
    for result in results:
        status = result.status
        warnings = list(result.warnings)
        if status == "ok" and looks_like_job_description(result.text):
            status = "rejected"
            warnings.append(candidate_context_error(result.filename))
        statuses.append(status)
        per_source_warnings.append(warnings)

    valid_indices = [
        index for index, status in enumerate(statuses) if status == "ok"
    ]
    multi_source = len(valid_indices) > 1
    separator_overhead = 0
    if multi_source:
        for position, index in enumerate(valid_indices):
            prefix = f"=== DOCUMENT: {_source_name(results[index].filename)} ===\n"
            separator_overhead += len(prefix)
            if position:
                separator_overhead += 2
    content_budget = max(0, MAX_TOTAL_CHARS - separator_overhead)
    allocations = _fair_allocations(
        [len(results[index].text) for index in valid_indices],
        content_budget,
    )
    included_by_index = {
        index: _head_tail_text(results[index].text, allocation)
        for index, allocation in zip(valid_indices, allocations, strict=True)
    }

    manifest: list[SourceManifestItem] = []
    for index, result in enumerate(results):
        included_text = included_by_index.get(index, result.text)
        inclusion_truncated = (
            statuses[index] == "ok"
            and len(included_text) < len(result.text)
        )
        warnings = per_source_warnings[index]
        if inclusion_truncated:
            warnings.append(
                "Only a fair head-and-tail share of this source was included "
                "because the combined knowledge base exceeds "
                f"{MAX_TOTAL_CHARS:,} characters."
            )
        manifest.append(
            SourceManifestItem(
                name=_source_name(result.filename),
                kind=result.kind,
                status=statuses[index],
                bytes=result.bytes,
                pages=result.pages,
                extracted_chars=result.extracted_chars,
                included_chars=(
                    len(included_text) if statuses[index] == "ok" else 0
                ),
                words=(
                    len(_WORD_RE.findall(included_text))
                    if statuses[index] == "ok"
                    else result.words
                ),
                truncated=result.truncated or inclusion_truncated,
                detected_sections=list(result.detected_sections),
                warnings=warnings,
                text=included_text,
            )
        )

    candidate_documents = [
        (item.name, item.text)
        for item in manifest
        if item.status == "ok" and item.text
    ]
    if len(candidate_documents) == 1:
        candidate_text = candidate_documents[0][1]
    else:
        candidate_text = "\n\n".join(
            f"=== DOCUMENT: {_source_name(name)} ===\n{text}"
            for name, text in candidate_documents
        )

    warnings: list[str] = []
    if any(item.truncated for item in manifest):
        warnings.append(
            "One or more sources were compacted. Review each extracted-text "
            "preview; the displayed text is exactly what generation will use."
        )
    if any(item.status != "ok" for item in manifest):
        warnings.append(
            "One or more sources could not be included. Remove or replace "
            "them before generating."
        )
    ready = bool(candidate_documents) and all(
        item.status == "ok" for item in manifest
    )
    return _PreparedSources(
        candidate_text=candidate_text,
        source_manifest=manifest,
        ready=ready,
        warnings=warnings,
    )


def _store_source_bundle(prepared: _PreparedSources) -> str:
    _prune_source_bundles()
    bundle_id = uuid.uuid4().hex
    _SOURCE_BUNDLES[bundle_id] = _SourceBundle(
        candidate_text=prepared.candidate_text,
        source_manifest=[
            SourceManifestItem.model_validate(item.model_dump())
            for item in prepared.source_manifest
        ],
        created_at=time.monotonic(),
    )
    _prune_source_bundles()
    return bundle_id


@router.post(
    "/sources/preflight",
    response_model=SourcePreflightResponse,
)
async def preflight_sources(
    files: list[UploadFile] = File(default=[]),
    resume_file: UploadFile | None = File(None),
    resume_text: str | None = Form(None),
) -> SourcePreflightResponse:
    """Extract and preview candidate sources without calling an AI model."""

    prepared = _prepare_sources(
        await _read_source_results(
            files=files,
            resume_file=resume_file,
            resume_text=resume_text,
        )
    )
    bundle_id = _store_source_bundle(prepared) if prepared.ready else None
    return SourcePreflightResponse(
        source_bundle_id=bundle_id,
        ready=prepared.ready,
        sources=prepared.source_manifest,
        total_chars=sum(
            item.included_chars for item in prepared.source_manifest
        ),
        total_words=sum(
            item.words
            for item in prepared.source_manifest
            if item.status == "ok"
        ),
        warnings=prepared.warnings,
        expires_in_minutes=math.ceil(_SOURCE_BUNDLE_TTL_SECONDS / 60),
    )


@router.post("/generate", response_model=ResumeGenerateResponse)
async def generate_resume(
    job_description: str = Form(...),
    files: list[UploadFile] = File(default=[]),
    resume_file: UploadFile | None = File(None),
    resume_text: str | None = Form(None),
    source_bundle_id: str | None = Form(None),
    engine: str = Form("google_adk"),
    model_name: str | None = Form(langgraph_runner.DEFAULT_OPENROUTER_MODEL),
    langsmith_enabled: bool = Form(False),
    langsmith_project: str = Form(""),
    trace_content: bool = Form(False),
    x_openrouter_api_key: str | None = Header(
        None, alias="X-OpenRouter-Api-Key"
    ),
    x_langsmith_api_key: str | None = Header(None, alias="X-LangSmith-Api-Key"),
) -> ResumeGenerateResponse:
    """Upload any mix of documents — existing resumes, project notes, a
    knowledge library — and they are merged into one candidate context for
    the profile_analyzer agent, which dedupes them into a fact inventory."""
    if not job_description.strip():
        raise HTTPException(status_code=422, detail="job_description must not be blank")

    if source_bundle_id and source_bundle_id.strip():
        bundle = _source_bundle(source_bundle_id)
        candidate_text = bundle.candidate_text
        source_manifest = [
            SourceManifestItem.model_validate(item.model_dump())
            for item in bundle.source_manifest
        ]
    else:
        prepared = _prepare_sources(
            await _read_source_results(
                files=files,
                resume_file=resume_file,
                resume_text=resume_text,
            )
        )
        if not prepared.source_manifest:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Provide at least one of: files, resume_file, resume_text, "
                    "or source_bundle_id"
                ),
            )
        rejected = next(
            (
                item
                for item in prepared.source_manifest
                if item.status != "ok"
            ),
            None,
        )
        if rejected is not None:
            if rejected.status == "unsupported":
                detail = (
                    f"'{rejected.name}': unsupported type — upload PDF, "
                    "DOCX, TXT, or MD"
                )
            elif rejected.status == "rejected":
                detail = (
                    rejected.warnings[-1]
                    if rejected.warnings
                    else candidate_context_error(rejected.name)
                )
            elif rejected.bytes == 0:
                detail = f"'{rejected.name}' is empty"
            elif rejected.warnings:
                detail = f"'{rejected.name}': {rejected.warnings[0]}"
            else:
                detail = (
                    f"'{rejected.name}': could not extract text "
                    "(scanned/image-only PDF?)"
                )
            raise HTTPException(
                status_code=400 if rejected.bytes == 0 else 422,
                detail=detail,
            )
        candidate_text = prepared.candidate_text
        source_manifest = prepared.source_manifest

    if engine not in {"google_adk", "langgraph_openrouter"}:
        raise HTTPException(
            status_code=422,
            detail="engine must be google_adk or langgraph_openrouter",
        )
    if len(langsmith_project) > 100:
        raise HTTPException(
            status_code=422, detail="langsmith_project is too long"
        )
    if engine == "langgraph_openrouter":
        openrouter_api_key = _openrouter_key(x_openrouter_api_key)
        langsmith_api_key = _langsmith_key(x_langsmith_api_key)
        _validate_openrouter_credentials(
            openrouter_api_key=openrouter_api_key,
            langsmith_enabled=langsmith_enabled,
            langsmith_api_key=langsmith_api_key,
        )

    try:
        if engine == "google_adk":
            result = await adk_runner.run_pipeline(
                job_description=job_description, candidate_text=candidate_text
            )
        else:
            selected_model = (
                model_name or langgraph_runner.DEFAULT_OPENROUTER_MODEL
            ).strip() or langgraph_runner.DEFAULT_OPENROUTER_MODEL
            result = await langgraph_runner.run_pipeline(
                job_description=job_description,
                candidate_text=candidate_text,
                openrouter_api_key=openrouter_api_key,
                model_name=selected_model,
                langsmith_enabled=langsmith_enabled,
                langsmith_api_key=langsmith_api_key or None,
                langsmith_project=langsmith_project.strip()
                or normalize_env_text(os.getenv("LANGSMITH_PROJECT"))
                or "rolefit-resume-agent",
                trace_content=trace_content,
            )
    except Exception as exc:
        logger.exception(
            "Resume pipeline failed (engine=%s, model=%s, error=%s)",
            engine,
            model_name,
            type(exc).__name__,
        )
        raise _pipeline_http_exception(exc) from exc

    result["resume_markdown"] = normalize_skill_category_markdown(
        normalize_experience_chronology(
            result.get("resume_markdown", "")
        )
    )
    publication_repair = repair_resume_for_publication(
        result["resume_markdown"],
        candidate_profile=result.get("candidate_profile", ""),
        jd_analysis=result.get("jd_analysis", ""),
        match_strategy=result.get("match_strategy", ""),
    )
    result["resume_markdown"] = publication_repair.markdown.strip()
    if not result["resume_markdown"].strip():
        raise HTTPException(
            status_code=502, detail="Pipeline produced no resume — check server logs"
        )
    reviewer = parse_reviewer_decision(result.get("review_feedback", ""))
    authentic_scorecard = build_scorecard(
        resume_markdown=result["resume_markdown"],
        jd_analysis=result["jd_analysis"],
        match_strategy=result["match_strategy"],
        reviewer=reviewer,
    )
    if not authentic_scorecard.structure_valid:
        issues = "; ".join(authentic_scorecard.structure_issues[:4])
        logger.warning(
            "Rejected incomplete resume before PDF rendering "
            "(engine=%s, model=%s, issues=%s)",
            result.get("engine"),
            result.get("model_name"),
            issues,
        )
        raise HTTPException(
            status_code=502,
            detail=(
                "The selected model returned an incomplete resume after its "
                f"revision passes: {issues}. No PDF was created. Retry with "
                "a reliable long-form model such as anthropic/claude-haiku-4.5, "
                "openai/gpt-4.1-mini, or google/gemini-2.5-flash-lite."
            ),
        )

    sid = result["session_id"]
    candidate_name = _candidate_name(result["resume_markdown"])
    company_name = _company_name(result["jd_analysis"])
    file_prefix = f"{_filename_part(company_name)}_{_filename_part(candidate_name)}"
    download_names = {
        "resume": f"{file_prefix}_Resume.pdf",
        "cover_letter": f"{file_prefix}_Cover_Letter.pdf",
    }
    _DOWNLOAD_NAMES[sid] = download_names

    # xhtml2pdf is synchronous/CPU-bound; keep the event loop free.
    await asyncio.to_thread(
        pdf_renderer.render_pdf,
        result["resume_markdown"],
        _DOC_FILES["resume"].format(sid=sid),
    )
    cover_letter_available = bool(result["cover_letter_markdown"].strip())
    if cover_letter_available:
        await asyncio.to_thread(
            pdf_renderer.render_pdf,
            result["cover_letter_markdown"],
            _DOC_FILES["cover_letter"].format(sid=sid),
        )

    _prune_maximum_contexts()
    _MAXIMUM_CONTEXTS[sid] = _MaximumMatchContext(
        jd_analysis=result["jd_analysis"],
        candidate_profile=result["candidate_profile"],
        match_strategy=result["match_strategy"],
        candidate_name=candidate_name,
        company_name=company_name,
        engine=result["engine"],
        model_name=result["model_name"],
        langsmith_enabled=result.get("langsmith_enabled", False),
        langsmith_project=result.get("langsmith_project"),
        trace_content=result.get("trace_content", False),
        created_at=time.monotonic(),
    )
    _MAXIMUM_LOCKS[sid] = asyncio.Lock()

    return ResumeGenerateResponse(
        resume_markdown=result["resume_markdown"],
        cover_letter_markdown=result["cover_letter_markdown"],
        artifacts=PipelineArtifacts(
            jd_analysis=result["jd_analysis"],
            candidate_profile=result["candidate_profile"],
            match_strategy=result["match_strategy"],
            review_feedback=result["review_feedback"],
        ),
        approved=result["approved"],
        resume_pdf_url=f"/api/resume/download/{sid}/resume",
        cover_letter_pdf_url=(
            f"/api/resume/download/{sid}/cover_letter"
            if cover_letter_available
            else ""
        ),
        resume_filename=download_names["resume"],
        cover_letter_filename=download_names["cover_letter"],
        candidate_name=candidate_name,
        company_name=company_name,
        engine=result["engine"],
        model_name=result["model_name"],
        review_score=result.get("review_score"),
        ats_coverage=result.get("ats_coverage"),
        review_valid=result.get("review_valid", reviewer is not None),
        scores=MatchScorecard.model_validate(authentic_scorecard.model_dump()),
        revision_count=result.get("revision_count"),
        usage=result.get("usage", {}),
        langsmith_enabled=result.get("langsmith_enabled", False),
        langsmith_project=result.get("langsmith_project"),
        trace_content=result.get("trace_content", False),
        warnings=(
            [result["cover_letter_error"]]
            if result.get("cover_letter_error")
            else []
        )
        + list(publication_repair.repair_notes)
        + [
            f"{item.name}: {warning}"
            for item in source_manifest
            for warning in item.warnings
        ],
        source_manifest=source_manifest,
        maximum_match_generate_url=f"/api/resume/maximum-match/{sid}",
        session_id=sid,
    )


@router.get(
    "/maximum-match/{session_id}/gaps",
    response_model=MaximumMatchGapsResponse,
)
async def maximum_match_gaps(
    session_id: str,
) -> MaximumMatchGapsResponse:
    """Return the candidate-visible gap checklist without spending tokens."""
    context = _maximum_context(session_id)
    gaps = extract_maximum_match_gaps(
        context.match_strategy,
        context.jd_analysis,
    )
    remaining_seconds = max(
        0,
        _MAXIMUM_CONTEXT_TTL_SECONDS
        - (time.monotonic() - context.created_at),
    )
    return MaximumMatchGapsResponse(
        session_id=session_id,
        gaps=gaps,
        expires_in_minutes=max(1, math.ceil(remaining_seconds / 60)),
    )


@router.post(
    "/maximum-match/{session_id}/evidence/validate",
    response_model=MaximumMatchEvidenceValidationResponse,
)
async def validate_maximum_match_evidence(
    session_id: str,
    payload: MaximumMatchEvidenceRequest,
) -> MaximumMatchEvidenceValidationResponse:
    """Validate and bind evidence to known gaps before any model call."""
    context = _maximum_context(session_id)
    gaps, evidence = _validated_evidence(context, payload)
    return MaximumMatchEvidenceValidationResponse(
        session_id=session_id,
        accepted=evidence_validation_rows(evidence),
        unresolved_gap_count=max(0, len(gaps) - len(evidence)),
        message=(
            f"{len(evidence)} evidence "
            f"{'item' if len(evidence) == 1 else 'items'} accepted. "
            "No model tokens have been used yet."
        ),
    )


@router.post(
    "/maximum-match/{session_id}",
    response_model=MaximumMatchResponse,
)
async def generate_maximum_match(
    session_id: str,
    payload: MaximumMatchEvidenceRequest | None = None,
    x_openrouter_api_key: str | None = Header(
        None, alias="X-OpenRouter-Api-Key"
    ),
    x_langsmith_api_key: str | None = Header(
        None, alias="X-LangSmith-Api-Key"
    ),
) -> MaximumMatchResponse:
    """Generate the optional evidence-maximized resume from cached analyses."""
    context = _maximum_context(session_id)
    _, evidence = _validated_evidence(context, payload)
    signature = evidence_signature(evidence)
    if (
        context.response is not None
        and context.response_signature == signature
    ):
        return context.response

    lock = _MAXIMUM_LOCKS.setdefault(session_id, asyncio.Lock())
    async with lock:
        if (
            context.response is not None
            and context.response_signature == signature
        ):
            return context.response

        candidate_profile = augment_profile_with_gap_evidence(
            context.candidate_profile,
            evidence,
        )
        match_strategy = augment_strategy_with_gap_evidence(
            context.match_strategy,
            evidence,
        )

        try:
            if context.engine == "google_adk":
                result = await adk_runner.run_maximum_match(
                    jd_analysis=context.jd_analysis,
                    candidate_profile=candidate_profile,
                    match_strategy=match_strategy,
                )
            else:
                openrouter_api_key = _openrouter_key(x_openrouter_api_key)
                langsmith_api_key = _langsmith_key(x_langsmith_api_key)
                _validate_openrouter_credentials(
                    openrouter_api_key=openrouter_api_key,
                    langsmith_enabled=context.langsmith_enabled,
                    langsmith_api_key=langsmith_api_key,
                )
                result = await langgraph_runner.run_maximum_match(
                    jd_analysis=context.jd_analysis,
                    candidate_profile=candidate_profile,
                    match_strategy=match_strategy,
                    openrouter_api_key=openrouter_api_key,
                    model_name=context.model_name,
                    langsmith_enabled=context.langsmith_enabled,
                    langsmith_api_key=langsmith_api_key or None,
                    langsmith_project=(
                        context.langsmith_project or "rolefit-resume-agent"
                    ),
                    trace_content=context.trace_content,
                )
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception(
                "Maximum-match pipeline failed (engine=%s, model=%s, error=%s)",
                context.engine,
                context.model_name,
                type(exc).__name__,
            )
            raise _pipeline_http_exception(exc) from exc

        maximum_resume = normalize_skill_category_markdown(
            normalize_experience_chronology(
                result.get("resume_markdown", "")
            )
        ).strip()
        publication_repair = repair_resume_for_publication(
            maximum_resume,
            candidate_profile=candidate_profile,
            jd_analysis=context.jd_analysis,
            match_strategy=match_strategy,
        )
        maximum_resume = publication_repair.markdown.strip()
        if not maximum_resume:
            raise HTTPException(
                status_code=502,
                detail=(
                    "The maximum-match agents produced no resume. "
                    "The authentic resume remains available."
                ),
            )
        maximum_scores = MatchScorecard.model_validate(
            build_scorecard(
                resume_markdown=maximum_resume,
                jd_analysis=context.jd_analysis,
                match_strategy=match_strategy,
                reviewer=parse_reviewer_decision(
                    result.get("review_feedback", "")
                ),
            ).model_dump()
        )
        if not maximum_scores.structure_valid:
            issues = "; ".join(maximum_scores.structure_issues[:4])
            raise HTTPException(
                status_code=502,
                detail=(
                    "The maximum-match model returned an incomplete document: "
                    f"{issues}. No maximum-match PDF was created. Your "
                    "authentic resume remains available."
                ),
            )

        maximum_filename = (
            f"{_filename_part(context.company_name)}_"
            f"{_filename_part(context.candidate_name)}_"
            "Maximum_Match_Resume.pdf"
        )
        _DOWNLOAD_NAMES.setdefault(session_id, {})[
            "maximum_match"
        ] = maximum_filename
        await asyncio.to_thread(
            pdf_renderer.render_pdf,
            maximum_resume,
            _DOC_FILES["maximum_match"].format(sid=session_id),
        )

        warning = ""
        if result.get("scores", {}).get("score_status") != "valid":
            warning = (
                "The claim-auditor score was unavailable or partial. "
                "Review the deterministic coverage and agent insights."
            )
        response = MaximumMatchResponse(
            resume_markdown=maximum_resume,
            resume_pdf_url=(
                f"/api/resume/download/{session_id}/maximum_match"
            ),
            resume_filename=maximum_filename,
            approved=bool(result.get("approved")),
            scores=maximum_scores,
            insights_markdown=build_maximum_match_insights(
                maximum_scores,
                match_strategy,
                result.get("review_feedback", ""),
            ),
            review_feedback=result.get("review_feedback", ""),
            revision_count=result.get("revision_count"),
            usage=result.get("usage", {}),
            engine=result["engine"],
            model_name=result["model_name"],
            langsmith_enabled=result.get("langsmith_enabled", False),
            langsmith_project=result.get("langsmith_project"),
            trace_content=result.get("trace_content", False),
            warnings=(
                ([warning] if warning else [])
                + list(publication_repair.repair_notes)
            ),
            evidence_count=len(evidence),
            resolved_gaps=resolved_gap_names(evidence),
            session_id=session_id,
        )
        context.response = response
        context.response_signature = signature
        return response


@router.get("/download/{session_id}/{doc}")
async def download_pdf(session_id: str, doc: str) -> FileResponse:
    # session ids are uuid4().hex — alnum check also guards path traversal
    if doc not in _DOC_FILES or not session_id.isalnum():
        raise HTTPException(status_code=404, detail="Unknown document")
    pdf_path = pdf_renderer.PDF_DIR / _DOC_FILES[doc].format(sid=session_id)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="No PDF for this session")
    download_name = _DOWNLOAD_NAMES.get(session_id, {}).get(
        doc, f"tailored_{doc}.pdf"
    )
    return FileResponse(
        pdf_path, media_type="application/pdf", filename=download_name
    )
