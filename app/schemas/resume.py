"""Response models for the resume API.

(The generate endpoint takes multipart form data — file + form fields — so
its request side is declared with File/Form parameters, not a Pydantic model.)
"""

from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


class PipelineArtifacts(BaseModel):
    """Intermediate outputs of the pipeline — exposed for transparency/demos."""

    jd_analysis: str
    candidate_profile: str
    match_strategy: str
    review_feedback: str


class MatchScorecard(BaseModel):
    """Transparent scores with different meanings kept separate."""

    supported_ats_coverage: int | None = None
    overall_requirement_match: int | None = None
    evidence_integrity: int | None = None
    quality_score: int | None = None
    score_status: str
    structure_valid: bool = False
    structure_issues: list[str] = Field(default_factory=list)
    word_count: int = 0
    claimable_keywords: list[str] = Field(default_factory=list)
    placed_keywords: list[str] = Field(default_factory=list)
    missing_supported_keywords: list[str] = Field(default_factory=list)
    unsupported_keywords: list[str] = Field(default_factory=list)


class ResumeGenerateResponse(BaseModel):
    resume_markdown: str
    cover_letter_markdown: str
    artifacts: PipelineArtifacts
    approved: bool  # True if the reviewer called exit_loop before max_iterations
    resume_pdf_url: str
    cover_letter_pdf_url: str
    resume_filename: str
    cover_letter_filename: str
    candidate_name: str
    company_name: str
    engine: str = "google_adk"
    model_name: str
    review_score: int | None = None
    ats_coverage: int | None = None
    review_valid: bool = False
    scores: MatchScorecard | None = None
    revision_count: int | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    langsmith_enabled: bool = False
    langsmith_project: str | None = None
    trace_content: bool = False
    warnings: list[str] = Field(default_factory=list)
    maximum_match_generate_url: str
    session_id: str


class MaximumMatchResponse(BaseModel):
    resume_markdown: str
    resume_pdf_url: str
    resume_filename: str
    approved: bool
    scores: MatchScorecard
    insights_markdown: str
    review_feedback: str
    revision_count: int | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    engine: str
    model_name: str
    langsmith_enabled: bool = False
    langsmith_project: str | None = None
    trace_content: bool = False
    warnings: list[str] = Field(default_factory=list)
    evidence_count: int = 0
    resolved_gaps: list[str] = Field(default_factory=list)
    session_id: str


class GapEvidenceSource(str, Enum):
    WORK_EXPERIENCE = "work_experience"
    PRODUCT_PROJECT = "product_project"


class MaximumMatchGap(BaseModel):
    id: str
    skill: str
    reason: str
    origin: str
    ats_keyword: bool = False


class MaximumMatchGapsResponse(BaseModel):
    session_id: str
    gaps: list[MaximumMatchGap] = Field(default_factory=list)
    expires_in_minutes: int = 60


class GapEvidenceSubmission(BaseModel):
    gap_id: str = Field(min_length=8, max_length=32)
    source_type: GapEvidenceSource
    source_name: str = Field(min_length=2, max_length=160)
    role_or_contribution: str = Field(min_length=2, max_length=180)
    dates: str = Field(min_length=4, max_length=100)
    evidence_text: str = Field(min_length=40, max_length=2_000)
    outcome: str = Field(default="", max_length=500)
    reference_url: str = Field(default="", max_length=500)
    candidate_attested: bool = False

    @field_validator(
        "gap_id",
        "source_name",
        "role_or_contribution",
        "dates",
        "evidence_text",
        "outcome",
        "reference_url",
        mode="before",
    )
    @classmethod
    def normalize_text(cls, value: object) -> str:
        if value is None:
            return ""
        return " ".join(str(value).replace("\x00", " ").split())

    @model_validator(mode="after")
    def require_attestation(self) -> "GapEvidenceSubmission":
        if not self.candidate_attested:
            raise ValueError(
                "Confirm that the submitted gap evidence is accurate."
            )
        if self.reference_url and not self.reference_url.startswith(
            ("https://", "http://")
        ):
            raise ValueError("Reference links must begin with http:// or https://.")
        return self


class MaximumMatchEvidenceRequest(BaseModel):
    evidence: list[GapEvidenceSubmission] = Field(
        default_factory=list,
        max_length=20,
    )


class GapEvidenceValidation(BaseModel):
    gap_id: str
    skill: str
    source_type: GapEvidenceSource
    source_label: str
    status: str = "accepted"


class MaximumMatchEvidenceValidationResponse(BaseModel):
    session_id: str
    accepted: list[GapEvidenceValidation] = Field(default_factory=list)
    unresolved_gap_count: int = 0
    message: str


class HealthResponse(BaseModel):
    status: str = "ok"
