"""Provider discovery and credential-validation endpoints."""

import asyncio
import os

import httpx
from fastapi import APIRouter, Header, HTTPException, Query
from langsmith import Client
from pydantic import BaseModel, Field

from ..services.credentials import normalize_api_key, normalize_env_text

router = APIRouter(prefix="/api/providers", tags=["providers"])


class ProviderSummary(BaseModel):
    id: str
    name: str
    configured: bool
    description: str
    tracing_configured: bool = False
    langsmith_project: str | None = None


class OpenRouterModel(BaseModel):
    id: str
    name: str
    context_length: int | None = None
    prompt_price: str | None = None
    completion_price: str | None = None
    supported_parameters: list[str] = Field(default_factory=list)


class ValidationResponse(BaseModel):
    valid: bool = True
    message: str


@router.get("", response_model=list[ProviderSummary])
async def providers() -> list[ProviderSummary]:
    openrouter_configured = bool(_server_openrouter_key())
    langsmith_configured = bool(_server_langsmith_key())
    return [
        ProviderSummary(
            id="google_adk",
            name="Google ADK",
            configured=True,
            description="Existing Gemini-powered Google ADK workflow.",
        ),
        ProviderSummary(
            id="langgraph_openrouter",
            name="LangGraph + OpenRouter",
            configured=openrouter_configured,
            description="Six-agent LangGraph workflow using your OpenRouter model.",
            tracing_configured=langsmith_configured,
            langsmith_project=normalize_env_text(
                os.getenv("LANGSMITH_PROJECT")
            )
            or "rolefit-resume-agent",
        ),
    ]


def _server_openrouter_key() -> str:
    value = os.getenv("OPENROUTER") or os.getenv("OPENROUTER_API_KEY")
    return normalize_api_key(value, "OPENROUTER_API_KEY")


def _server_langsmith_key() -> str:
    return normalize_api_key(
        os.getenv("LANGSMITH_API_KEY"), "LANGSMITH_API_KEY"
    )


def _required_key(header_value: str | None, provider: str) -> str:
    if provider == "openrouter":
        key = normalize_api_key(header_value, "OPENROUTER_API_KEY")
        key = key or _server_openrouter_key()
        detail = (
            "OpenRouter API key is required. Add OPENROUTER= to .env or "
            "provide a browser key."
        )
    else:
        key = normalize_api_key(header_value, "LANGSMITH_API_KEY")
        key = key or _server_langsmith_key()
        detail = (
            "LangSmith API key is required. Add LANGSMITH_API_KEY= to .env "
            "or provide a browser key."
        )
    if not key:
        raise HTTPException(status_code=422, detail=detail)
    return key


async def _openrouter_models(api_key: str) -> list[dict]:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
    if response.status_code == 401:
        raise HTTPException(status_code=401, detail="Invalid OpenRouter API key")
    if response.status_code == 429:
        raise HTTPException(
            status_code=503, detail="OpenRouter rate limit reached"
        )
    response.raise_for_status()
    return response.json().get("data", [])


async def _validate_openrouter_key(api_key: str) -> dict:
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(
            "https://openrouter.ai/api/v1/key",
            headers={"Authorization": f"Bearer {api_key}"},
        )
    if response.status_code == 401:
        raise HTTPException(status_code=401, detail="Invalid OpenRouter API key")
    if response.status_code == 429:
        raise HTTPException(
            status_code=503, detail="OpenRouter rate limit reached"
        )
    response.raise_for_status()
    return response.json().get("data", {})


def _openrouter_error(response: httpx.Response) -> str:
    try:
        payload = response.json()
        error = payload.get("error", {})
        if isinstance(error, dict):
            return str(error.get("message") or error.get("code") or "").strip()
        return str(error).strip()
    except Exception:
        return response.text.strip()


async def _validate_openrouter_model(api_key: str, model_name: str) -> None:
    try:
        async with httpx.AsyncClient(timeout=75) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "X-OpenRouter-Title": "RoleFit Resume Agent",
                },
                json={
                    "model": model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Reply with exactly OK.",
                        }
                    ],
                    "max_tokens": 4,
                    "temperature": 0,
                },
            )
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail=(
                f"OpenRouter model '{model_name}' timed out during its test. "
                "Free models can be congested; retry or choose another model."
            ),
        ) from exc

    if response.is_success:
        return
    message = _openrouter_error(response) or "OpenRouter model test failed"
    if response.status_code == 401:
        raise HTTPException(status_code=401, detail="Invalid OpenRouter API key")
    if response.status_code == 402:
        raise HTTPException(
            status_code=402, detail="OpenRouter account has insufficient credits"
        )
    if response.status_code in {400, 404}:
        raise HTTPException(status_code=422, detail=message)
    if response.status_code == 429:
        raise HTTPException(
            status_code=503,
            detail=f"OpenRouter rate limit reached: {message}",
        )
    raise HTTPException(
        status_code=502, detail=f"OpenRouter provider error: {message}"
    )


@router.get("/openrouter/models", response_model=list[OpenRouterModel])
async def openrouter_models(
    x_openrouter_api_key: str | None = Header(
        None, alias="X-OpenRouter-Api-Key"
    ),
    q: str = Query("", max_length=100),
    limit: int = Query(60, ge=1, le=200),
) -> list[OpenRouterModel]:
    api_key = _required_key(x_openrouter_api_key, "openrouter")
    await _validate_openrouter_key(api_key)
    models = await _openrouter_models(api_key)
    query = q.strip().lower()
    if query:
        models = [
            model
            for model in models
            if query in str(model.get("id", "")).lower()
            or query in str(model.get("name", "")).lower()
        ]
    models.sort(key=lambda model: str(model.get("name", "")).lower())
    return [
        OpenRouterModel(
            id=model.get("id", ""),
            name=model.get("name") or model.get("id", ""),
            context_length=model.get("context_length"),
            prompt_price=(model.get("pricing") or {}).get("prompt"),
            completion_price=(model.get("pricing") or {}).get("completion"),
            supported_parameters=model.get("supported_parameters") or [],
        )
        for model in models[:limit]
        if model.get("id")
    ]


@router.get("/openrouter/validate", response_model=ValidationResponse)
async def validate_openrouter(
    x_openrouter_api_key: str | None = Header(
        None, alias="X-OpenRouter-Api-Key"
    ),
    model_name: str | None = Query(None, max_length=200),
) -> ValidationResponse:
    api_key = _required_key(x_openrouter_api_key, "openrouter")
    if not api_key.startswith("sk-or-"):
        raise HTTPException(
            status_code=422,
            detail=(
                "This does not look like an OpenRouter key. Copy an inference "
                "key from openrouter.ai/keys; it normally starts with 'sk-or-'."
            ),
        )
    key_info = await _validate_openrouter_key(api_key)
    selected_model = (model_name or "").strip()
    if selected_model:
        await _validate_openrouter_model(api_key, selected_model)
    remaining = key_info.get("limit_remaining")
    suffix = (
        f" — ${remaining:.2f} key limit remaining"
        if isinstance(remaining, (int, float))
        else ""
    )
    return ValidationResponse(
        message=(
            f"OpenRouter key and model connected{suffix}"
            if selected_model
            else f"OpenRouter connected{suffix}"
        )
    )


@router.get("/langsmith/validate", response_model=ValidationResponse)
async def validate_langsmith(
    x_langsmith_api_key: str | None = Header(
        None, alias="X-LangSmith-Api-Key"
    ),
) -> ValidationResponse:
    api_key = _required_key(x_langsmith_api_key, "langsmith")

    def check() -> None:
        client = Client(api_key=api_key)
        try:
            next(client.list_projects(limit=1), None)
        finally:
            client.close(timeout=2)

    try:
        await asyncio.to_thread(check)
    except Exception as exc:
        raise HTTPException(
            status_code=401, detail="Invalid LangSmith API key or workspace access"
        ) from exc
    return ValidationResponse(message="LangSmith connected")
