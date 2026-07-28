"""FastAPI entrypoint.

Run from the project root:

    uvicorn app.main:app --reload --port 8080

(8080 on purpose — `adk web` uses 8000, and both can run at the same time.)
"""

from pathlib import Path

from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Load both supported locations before importing routes/model clients. Existing
# process environment values retain precedence.
load_dotenv(PROJECT_DIR / ".env")
load_dotenv(PROJECT_DIR / "resume_agent" / ".env")

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from .routers import providers, resume  # noqa: E402  (import after env is loaded)

app = FastAPI(
    title="Resume Maker Agent",
    description=(
        "Dual-engine agent pipeline for authentic and maximum-verified-match "
        "resumes plus cover letters."
    ),
    version="0.2.0",
)
app.include_router(resume.router)
app.include_router(providers.router)
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")


@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    """Serve the integrated resume-builder frontend."""
    return FileResponse(APP_DIR / "static" / "index.html")
