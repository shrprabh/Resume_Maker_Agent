import time
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.routers import resume as resume_router


VALID_RESUME = """\
# Candidate Name

candidate@example.com | Chicago, IL

## Summary
Software engineer with experience building reliable web applications and
service integrations for operational teams. Delivers maintainable features
through thoughtful API design, practical testing, clear documentation, and
close collaboration with product partners. Known for translating ambiguous
requirements into secure, measurable improvements while preserving system
stability and a straightforward user experience.

## Skills
**Languages:** Python, TypeScript
**Frameworks & Libraries:** FastAPI, React

## Experience
### Software Engineer — Example Company | January 2022 – Present
- Built and maintained internal web applications that helped operations teams
  review customer records, resolve workflow exceptions, document decisions,
  and complete recurring service tasks with consistent validation and clear
  audit history across each production release.
- Designed REST API endpoints with explicit request validation, predictable
  error responses, focused automated tests, and practical operational
  documentation, helping engineers integrate new features while reducing
  avoidable regressions during routine application maintenance.
- Partnered with product managers and users to clarify requirements, break
  work into testable increments, review implementation tradeoffs, and deliver
  accessible interfaces that supported real workflows without adding
  unnecessary complexity to the broader platform.
- Investigated production issues by reviewing logs, reproducing failure paths,
  checking data assumptions, and coordinating verified fixes, then documented
  lessons and monitoring improvements so similar incidents could be diagnosed
  more quickly by the engineering team.

## Education
Bachelor of Science in Computer Science — Example University, 2021

Additional coursework included software engineering, database systems,
distributed applications, information security, and human-computer
interaction with collaborative laboratory projects.
"""

PIPELINE_RESULT = {
    "session_id": "sourcebundle123",
    "approved": True,
    "resume_markdown": VALID_RESUME,
    "cover_letter_markdown": "",
    "cover_letter_error": "",
    "jd_analysis": "## Target Company\nExample Company\n\n## ATS Keywords (verbatim)\n",
    "candidate_profile": "## Contact\nCandidate Name",
    "match_strategy": "## Keyword Placement Plan\n",
    "review_feedback": (
        '{"score":95,"ats_coverage":100,"fabrication_count":0,'
        '"approved":true,"feedback":[]}'
    ),
    "review_score": 95,
    "ats_coverage": 100,
    "review_valid": True,
    "revision_count": 1,
    "usage": {},
    "engine": "google_adk",
    "model_name": "gemini-test",
    "langsmith_enabled": False,
    "langsmith_project": None,
    "trace_content": False,
}


class SourcePreflightTests(unittest.TestCase):
    def setUp(self):
        resume_router._SOURCE_BUNDLES.clear()

    def tearDown(self):
        resume_router._SOURCE_BUNDLES.clear()

    @patch(
        "app.routers.resume.adk_runner.run_pipeline",
        new_callable=AsyncMock,
    )
    def test_preflight_extracts_text_without_calling_model(self, run_pipeline):
        with TestClient(app) as client:
            response = client.post(
                "/api/resume/sources/preflight",
                files={
                    "files": (
                        "candidate.txt",
                        b"Candidate Name\r\n\r\nEducation\r\nExample University",
                        "text/plain",
                    )
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertTrue(payload["ready"])
        self.assertEqual(len(payload["source_bundle_id"]), 32)
        self.assertEqual(payload["sources"][0]["status"], "ok")
        self.assertEqual(
            payload["sources"][0]["text"],
            "Candidate Name\n\nEducation\nExample University",
        )
        self.assertIn("Education", payload["sources"][0]["detected_sections"])
        self.assertEqual(
            payload["sources"][0]["included_chars"],
            len(payload["sources"][0]["text"]),
        )
        run_pipeline.assert_not_awaited()

    def test_combined_budget_keeps_each_source_head_and_education_tail(self):
        first = (
            "Candidate One\n"
            + ("Built supported application evidence.\n" * 18)
            + "Education\nExample University One"
        )
        second = (
            "Candidate Two\n"
            + ("Documented supported project evidence.\n" * 18)
            + "Education\nExample University Two"
        )
        with patch.object(resume_router, "MAX_TOTAL_CHARS", 360):
            with TestClient(app) as client:
                response = client.post(
                    "/api/resume/sources/preflight",
                    files=[
                        (
                            "files",
                            ("one.txt", first.encode(), "text/plain"),
                        ),
                        (
                            "files",
                            ("two.txt", second.encode(), "text/plain"),
                        ),
                    ],
                )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertTrue(payload["ready"])
        self.assertLessEqual(payload["total_chars"], 360)
        self.assertTrue(all(item["truncated"] for item in payload["sources"]))
        self.assertTrue(
            all("middle source text omitted" in item["text"] for item in payload["sources"])
        )
        self.assertIn("Example University One", payload["sources"][0]["text"])
        self.assertIn("Example University Two", payload["sources"][1]["text"])

    def test_oversized_source_is_visible_and_never_bundled(self):
        with patch.object(resume_router, "MAX_SOURCE_BYTES", 16):
            with TestClient(app) as client:
                response = client.post(
                    "/api/resume/sources/preflight",
                    files={
                        "files": (
                            "oversized.txt",
                            b"candidate evidence exceeds limit",
                            "text/plain",
                        )
                    },
                )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertFalse(payload["ready"])
        self.assertIsNone(payload["source_bundle_id"])
        self.assertEqual(payload["sources"][0]["status"], "failed")
        self.assertIn("upload limit", payload["sources"][0]["warnings"][0])

    @patch("app.routers.resume.pdf_renderer.render_pdf")
    @patch(
        "app.routers.resume.adk_runner.run_pipeline",
        new_callable=AsyncMock,
    )
    def test_generate_reuses_exact_preflight_bundle(
        self, run_pipeline, render_pdf
    ):
        run_pipeline.return_value = dict(PIPELINE_RESULT)
        source = (
            "Candidate Name\n\nExperience\nBuilt APIs.\n\n"
            "Education\nExample University"
        )
        with TestClient(app) as client:
            preflight = client.post(
                "/api/resume/sources/preflight",
                files={
                    "files": (
                        "candidate.txt",
                        source.encode(),
                        "text/plain",
                    )
                },
            )
            self.assertEqual(preflight.status_code, 200, preflight.text)
            preview = preflight.json()
            generated = client.post(
                "/api/resume/generate",
                data={
                    "job_description": "Build reliable web applications.",
                    "source_bundle_id": preview["source_bundle_id"],
                },
            )

        self.assertEqual(generated.status_code, 200, generated.text)
        passed_context = run_pipeline.await_args.kwargs["candidate_text"]
        self.assertEqual(passed_context, preview["sources"][0]["text"])
        self.assertEqual(
            generated.json()["source_manifest"],
            preview["sources"],
        )
        render_pdf.assert_called_once()

    @patch(
        "app.routers.resume.adk_runner.run_pipeline",
        new_callable=AsyncMock,
    )
    def test_expired_bundle_returns_410_before_model_call(self, run_pipeline):
        with TestClient(app) as client:
            preflight = client.post(
                "/api/resume/sources/preflight",
                data={"resume_text": "Candidate facts and project evidence."},
            )
            self.assertEqual(preflight.status_code, 200, preflight.text)
            bundle_id = preflight.json()["source_bundle_id"]
            resume_router._SOURCE_BUNDLES[bundle_id].created_at = (
                time.monotonic()
                - resume_router._SOURCE_BUNDLE_TTL_SECONDS
                - 1
            )
            generated = client.post(
                "/api/resume/generate",
                data={
                    "job_description": "Build reliable web applications.",
                    "source_bundle_id": bundle_id,
                },
            )

        self.assertEqual(generated.status_code, 410, generated.text)
        self.assertIn("source preview expired", generated.json()["detail"])
        run_pipeline.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
