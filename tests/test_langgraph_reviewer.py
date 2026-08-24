import unittest
from types import SimpleNamespace

from langchain_core.messages import AIMessage

from app.services.langgraph_runner import (
    _QUALITY_REVIEWER_JSON_INSTRUCTION,
    _deterministic_review_note,
    _invoke_json_reviewer,
    _is_schema_unsupported_error,
    _maximum_review_approved,
    _reasoning_config,
    _review_response_format,
)
from app.services.resume_scoring import ReviewerDecision


VALID_DECISION = (
    '{"score":91,"ats_coverage":96,"fabrication_count":0,'
    '"approved":true,"feedback":[]}'
)


class FakeProviderError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.raw_response = SimpleNamespace(
            status_code=status_code,
            text=message,
        )


class FakeBoundModel:
    def __init__(self, parent, bind_kwargs):
        self.parent = parent
        self.bind_kwargs = bind_kwargs

    async def ainvoke(self, messages, config):
        self.parent.invocations.append(
            {
                "bind": self.bind_kwargs,
                "messages": messages,
                "config": config,
            }
        )
        result = self.parent.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeModel:
    def __init__(self, *results):
        self.results = list(results)
        self.bind_calls = []
        self.invocations = []

    def bind(self, **kwargs):
        self.bind_calls.append(kwargs)
        return FakeBoundModel(self, kwargs)


def message(content: str) -> AIMessage:
    return AIMessage(
        content=content,
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 25,
            "total_tokens": 125,
        },
        response_metadata={"finish_reason": "stop"},
    )


class ReviewerContractTests(unittest.IsolatedAsyncioTestCase):
    def test_all_writer_calls_keep_mandatory_reasoning_enabled(self):
        for run_name in (
            "resume_writer",
            "cover_letter_writer",
            "maximum_match_writer",
        ):
            with self.subTest(run_name=run_name):
                config = _reasoning_config(run_name)
                self.assertEqual(config["effort"], "minimal")
                self.assertTrue(config["exclude"])

    def test_schema_requires_all_fields_and_rejects_extras(self):
        response_format = _review_response_format()
        self.assertEqual(response_format["type"], "json_schema")
        envelope = response_format["json_schema"]
        self.assertTrue(envelope["strict"])
        schema = envelope["schema"]
        self.assertEqual(
            schema["required"],
            [
                "score",
                "ats_coverage",
                "fabrication_count",
                "approved",
                "feedback",
            ],
        )
        self.assertFalse(schema["additionalProperties"])

    async def test_strict_schema_success_uses_one_completion(self):
        model = FakeModel(message(VALID_DECISION))
        decision, usage = await _invoke_json_reviewer(
            model=model,
            state={
                "draft_resume": "# Candidate",
                "candidate_profile": "Verified facts",
                "jd_analysis": "ATS terms",
                "match_strategy": "Placement plan",
            },
            instruction=_QUALITY_REVIEWER_JSON_INSTRUCTION,
            run_name="quality_reviewer",
            model_name="example/model",
            tags=["review"],
        )

        self.assertIsNotNone(decision)
        self.assertTrue(decision.approved)
        self.assertEqual(usage["total_tokens"], 125)
        self.assertEqual(len(model.invocations), 1)
        strict_call = model.bind_calls[0]
        self.assertIn("response_format", strict_call)
        self.assertEqual(
            strict_call["response_format"]["json_schema"]["schema"][
                "additionalProperties"
            ],
            False,
        )
        rendered_prompt = model.invocations[0]["messages"][0].content
        self.assertNotIn("exit_loop", rendered_prompt)
        self.assertIn("at least 85", rendered_prompt)

    async def test_explicit_schema_capability_error_falls_back_once(self):
        unsupported = FakeProviderError(
            400,
            "No endpoints support response_format json_schema; "
            "structured output is unsupported",
        )
        model = FakeModel(unsupported, message(VALID_DECISION))

        decision, _ = await _invoke_json_reviewer(
            model=model,
            state={},
            instruction=_QUALITY_REVIEWER_JSON_INSTRUCTION,
            run_name="quality_reviewer",
            model_name="example/model",
            tags=["review"],
        )

        self.assertIsNotNone(decision)
        self.assertEqual(len(model.invocations), 2)
        self.assertIn("response_format", model.bind_calls[0])
        self.assertNotIn("response_format", model.bind_calls[1])
        fallback_metadata = model.invocations[1]["config"]["metadata"]
        self.assertTrue(fallback_metadata["schema_capability_fallback"])

    async def test_malformed_completed_response_is_not_retried(self):
        model = FakeModel(message("I approve this resume."))

        decision, usage = await _invoke_json_reviewer(
            model=model,
            state={},
            instruction=_QUALITY_REVIEWER_JSON_INSTRUCTION,
            run_name="quality_reviewer",
            model_name="example/model",
            tags=["review"],
        )

        self.assertIsNone(decision)
        self.assertEqual(usage["total_tokens"], 125)
        self.assertEqual(len(model.invocations), 1)
        self.assertEqual(len(model.results), 0)

    async def test_contradictory_approval_is_safely_rejected(self):
        model = FakeModel(
            message(
                '{"score":96,"ats_coverage":100,"fabrication_count":0,'
                '"approved":true,"feedback":["Narrow one claim."]}'
            )
        )

        decision, _ = await _invoke_json_reviewer(
            model=model,
            state={},
            instruction=_QUALITY_REVIEWER_JSON_INSTRUCTION,
            run_name="quality_reviewer",
            model_name="example/model",
            tags=["review"],
        )

        self.assertIsNotNone(decision)
        self.assertFalse(decision.approved)
        self.assertEqual(decision.feedback, ["Narrow one claim."])
        self.assertEqual(len(model.invocations), 1)

    async def test_timeout_and_unrelated_400_are_not_retried(self):
        for error in (
            TimeoutError("provider timed out"),
            FakeProviderError(400, "maximum context length exceeded"),
            FakeProviderError(429, "response_format temporarily rate limited"),
        ):
            with self.subTest(error=type(error).__name__, detail=str(error)):
                model = FakeModel(error)
                with self.assertRaises(type(error)):
                    await _invoke_json_reviewer(
                        model=model,
                        state={},
                        instruction=_QUALITY_REVIEWER_JSON_INSTRUCTION,
                        run_name="quality_reviewer",
                        model_name="example/model",
                        tags=["review"],
                    )
                self.assertEqual(len(model.invocations), 1)

    def test_schema_fallback_detection_is_narrow(self):
        self.assertTrue(
            _is_schema_unsupported_error(
                FakeProviderError(422, "json_schema is not supported")
            )
        )
        self.assertFalse(
            _is_schema_unsupported_error(
                FakeProviderError(400, "maximum context length exceeded")
            )
        )
        self.assertFalse(
            _is_schema_unsupported_error(
                FakeProviderError(429, "response_format unsupported")
            )
        )

    def test_unverified_note_reports_only_deterministic_checks(self):
        scorecard = SimpleNamespace(
            supported_ats_coverage=84,
            structure_valid=False,
            structure_issues=["Missing Education section."],
        )
        note = _deterministic_review_note(scorecard)
        self.assertIn("supported ATS coverage 84%", note)
        self.assertIn("Missing Education section.", note)
        self.assertIn("without inventing a quality or fabrication score", note)
        self.assertNotIn("Fabrications: 0", note)
        self.assertNotIn("REVIEW UNAVAILABLE", note)

    def test_maximum_match_requires_explicit_reviewer_approval(self):
        scorecard = SimpleNamespace(
            quality_score=96,
            structure_valid=True,
            supported_ats_coverage=100,
        )
        rejected = ReviewerDecision(
            score=96,
            ats_coverage=100,
            fabrication_count=0,
            approved=False,
            feedback=["Narrow one claim."],
        )
        approved = rejected.model_copy(
            update={"approved": True, "feedback": []}
        )

        self.assertFalse(_maximum_review_approved(rejected, scorecard))
        self.assertTrue(_maximum_review_approved(approved, scorecard))


if __name__ == "__main__":
    unittest.main()
