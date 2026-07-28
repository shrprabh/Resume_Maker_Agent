"""Regression tests for atomic Google ADK reviewer decisions."""

from types import SimpleNamespace
import unittest

from google.adk.agents import LlmAgent, LoopAgent
from google.adk.models import BaseLlm, LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.services.adk_runner import _unverified_review_note
from app.services.resume_scoring import parse_reviewer_decision
from resume_agent import config
from resume_agent.agent import quality_reviewer
from resume_agent.maximum_agent import maximum_match_reviewer
from resume_agent.tools import (
    submit_maximum_match_review,
    submit_quality_review,
)


def _tool_context():
    return SimpleNamespace(
        state={},
        actions=SimpleNamespace(
            skip_summarization=False,
            escalate=None,
        ),
    )


class _FunctionCallModel(BaseLlm):
    model: str = "fake-review-model"
    tool_name: str
    arguments: dict[str, object]
    calls: int = 0

    async def generate_content_async(self, llm_request, stream=False):
        self.calls += 1
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_function_call(
                        name=self.tool_name,
                        args=self.arguments,
                    )
                ],
            )
        )


def _forced_tool_config(tool_name: str) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode=types.FunctionCallingConfigMode.ANY,
                allowed_function_names=[tool_name],
            )
        )
    )


class AdkReviewSubmissionTests(unittest.TestCase):
    def test_unverified_adk_note_uses_deterministic_checks_without_false_scores(self):
        scorecard = SimpleNamespace(
            supported_ats_coverage=82,
            structure_valid=False,
            structure_issues=["Missing Education section."],
        )

        note = _unverified_review_note(scorecard)

        self.assertIn("supported ATS coverage 82%", note)
        self.assertIn("Missing Education section.", note)
        self.assertIn("without inventing a quality or fabrication score", note)
        self.assertNotIn("REVIEW UNAVAILABLE", note)
        self.assertNotIn("Fabrications: 0", note)

    def test_authentic_approval_is_persisted_before_loop_exit(self):
        context = _tool_context()

        result = submit_quality_review(
            score=config.QUALITY_THRESHOLD,
            ats_coverage=config.MIN_SUPPORTED_ATS_COVERAGE,
            fabrication_count=0,
            approved=True,
            feedback=[],
            tool_context=context,
        )

        feedback = context.state[config.STATE_REVIEW_FEEDBACK]
        decision = parse_reviewer_decision(feedback)
        self.assertEqual(result["status"], "approved")
        self.assertTrue(context.actions.skip_summarization)
        self.assertTrue(context.actions.escalate)
        self.assertIsNotNone(decision)
        self.assertTrue(decision.approved)
        self.assertEqual(decision.score, config.QUALITY_THRESHOLD)
        self.assertEqual(
            decision.ats_coverage,
            config.MIN_SUPPORTED_ATS_COVERAGE,
        )
        self.assertEqual(decision.fabrication_count, 0)

    def test_rejection_stays_in_loop_and_preserves_numbered_writer_feedback(self):
        context = _tool_context()

        result = submit_quality_review(
            score=81,
            ats_coverage=79,
            fabrication_count=1,
            approved=False,
            feedback=[
                "Remove the unsupported ownership claim.",
                "Add supported TypeScript evidence to Skills.",
            ],
            tool_context=context,
        )

        feedback = context.state[config.STATE_REVIEW_FEEDBACK]
        decision = parse_reviewer_decision(feedback)
        self.assertEqual(result["status"], "revision_required")
        self.assertTrue(context.actions.skip_summarization)
        self.assertFalse(context.actions.escalate)
        self.assertIn(
            "1. Remove the unsupported ownership claim.",
            feedback,
        )
        self.assertIn(
            "2. Add supported TypeScript evidence to Skills.",
            feedback,
        )
        self.assertIsNotNone(decision)
        self.assertFalse(decision.approved)
        self.assertGreaterEqual(len(decision.feedback), 3)

    def test_claimed_approval_cannot_bypass_server_side_gates(self):
        context = _tool_context()

        result = submit_quality_review(
            score=100,
            ats_coverage=config.MIN_SUPPORTED_ATS_COVERAGE - 1,
            fabrication_count=0,
            approved=True,
            feedback=[],
            tool_context=context,
        )

        feedback = context.state[config.STATE_REVIEW_FEEDBACK]
        decision = parse_reviewer_decision(feedback)
        self.assertEqual(result["status"], "revision_required")
        self.assertFalse(context.actions.escalate)
        self.assertIn("coverage is", feedback)
        self.assertIsNotNone(decision)
        self.assertFalse(decision.approved)

    def test_maximum_match_uses_its_stricter_gates_and_state_key(self):
        rejected_context = _tool_context()
        submit_maximum_match_review(
            score=config.MAXIMUM_MATCH_THRESHOLD,
            ats_coverage=94,
            fabrication_count=0,
            approved=True,
            feedback=[],
            tool_context=rejected_context,
        )
        self.assertFalse(rejected_context.actions.escalate)
        self.assertNotIn(
            config.STATE_REVIEW_FEEDBACK,
            rejected_context.state,
        )
        self.assertIn(
            config.STATE_MAXIMUM_MATCH_FEEDBACK,
            rejected_context.state,
        )

        approved_context = _tool_context()
        submit_maximum_match_review(
            score=config.MAXIMUM_MATCH_THRESHOLD,
            ats_coverage=95,
            fabrication_count=0,
            approved=True,
            feedback=[],
            tool_context=approved_context,
        )
        decision = parse_reviewer_decision(
            approved_context.state[config.STATE_MAXIMUM_MATCH_FEEDBACK]
        )
        self.assertTrue(approved_context.actions.escalate)
        self.assertIsNotNone(decision)
        self.assertTrue(decision.approved)

    def test_invalid_or_incomplete_decision_is_not_committed(self):
        cases = (
            {"score": 101, "ats_coverage": 90, "fabrication_count": 0},
            {"score": 90, "ats_coverage": -1, "fabrication_count": 0},
            {"score": 90, "ats_coverage": 90, "fabrication_count": -1},
        )
        for values in cases:
            with self.subTest(values=values):
                context = _tool_context()
                with self.assertRaises(ValueError):
                    submit_quality_review(
                        **values,
                        approved=False,
                        feedback=["Make one correction."],
                        tool_context=context,
                    )
                self.assertEqual(context.state, {})
                self.assertFalse(context.actions.skip_summarization)
                self.assertIsNone(context.actions.escalate)

        context = _tool_context()
        with self.assertRaises(ValueError):
            submit_quality_review(
                score=80,
                ats_coverage=80,
                fabrication_count=0,
                approved=False,
                feedback=[],
                tool_context=context,
            )
        self.assertEqual(context.state, {})

        context = _tool_context()
        with self.assertRaises(ValueError):
            submit_quality_review(
                score=95,
                ats_coverage=95,
                fabrication_count=0,
                approved=True,
                feedback=["This contradicts approval."],
                tool_context=context,
            )
        self.assertEqual(context.state, {})

    def test_reviewer_agents_force_exactly_the_typed_submission_tool(self):
        cases = (
            (
                quality_reviewer,
                "submit_quality_review",
            ),
            (
                maximum_match_reviewer,
                "submit_maximum_match_review",
            ),
        )
        for agent, tool_name in cases:
            with self.subTest(agent=agent.name):
                function_config = (
                    agent.generate_content_config.tool_config
                    .function_calling_config
                )
                self.assertEqual(
                    function_config.mode,
                    types.FunctionCallingConfigMode.ANY,
                )
                self.assertEqual(
                    function_config.allowed_function_names,
                    [tool_name],
                )
                self.assertIsNone(agent.output_key)
                self.assertEqual(len(agent.tools), 1)
                self.assertEqual(agent.tools[0].__name__, tool_name)


class AdkReviewLoopIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def _run_loop(self, arguments: dict[str, object], passes: int):
        model = _FunctionCallModel(
            tool_name="submit_quality_review",
            arguments=arguments,
        )
        reviewer = LlmAgent(
            name="typed_quality_reviewer",
            model=model,
            instruction="Submit the supplied review decision.",
            tools=[submit_quality_review],
            generate_content_config=_forced_tool_config(
                "submit_quality_review"
            ),
        )
        loop = LoopAgent(
            name="typed_review_loop",
            sub_agents=[reviewer],
            max_iterations=passes,
        )
        sessions = InMemorySessionService()
        await sessions.create_session(
            app_name="typed_review_test",
            user_id="test-user",
            session_id="test-session",
            state={config.STATE_REVIEW_FEEDBACK: ""},
        )
        runner = Runner(
            agent=loop,
            app_name="typed_review_test",
            session_service=sessions,
        )
        events = []
        async for event in runner.run_async(
            user_id="test-user",
            session_id="test-session",
            new_message=types.Content(
                role="user",
                parts=[types.Part(text="Review now.")],
            ),
        ):
            events.append(event)
        session = await sessions.get_session(
            app_name="typed_review_test",
            user_id="test-user",
            session_id="test-session",
        )
        return model, events, session

    async def test_approved_submission_commits_state_and_exits_in_one_call(self):
        model, events, session = await self._run_loop(
            {
                "score": 90,
                "ats_coverage": 90,
                "fabrication_count": 0,
                "approved": True,
                "feedback": [],
            },
            passes=3,
        )

        self.assertEqual(model.calls, 1)
        self.assertTrue(any(event.actions.escalate for event in events))
        self.assertTrue(
            any(event.actions.skip_summarization for event in events)
        )
        decision = parse_reviewer_decision(
            session.state[config.STATE_REVIEW_FEEDBACK]
        )
        self.assertIsNotNone(decision)
        self.assertTrue(decision.approved)

    async def test_rejected_submission_commits_feedback_and_keeps_looping(self):
        model, events, session = await self._run_loop(
            {
                "score": 80,
                "ats_coverage": 80,
                "fabrication_count": 0,
                "approved": False,
                "feedback": ["Strengthen the supported lead-role bullet."],
            },
            passes=2,
        )

        self.assertEqual(model.calls, 2)
        self.assertFalse(any(event.actions.escalate for event in events))
        self.assertEqual(
            sum(
                config.STATE_REVIEW_FEEDBACK
                in event.actions.state_delta
                for event in events
            ),
            2,
        )
        self.assertIn(
            "1. Strengthen the supported lead-role bullet.",
            session.state[config.STATE_REVIEW_FEEDBACK],
        )


if __name__ == "__main__":
    unittest.main()
