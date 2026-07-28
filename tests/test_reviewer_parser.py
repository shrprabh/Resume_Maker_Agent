"""Focused regression tests for portable reviewer-output parsing."""

import unittest

from app.services.resume_scoring import parse_reviewer_decision


class ReviewerParserTests(unittest.TestCase):
    def test_accepts_mapping_input_and_safe_aliases(self):
        decision = parse_reviewer_decision(
            {
                "qualityScore": "92/100",
                "supported_ats_coverage": "97%",
                "fabrications": "0",
                "verdict": "approved",
                "issues": [],
            }
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.score, 92)
        self.assertEqual(decision.ats_coverage, 97)
        self.assertEqual(decision.fabrication_count, 0)
        self.assertTrue(decision.approved)
        self.assertEqual(decision.feedback, [])

    def test_accepts_fenced_json_with_surrounding_commentary(self):
        decision = parse_reviewer_decision(
            """Audit complete.
```json
{
  "review_score": "88%",
  "atsCoverage": "95/100",
  "unsupported_claim_count": 1,
  "is_approved": false,
  "corrections": ["Remove the unsupported metric."]
}
```
"""
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.score, 88)
        self.assertEqual(decision.ats_coverage, 95)
        self.assertEqual(decision.fabrication_count, 1)
        self.assertFalse(decision.approved)
        self.assertEqual(
            decision.feedback,
            ["Remove the unsupported metric."],
        )

    def test_balanced_object_scanner_ignores_unrelated_braces(self):
        decision = parse_reviewer_decision(
            """Preliminary note: {this is not JSON}. An unmatched " is prose.
Final result:
{"score":91,"ats_coverage":98,"fabrication_count":0,
 "approved":false,"feedback":["Tighten the summary {without rewriting it}."]}
"""
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.score, 91)
        self.assertEqual(
            decision.feedback,
            ["Tighten the summary {without rewriting it}."],
        )

    def test_rejects_conflicting_complete_json_objects(self):
        self.assertIsNone(
            parse_reviewer_decision(
                """
{"score":0,"ats_coverage":0,"fabrication_count":0,
 "approved":false,"feedback":[]}
{"score":94,"ats_coverage":100,"fabrication_count":0,
 "approved":true,"feedback":[]}
"""
            )
        )

    def test_accepts_canonical_nonapproval_prose_and_feedback(self):
        decision = parse_reviewer_decision(
            """SCORE: 84/100 | ATS coverage: 96% | Fabrications: 1
1. Remove the unsupported production-volume claim.
2) Add the supported PostgreSQL keyword to Skills.
"""
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.score, 84)
        self.assertEqual(decision.ats_coverage, 96)
        self.assertEqual(decision.fabrication_count, 1)
        self.assertFalse(decision.approved)
        self.assertEqual(
            decision.feedback,
            [
                "Remove the unsupported production-volume claim.",
                "Add the supported PostgreSQL keyword to Skills.",
            ],
        )

    def test_accepts_canonical_approved_prose_only_with_explicit_count(self):
        decision = parse_reviewer_decision(
            "APPROVED — score 93/100, ATS coverage 100%, Fabrications: 0."
        )

        self.assertIsNotNone(decision)
        self.assertTrue(decision.approved)
        self.assertEqual(decision.fabrication_count, 0)

    def test_accepts_markdown_labels_and_table_rows(self):
        markdown = parse_reviewer_decision(
            """**Score:** 90%
**ATS Coverage:** 98/100
**Fabrications:** 0
**Approved:** false
1. Strengthen the lead bullet.
"""
        )
        table = parse_reviewer_decision(
            """| Score | 91/100 |
| ATS Coverage | 99% |
| Fabrications | 0 |
| Verdict | rejected |
"""
        )

        self.assertIsNotNone(markdown)
        self.assertEqual(markdown.score, 90)
        self.assertFalse(markdown.approved)
        self.assertIsNotNone(table)
        self.assertEqual(table.ats_coverage, 99)
        self.assertFalse(table.approved)

    def test_missing_fabrication_count_is_never_assumed_zero(self):
        self.assertIsNone(
            parse_reviewer_decision(
                "APPROVED — score 95/100, ATS coverage 100%."
            )
        )
        self.assertIsNone(
            parse_reviewer_decision(
                {
                    "score": 95,
                    "ats_coverage": 100,
                    "approved": True,
                    "feedback": [],
                }
            )
        )

    def test_missing_approval_is_not_inferred_from_high_score(self):
        self.assertIsNone(
            parse_reviewer_decision(
                {
                    "score": 99,
                    "ats_coverage": 100,
                    "fabrication_count": 0,
                    "feedback": [],
                }
            )
        )
        self.assertIsNone(
            parse_reviewer_decision(
                """Quality score: 99%
ATS coverage: 100%
Fabrications: 0
"""
            )
        )

    def test_rejects_fractional_out_of_range_and_boolean_numbers(self):
        invalid_scores = (0.92, 92.5, -1, 101, True, "101%", "92.5%")
        for score in invalid_scores:
            with self.subTest(score=score):
                self.assertIsNone(
                    parse_reviewer_decision(
                        {
                            "score": score,
                            "ats_coverage": 100,
                            "fabrication_count": 0,
                            "approved": False,
                        }
                    )
                )

    def test_rejects_conflicting_alias_values(self):
        self.assertIsNone(
            parse_reviewer_decision(
                {
                    "score": 90,
                    "review_score": 91,
                    "ats_coverage": 100,
                    "fabrication_count": 0,
                    "approved": False,
                }
            )
        )

    def test_rejects_contradictory_approval_and_fabrications(self):
        self.assertIsNone(
            parse_reviewer_decision(
                {
                    "score": 95,
                    "ats_coverage": 100,
                    "fabrication_count": 2,
                    "approved": True,
                }
            )
        )

    def test_does_not_parse_thresholds_from_arbitrary_prose(self):
        self.assertIsNone(
            parse_reviewer_decision(
                "Approve only when score: 85 and ATS coverage: 95%; "
                "fabrications: 0."
            )
        )


if __name__ == "__main__":
    unittest.main()
