from __future__ import annotations

import re
import unittest

from app.services.resume_repair import repair_resume_for_publication
from app.services.resume_scoring import audit_resume_structure


def _long_resume() -> str:
    summary = (
        "Backend and full-stack engineer delivering secure payment services, "
        "maintainable APIs, reliable data workflows, and production-ready user "
        "experiences. Brings practical testing, release, documentation, and "
        "troubleshooting experience across collaborative product teams while "
        "keeping implementation decisions grounded in measurable business and "
        "customer outcomes."
    )
    filler = (
        "Designed maintainable service behavior, documented technical decisions, "
        "tested edge cases, reviewed implementation details, coordinated release "
        "readiness, diagnosed production behavior, and improved operational "
        "clarity for a distributed product team"
    )
    current_bullets = [
        (
            "Built secure Stripe payment workflows in TypeScript and Node.js, "
            "reducing failed invoice retries by 28 percent while preserving "
            "auditable transaction state."
        ),
        *[
            f"{filler} during delivery cycle {index}."
            for index in range(1, 18)
        ],
    ]
    older_bullets = [
        (
            f"Supported cross-functional application delivery and {filler.lower()} "
            f"for platform initiative {index}."
        )
        for index in range(1, 15)
    ]
    return f"""\
# Candidate Name
candidate@example.com | Austin, TX

## Summary
{summary}

## Skills
**Languages:** TypeScript, JavaScript, C#, SQL
**Frameworks:** Node.js, React, ASP.NET Core
**Data & APIs:** PostgreSQL, REST APIs, Stripe

## Experience
### Senior Software Engineer — Current Company | 2023 – Present
{chr(10).join(f"- {bullet}" for bullet in current_bullets)}

### Software Engineer — Earlier Company | 2020 – 2023
{chr(10).join(f"- {bullet}" for bullet in older_bullets)}

## Education
Bachelor of Science in Computer Science — Example University
"""


class PublicationRepairTests(unittest.TestCase):
    def test_normalizes_education_alias(self):
        draft = "# Candidate\n\n## Education & Certifications\nB.S., University"
        repaired = repair_resume_for_publication(draft)
        self.assertIn("## Education\nB.S., University", repaired.markdown)
        self.assertNotIn("## Education & Certifications", repaired.markdown)

    def test_restores_verified_education_body_verbatim(self):
        body = (
            "Bachelor of Science in Computer Science — Texas Tech University\n"
            "- Lubbock, Texas | May 2022"
        )
        draft = "# Candidate\n\n## Summary\nA grounded candidate summary."
        profile = f"## Contact\nCandidate\n\n## Education & Certifications\n{body}\n"
        repaired = repair_resume_for_publication(
            draft,
            candidate_profile=profile,
        )
        self.assertIn(f"## Education\n{body}", repaired.markdown)
        self.assertIn("Restored the Education", " ".join(repaired.repair_notes))

    def test_does_not_restore_placeholder_education(self):
        repaired = repair_resume_for_publication(
            "# Candidate\n\n## Summary\nA grounded candidate summary.",
            candidate_profile=(
                "## Education & Certifications\n"
                "No education information was provided in the source material."
            ),
        )
        self.assertNotIn("## Education", repaired.markdown)

    def test_compacts_long_resume_and_preserves_claimable_keyword(self):
        draft = _long_resume()
        self.assertGreater(
            len(re.findall(r"\b[\w+#./-]+\b", draft)),
            1_120,
        )
        jd_analysis = """\
## ATS Keywords (verbatim)
- Stripe
- TypeScript
"""
        match_strategy = """\
## Keyword Placement Plan
| ATS Keyword | Placement |
|---|---|
| Stripe | Current role |
| TypeScript | Current role |
"""
        repaired = repair_resume_for_publication(
            draft,
            candidate_profile="",
            jd_analysis=jd_analysis,
            match_strategy=match_strategy,
        )
        audit = audit_resume_structure(repaired.markdown)
        self.assertLessEqual(repaired.final_word_count, 950)
        self.assertTrue(audit.valid, audit.issues)
        self.assertIn("Stripe payment workflows", repaired.markdown)
        self.assertGreaterEqual(audit.bullet_count, 4)
        self.assertIn("### Senior Software Engineer", repaired.markdown)
        self.assertIn("### Software Engineer", repaired.markdown)

    def test_repair_is_idempotent(self):
        first = repair_resume_for_publication(
            _long_resume(),
            jd_analysis="## ATS Keywords (verbatim)\n- Stripe",
            match_strategy=(
                "## Keyword Placement Plan\n"
                "| ATS Keyword | Placement |\n"
                "|---|---|\n"
                "| Stripe | Experience |"
            ),
        )
        second = repair_resume_for_publication(
            first.markdown,
            jd_analysis="## ATS Keywords (verbatim)\n- Stripe",
            match_strategy=(
                "## Keyword Placement Plan\n"
                "| ATS Keyword | Placement |\n"
                "|---|---|\n"
                "| Stripe | Experience |"
            ),
        )
        self.assertEqual(first.markdown, second.markdown)
        self.assertEqual(first.final_word_count, second.final_word_count)


if __name__ == "__main__":
    unittest.main()
