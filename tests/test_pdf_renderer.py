import unittest

from pypdf import PdfReader

from app.services.pdf_renderer import (
    markdown_to_html,
    normalize_resume_markdown,
    render_pdf,
)


SAMPLE_RESUME = """\
# Alex Candidate
alex@example.com | Austin, TX | linkedin.com/in/alex

## Summary
Full-stack engineer delivering regulated applications with C#, ASP.NET Core,
React, and TypeScript.

## Skills
**Languages:** C#, TypeScript **Frameworks:** ASP.NET Core, React **Tools:** Git, CI/CD

## Experience
### Software Engineer — Example Company | Remote | 2024–Present

- Built React interfaces backed by C# and ASP.NET Core REST APIs.
- Supported CI/CD releases and reviewed application changes.

## Education
**B.S., Computer Science — Example University**

## Certifications
AWS Certified Example
Azure Certified Example
"""


class PdfRendererTests(unittest.TestCase):
    def test_run_on_skill_categories_become_separate_paragraphs(self):
        normalized = normalize_resume_markdown(SAMPLE_RESUME)
        self.assertIn(
            "**Languages:** C#, TypeScript\n\n**Frameworks:** ASP.NET Core, React",
            normalized,
        )
        self.assertIn(
            "**Frameworks:** ASP.NET Core, React\n\n**Tools:** Git, CI/CD",
            normalized,
        )

    def test_html_uses_contact_skills_and_literal_hanging_hyphens(self):
        html = markdown_to_html(SAMPLE_RESUME)
        self.assertIn('class="contact"', html)
        self.assertEqual(html.count('class="skill-line"'), 3)
        self.assertIn('class="document-entry"', html)
        self.assertIn('class="resume-bullet">-&nbsp;', html)
        self.assertNotIn("resume-bullet keep-with-next", html)
        self.assertIn("<p>AWS Certified Example</p>", html)
        self.assertIn("<p>Azure Certified Example</p>", html)
        self.assertNotIn("<ul>", html)
        self.assertNotIn("<li>", html)

    def test_pdf_text_is_parseable_without_control_character_bullets(self):
        path = render_pdf(SAMPLE_RESUME, "_pdf_renderer_regression.pdf")
        reader = PdfReader(path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        self.assertEqual(len(reader.pages), 1)
        self.assertNotIn("\x7f", text)
        self.assertIn("- Built React interfaces", text)
        self.assertIn("Languages: C#, TypeScript", text)
        self.assertIn("Education", text)

    def test_long_resume_uses_dense_page_breakable_layout(self):
        long_resume = SAMPLE_RESUME + "\n" + ("delivery evidence " * 430)
        html = markdown_to_html(long_resume)
        self.assertIn('<body class="dense">', html)
        self.assertIn('class="document-entry"', html)
        self.assertNotIn("resume-bullet keep-with-next", html)


if __name__ == "__main__":
    unittest.main()
