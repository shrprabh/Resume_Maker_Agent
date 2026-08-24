"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const insights = require("../../app/static/insights.js");

const JD_FIXTURE = `## Target Company
Hicare.net Inc.

## Target Role
Senior Full-Stack Software Engineer

## Role Summary
Own backend services and supporting React interfaces for a digital-health platform.

## Must-Have Requirements
- Five years of backend experience
- Node.js
- NestJS
- React
- PostgreSQL
- Redis
- AWS
- Korean and English

## Nice-to-Have Requirements
- FHIR
- HL7
- Terraform
- Docker
- GitHub Actions
- Offline-first architecture
- Startup experience

## ATS Keywords (verbatim)
${Array.from({ length: 50 }, (_, index) => `- Keyword ${index + 1}`).join("\n")}

## Hidden Priorities
- Production ownership and reliability
- Secure handling of clinical data`;

const STRATEGY_FIXTURE = `## Requirement-to-Evidence Map

**Must-Have Requirements**
- Five years of backend experience: "Systems Engineer and Software Engineer roles, 2020-present"
- Node.js: "Built Node.js scientific workflows"
- NestJS: NO EVIDENCE
- React: "Developed React administration features"
- PostgreSQL: "PostgreSQL appears in the verified skills inventory"
- Redis: "Redis appears in the verified skills inventory"
- AWS: "AWS certification and deployment work"
- Korean and English: NO EVIDENCE

**Nice-to-Have Requirements**
- FHIR: NO EVIDENCE
- HL7: NO EVIDENCE
- Terraform: NO EVIDENCE
- Docker: "Containerized scientific services"
- GitHub Actions: "Automated GitHub Actions releases"
- Offline-first architecture: NO EVIDENCE
- Startup experience: NO EVIDENCE

## Genuine Gaps (do not paper over)
- NestJS and Next.js
- Bilingual Korean and English
- HIPAA and PHI
- FHIR and HL7
- Terraform
- Offline-first architecture
- Startup experience

## Keyword Placement Plan
- **Node.js:** LBNL experience
- **React:** MedLaunch experience

## Do-Not-Claim List
${Array.from({ length: 27 }, (_, index) => `- Protected term ${index + 1}`).join("\n")}`;

const CANDIDATE_FIXTURE = `## Contact
- **Name:** Test Candidate

## Work History
- **Software Engineer — Example Corp** (2024-present)
  - Improved reliability by 40%.
- **Systems Engineer — Earlier Corp** (2020-2023)
  - Built REST APIs.

## Projects
- **Clinical workflow prototype**
  - Built a React interface.

## Skills Inventory
- **Languages:** C#, TypeScript, JavaScript
- **Cloud & DevOps:** AWS, Docker, GitHub Actions

## Education & Certifications
- MS Computer Science

## Quantified Achievements Index
- Improved reliability by 40%.
- Served 10,000+ users.

## Conflicts & Gaps
**Conflicts**
- Two source documents use different role dates.`;

function reviewFixture() {
  const findings = [];
  for (let index = 1; index <= 3; index += 1) {
    findings.push(
      `${index}. CRITICAL FABRICATION: Unsupported claim ${index}.`
    );
  }
  for (let index = 4; index <= 27; index += 1) {
    findings.push(`${index}. ATS Gap: Missing supported keyword ${index}.`);
  }
  for (let index = 28; index <= 31; index += 1) {
    findings.push(`${index}. Craft Issue: Improve bullet ${index}.`);
  }
  findings.push("32. Missing required Education section.");
  return [
    "SCORE: 35/100 | ATS coverage: 92% | Fabrications: 3",
    ...findings,
  ].join("\n");
}

test("parses the representative agent reports into stable sections", () => {
  const jd = insights.parseMarkdownReport(JD_FIXTURE);
  const strategy = insights.parseMarkdownReport(STRATEGY_FIXTURE);

  assert.equal(jd.structured, true);
  assert.equal(
    jd.sections.find((section) => section.key === "must have requirements")
      .items.length,
    8
  );
  assert.equal(
    jd.sections.find((section) => section.key === "nice to have requirements")
      .items.length,
    7
  );
  assert.equal(
    jd.sections.find((section) => section.key === "ats keywords verbatim").items
      .length,
    50
  );
  assert.equal(
    strategy.sections.find(
      (section) => section.key === "genuine gaps do not paper over"
    ).items.length,
    7
  );
  assert.equal(
    strategy.sections.find((section) => section.key === "do not claim list")
      .items.length,
    27
  );
});

test("parses bold pseudo-headings in requirement maps", () => {
  const requirements = insights.parseRequirementMap(STRATEGY_FIXTURE);

  assert.equal(requirements.length, 15);
  assert.equal(
    requirements.filter((item) => item.priority === "must").length,
    8
  );
  assert.equal(
    requirements.filter((item) => item.priority === "nice").length,
    7
  );
  assert.equal(
    requirements.filter((item) => item.status === "gap").length,
    7
  );
  assert.deepEqual(
    requirements.find((item) => item.requirement === "Node.js"),
    {
      priority: "must",
      requirement: "Node.js",
      evidence: '"Built Node.js scientific workflows"',
      status: "supported",
    }
  );
});

test("supports level-three subsections and Markdown requirement tables", () => {
  const source = `## Requirement-to-Evidence Map
### Must-Have Requirements
| Requirement | Evidence |
| --- | --- |
| Node.js | Built a Node.js service |
| NestJS | NO EVIDENCE |
### Nice-to-Have Requirements
| Priority | Requirement | Evidence |
| --- | --- | --- |
| Nice | Docker | Containerized production services |`;
  const requirements = insights.parseRequirementMap(source);

  assert.deepEqual(
    requirements.map((item) => [
      item.priority,
      item.requirement,
      item.status,
    ]),
    [
      ["must", "Node.js", "supported"],
      ["must", "NestJS", "gap"],
      ["nice", "Docker", "supported"],
    ]
  );
});

test("groups all canonical reviewer findings without losing scores", () => {
  const review = insights.parseReviewFeedback(reviewFixture());

  assert.equal(review.formatValid, true);
  assert.equal(review.approved, false);
  assert.deepEqual(review.metrics, {
    score: 35,
    atsCoverage: 92,
    fabrications: 3,
  });
  assert.equal(review.findings.length, 32);
  assert.equal(review.groups.fabrication.length, 3);
  assert.equal(review.groups.ats.length, 24);
  assert.equal(review.groups.craft.length, 4);
  assert.equal(review.groups.structure.length, 1);
});

test("accepts approved prose and strict JSON reviewer contracts", () => {
  const approved = insights.parseReviewFeedback(
    "APPROVED — score 91/100, ATS coverage 96%, Fabrications: 0."
  );
  assert.equal(approved.formatValid, true);
  assert.equal(approved.approved, true);
  assert.equal(approved.metrics.score, 91);
  assert.equal(approved.findings.length, 0);

  const json = insights.parseReviewFeedback(
    JSON.stringify({
      qualityScore: "88/100",
      atsCoverage: "95%",
      unsupportedClaimCount: 0,
      verdict: "revision required",
      corrections: ["Craft Issue: Make the summary more specific."],
    })
  );
  assert.equal(json.formatValid, true);
  assert.equal(json.approved, false);
  assert.equal(json.metrics.score, 88);
  assert.equal(json.groups.craft.length, 1);
});

test("preserves malformed and free-form output as a graceful fallback", () => {
  const malformed = insights.parseReviewFeedback(
    "REVIEW UNAVAILABLE — the model returned an invalid review format."
  );
  assert.equal(malformed.formatValid, false);
  assert.equal(malformed.metrics.score, null);
  assert.equal(malformed.metrics.atsCoverage, null);
  assert.equal(malformed.metrics.fabrications, null);
  assert.equal(malformed.findings.length, 1);

  const freeForm = insights.parseMarkdownReport(
    "The agent returned one free-form explanation without headings."
  );
  assert.equal(freeForm.sections.length, 1);
  assert.equal(freeForm.sections[0].implicit, true);
  assert.match(freeForm.sections[0].paragraphs[0], /free-form explanation/);
});

test("keeps HTML and multilingual content literal for safe DOM rendering", () => {
  const payload =
    "## Role Summary\n<img src=x onerror=alert(1)> 한국어 Node.js C# C++";
  const report = insights.parseMarkdownReport(payload);
  const paragraph = report.sections[0].paragraphs[0];

  assert.match(paragraph, /<img src=x onerror=alert\(1\)>/);
  assert.match(paragraph, /한국어 Node\.js C# C\+\+/);

  const source = fs.readFileSync(
    path.resolve(__dirname, "../../app/static/insights.js"),
    "utf8"
  );
  assert.equal(source.includes(".innerHTML"), false);
  assert.match(source, /textContent/);
  assert.match(source, /replaceChildren/);
});

test("uses structured scorecard values as authority and preserves valid zero", () => {
  const model = insights.buildInsightsViewModel({
    review_valid: true,
    review_score: 0,
    approved: false,
    scores: {
      supported_ats_coverage: 92,
      overall_requirement_match: 73,
      evidence_integrity: 25,
      quality_score: 99,
      structure_valid: false,
      structure_issues: ["Missing required Education section."],
      claimable_keywords: ["Node.js", "AWS"],
      placed_keywords: ["Node.js"],
      missing_supported_keywords: ["AWS"],
      unsupported_keywords: ["NestJS"],
    },
    artifacts: {
      jd_analysis: JD_FIXTURE,
      candidate_profile: CANDIDATE_FIXTURE,
      match_strategy: STRATEGY_FIXTURE,
      review_feedback: reviewFixture(),
    },
    resume_markdown: "resume",
    cover_letter_markdown: "cover",
  });

  assert.deepEqual(
    model.metrics.map((metric) => metric.value),
    [92, 73, 25, 0]
  );
  assert.equal(model.review.metrics.score, 0);
  assert.deepEqual(model.keywords.placed, ["Node.js"]);
  assert.deepEqual(model.keywords.missingSupported, ["AWS"]);
  assert.deepEqual(model.keywords.protected, ["NestJS"]);
  assert.equal(model.candidateEvidence.roleCount, 2);
  assert.equal(model.candidateEvidence.projectCount, 1);
  assert.equal(model.candidateEvidence.quantifiedCount, 2);
});

test("never promotes raw reviewer numbers when review_valid is false", () => {
  const model = insights.buildInsightsViewModel({
    review_valid: false,
    review_score: 0,
    scores: {
      supported_ats_coverage: null,
      overall_requirement_match: null,
      evidence_integrity: null,
      quality_score: 100,
    },
    artifacts: {
      jd_analysis: "",
      candidate_profile: "",
      match_strategy: "",
      review_feedback:
        "SCORE: 100/100 | ATS coverage: 100% | Fabrications: 0",
    },
  });

  assert.equal(model.status, "unverified");
  assert.deepEqual(
    model.metrics.map((metric) => metric.value),
    [null, null, null, null]
  );
  assert.equal(model.review.metrics.score, null);
  assert.equal(model.review.metrics.atsCoverage, null);
  assert.equal(model.review.metrics.fabrications, null);
});

test("deduplicates structured keyword arrays without changing display spelling", () => {
  const model = insights.buildInsightsViewModel({
    review_valid: false,
    scores: {
      claimable_keywords: ["Node.js", "node.js", "C#", "C#"],
      placed_keywords: ["Node.js", "NODE.JS"],
      missing_supported_keywords: [],
      unsupported_keywords: [],
    },
    artifacts: {},
  });

  assert.deepEqual(model.keywords.claimable, ["Node.js", "C#"]);
  assert.deepEqual(model.keywords.placed, ["Node.js"]);
});
