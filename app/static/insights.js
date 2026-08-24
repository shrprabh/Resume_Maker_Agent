(function attachRoleFitInsights(globalObject, factory) {
  "use strict";

  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }
  if (globalObject) {
    globalObject.RoleFitInsights = api;
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function createApi() {
  "use strict";

  const MAX_INPUT_CHARS = 250000;
  const MAX_SECTIONS = 80;
  const MAX_ITEMS = 500;
  const MAX_DISPLAY_ITEMS = 80;

  const KNOWN_HEADINGS = [
    "Target Company",
    "Target Role",
    "Role Summary",
    "Must-Have Requirements",
    "Nice-to-Have Requirements",
    "ATS Keywords (verbatim)",
    "Key Responsibilities",
    "Hidden Priorities",
    "Contact",
    "Work History",
    "Projects",
    "Skills Inventory",
    "Education & Certifications",
    "Quantified Achievements Index",
    "Conflicts & Gaps",
    "Requirement-to-Evidence Map",
    "Genuine Gaps (do not paper over)",
    "Positioning Strategy",
    "Keyword Placement Plan",
    "Do-Not-Claim List",
    "User-Attested Gap Resolutions",
    "Score interpretation",
    "Verified keyword placement",
    "Missing but supported keywords",
    "Structure audit",
    "Protected exclusions",
    "User-attested gap resolutions",
    "Original genuine gaps",
    "Maximum-match auditor",
  ];

  function normalizedHeadingKey(value) {
    return String(value || "")
      .normalize("NFKC")
      .replace(/[*_`#]/g, "")
      .replace(/\([^)]*\)/g, " ")
      .replace(/[^a-zA-Z0-9]+/g, " ")
      .trim()
      .toLowerCase()
      .replace(/\s+/g, " ");
  }

  const HEADING_ALIASES = new Map(
    [
      ["company", "target company"],
      ["employer", "target company"],
      ["job title", "target role"],
      ["position", "target role"],
      ["must haves", "must have requirements"],
      ["must have", "must have requirements"],
      ["required qualifications", "must have requirements"],
      ["nice to haves", "nice to have requirements"],
      ["preferred qualifications", "nice to have requirements"],
      ["ats keywords", "ats keywords verbatim"],
      ["requirements to evidence map", "requirement to evidence map"],
      ["evidence map", "requirement to evidence map"],
      ["genuine gaps", "genuine gaps do not paper over"],
      ["gaps", "genuine gaps do not paper over"],
      ["do not claim", "do not claim list"],
      ["protected keywords", "do not claim list"],
      ["review", "maximum match auditor"],
    ].map(([alias, canonical]) => [
      normalizedHeadingKey(alias),
      normalizedHeadingKey(canonical),
    ])
  );

  function normalizeInput(value) {
    const original = String(value === null || value === undefined ? "" : value)
      .replace(/\u0000/g, "")
      .replace(/\r\n?/g, "\n");
    return {
      text: original.slice(0, MAX_INPUT_CHARS),
      truncated: original.length > MAX_INPUT_CHARS,
    };
  }

  function canonicalHeading(value) {
    const normalized = normalizedHeadingKey(value);
    return HEADING_ALIASES.has(normalized)
      ? HEADING_ALIASES.get(normalized)
      : normalized;
  }

  function cleanMarkdownText(value) {
    return String(value || "")
      .replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1")
      .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
      .replace(/\*\*([^*]+)\*\*/g, "$1")
      .replace(/__([^_]+)__/g, "$1")
      .replace(/`([^`]+)`/g, "$1")
      .replace(/\\([\\`*_[\]{}()#+.!-])/g, "$1")
      .trim();
  }

  function headingTitle(rawLine, knownKeys) {
    const line = rawLine.trim();
    let match = line.match(/^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$/);
    if (match) {
      return {
        title: cleanMarkdownText(match[2]),
        level: match[1].length,
        pseudo: false,
      };
    }
    match = line.match(/^\*\*([^*\n]{2,100})\*\*\s*:?\s*$/);
    if (match) {
      return {
        title: cleanMarkdownText(match[1]),
        level: 3,
        pseudo: true,
      };
    }
    const bare = cleanMarkdownText(line.replace(/:\s*$/, ""));
    if (knownKeys.has(canonicalHeading(bare))) {
      return { title: bare, level: 2, pseudo: false };
    }
    return null;
  }

  function parsePipeRow(line) {
    const trimmed = line.trim();
    if (!trimmed.includes("|")) return null;
    const value = trimmed.replace(/^\|/, "").replace(/\|$/, "");
    const cells = value.split("|").map((cell) => cleanMarkdownText(cell));
    return cells.length >= 2 ? cells : null;
  }

  function isPipeSeparator(cells) {
    return Boolean(
      cells &&
        cells.length &&
        cells.every((cell) => /^:?-{3,}:?$/.test(cell.replace(/\s+/g, "")))
    );
  }

  function parseMarkdownReport(value, extraHeadings) {
    const normalized = normalizeInput(value);
    const knownKeys = new Set(
      KNOWN_HEADINGS.concat(Array.isArray(extraHeadings) ? extraHeadings : [])
        .map(canonicalHeading)
        .filter(Boolean)
    );
    const report = {
      raw: normalized.text,
      truncated: normalized.truncated,
      structured: false,
      sections: [],
    };
    let section = null;
    let subsection = "";
    let lastItem = null;

    function createSection(title, level, implicit) {
      const next = {
        title: title || "Agent notes",
        key: canonicalHeading(title || "Agent notes"),
        level: level || 2,
        implicit: Boolean(implicit),
        paragraphs: [],
        items: [],
        rows: [],
      };
      if (report.sections.length < MAX_SECTIONS) {
        report.sections.push(next);
      }
      section = next;
      subsection = "";
      lastItem = null;
      return next;
    }

    function ensureSection() {
      return section || createSection("Agent notes", 2, true);
    }

    normalized.text.split("\n").forEach((rawLine) => {
      if (report.sections.length >= MAX_SECTIONS) return;
      const heading = headingTitle(rawLine, knownKeys);
      if (heading) {
        if (heading.pseudo && section && !section.implicit) {
          subsection = heading.title;
          section.items.push({
            text: heading.title,
            depth: 0,
            marker: "",
            subsection,
            kind: "subheading",
          });
          lastItem = null;
        } else {
          createSection(heading.title, heading.level, false);
          report.structured = true;
        }
        return;
      }

      const trimmed = rawLine.trim();
      if (!trimmed) {
        lastItem = null;
        return;
      }

      const cells = parsePipeRow(trimmed);
      if (cells) {
        if (!isPipeSeparator(cells)) {
          ensureSection().rows.push({ cells, subsection });
        }
        lastItem = null;
        return;
      }

      const listMatch = rawLine.match(/^([ \t]*)([-+*]|\d+[.)])\s+(.+)$/);
      if (listMatch) {
        const indentation = listMatch[1].replace(/\t/g, "    ").length;
        const item = {
          text: cleanMarkdownText(listMatch[3]),
          depth: Math.min(4, Math.floor(indentation / 2)),
          marker: listMatch[2],
          subsection,
          kind: /^\d/.test(listMatch[2]) ? "ordered" : "bullet",
        };
        const target = ensureSection();
        if (target.items.length < MAX_ITEMS) target.items.push(item);
        lastItem = item;
        return;
      }

      if (/^\s+/.test(rawLine) && lastItem) {
        lastItem.text = `${lastItem.text} ${cleanMarkdownText(trimmed)}`.trim();
        return;
      }

      const target = ensureSection();
      const cleaned = cleanMarkdownText(trimmed);
      const previous = target.paragraphs[target.paragraphs.length - 1];
      if (previous && !/[.!?:]$/.test(previous)) {
        target.paragraphs[target.paragraphs.length - 1] =
          `${previous} ${cleaned}`.trim();
      } else {
        target.paragraphs.push(cleaned);
      }
      lastItem = null;
    });

    if (!report.sections.length && normalized.text.trim()) {
      createSection("Agent notes", 2, true).paragraphs.push(
        cleanMarkdownText(normalized.text.trim())
      );
    }
    return report;
  }

  function matchingSections(report, names) {
    const keys = new Set(names.map(canonicalHeading));
    return (report && Array.isArray(report.sections) ? report.sections : []).filter(
      (section) => keys.has(section.key)
    );
  }

  function sectionText(report, names) {
    const sections = matchingSections(report, names);
    const values = [];
    sections.forEach((section) => {
      values.push(...section.paragraphs);
      section.items
        .filter((item) => item.kind !== "subheading")
        .forEach((item) => values.push(item.text));
    });
    return values.filter(Boolean).join(" ").trim();
  }

  function sectionItems(report, names, options) {
    const includeNested = !options || options.includeNested !== false;
    const items = [];
    matchingSections(report, names).forEach((section) => {
      section.items.forEach((item) => {
        if (
          item.kind !== "subheading" &&
          (includeNested || item.depth === 0) &&
          item.text
        ) {
          items.push(item);
        }
      });
    });
    return items;
  }

  function splitLabelAndDetail(value) {
    const text = cleanMarkdownText(value);
    const separator = text.indexOf(": ");
    if (separator < 1) return { label: text, detail: "" };
    return {
      label: text.slice(0, separator).trim(),
      detail: text.slice(separator + 2).trim(),
    };
  }

  function requirementPriority(subsection) {
    const key = canonicalHeading(subsection);
    if (key.includes("nice") || key.includes("preferred")) return "nice";
    return "must";
  }

  function requirementStatus(evidence) {
    if (!evidence) return "unknown";
    return /\b(?:NO EVIDENCE|UNSUPPORTED|NOT FOUND|ABSENT)\b/i.test(evidence)
      ? "gap"
      : "supported";
  }

  function parseRequirementMap(value) {
    const report =
      typeof value === "string" ? parseMarkdownReport(value) : value || {};
    const sections = matchingSections(report, [
      "Requirement-to-Evidence Map",
      "Requirements-to-Evidence Map",
      "Evidence Map",
      "Must-Have Requirements",
      "Nice-to-Have Requirements",
    ]);
    const requirements = [];

    sections.forEach((section) => {
      section.items.forEach((item) => {
        if (item.kind === "subheading") return;
        const pair = splitLabelAndDetail(item.text);
        if (!pair.label) return;
        requirements.push({
          priority: requirementPriority(item.subsection || section.title),
          requirement: pair.label,
          evidence: pair.detail,
          status: requirementStatus(pair.detail),
        });
      });

      section.rows.forEach((row, rowIndex) => {
        const cells = row.cells;
        const first = canonicalHeading(cells[0]);
        if (
          rowIndex === 0 &&
          ["priority", "type", "requirement", "requirement keyword"].includes(first)
        ) {
          return;
        }
        let priority = requirementPriority(row.subsection || section.title);
        let requirement = cells[0];
        let evidence = cells.slice(1).join(" — ");
        if (/^(must|nice|preferred|required)/.test(first) && cells.length >= 3) {
          priority = first.startsWith("nice") || first.startsWith("preferred")
            ? "nice"
            : "must";
          requirement = cells[1];
          evidence = cells.slice(2).join(" — ");
        }
        if (!requirement) return;
        requirements.push({
          priority,
          requirement,
          evidence,
          status: requirementStatus(evidence),
        });
      });
    });

    const seen = new Set();
    return requirements.filter((item) => {
      const key = `${item.priority}:${canonicalHeading(item.requirement)}`;
      if (!canonicalHeading(item.requirement) || seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  }

  function safePercentage(value) {
    if (typeof value === "string" && /^\s*\d{1,3}\s*(?:%|\/\s*100)?\s*$/.test(value)) {
      value = Number.parseInt(value, 10);
    }
    return Number.isInteger(value) && value >= 0 && value <= 100 ? value : null;
  }

  function safeCount(value) {
    if (typeof value === "string" && /^\s*\d+\s*$/.test(value)) {
      value = Number.parseInt(value, 10);
    }
    return Number.isInteger(value) && value >= 0 ? value : null;
  }

  function jsonReviewerPayload(text) {
    const candidates = [text.trim()];
    const fence = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
    if (fence) candidates.push(fence[1].trim());
    for (const candidate of candidates) {
      if (!candidate.startsWith("{")) continue;
      try {
        const parsed = JSON.parse(candidate);
        if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
          return parsed;
        }
      } catch (_error) {
        // A malformed JSON candidate is handled by the prose fallback.
      }
    }
    return null;
  }

  function firstMappedValue(payload, aliases) {
    if (!payload) return undefined;
    const reviewerKey = (value) =>
      normalizedHeadingKey(value).replace(/\s+/g, "");
    const canonicalAliases = new Set(aliases.map(reviewerKey));
    const key = Object.keys(payload).find((name) =>
      canonicalAliases.has(reviewerKey(name))
    );
    return key === undefined ? undefined : payload[key];
  }

  function classifyFinding(message) {
    const value = String(message || "").trim();
    if (/^(?:CRITICAL\s+)?FABRICATION\b/i.test(value)) {
      return { category: "fabrication", severity: "critical" };
    }
    if (/^(?:ATS|KEYWORD)(?:\s+GAP|\s+ISSUE|:|\b)/i.test(value)) {
      return { category: "ats", severity: "warning" };
    }
    if (/^(?:MISSING\s+REQUIRED|STRUCTURE|FORMAT)\b/i.test(value)) {
      return { category: "structure", severity: "warning" };
    }
    if (/^(?:CRAFT|WRITING|STYLE)\s+(?:ISSUE|GAP)?\b/i.test(value)) {
      return { category: "craft", severity: "advisory" };
    }
    return { category: "general", severity: "advisory" };
  }

  function parseReviewFeedback(value) {
    const normalized = normalizeInput(value);
    const text = normalized.text.trim();
    const result = {
      raw: normalized.text,
      truncated: normalized.truncated,
      formatValid: false,
      approved: null,
      metrics: {
        score: null,
        atsCoverage: null,
        fabrications: null,
      },
      findings: [],
      groups: {
        fabrication: [],
        ats: [],
        structure: [],
        craft: [],
        general: [],
      },
    };
    if (!text) return result;

    const payload = jsonReviewerPayload(text);
    let feedbackValues = null;
    if (payload) {
      result.metrics.score = safePercentage(
        firstMappedValue(payload, ["score", "review score", "quality score"])
      );
      result.metrics.atsCoverage = safePercentage(
        firstMappedValue(payload, [
          "ats coverage",
          "atsCoverage",
          "supported ATS coverage",
        ])
      );
      result.metrics.fabrications = safeCount(
        firstMappedValue(payload, [
          "fabrication count",
          "fabrications",
          "unsupported claim count",
        ])
      );
      const approvedValue = firstMappedValue(payload, [
        "approved",
        "is approved",
        "verdict",
      ]);
      if (typeof approvedValue === "boolean") {
        result.approved = approvedValue;
      } else if (typeof approvedValue === "string") {
        if (/^(?:approved|pass|passed|true|yes)$/i.test(approvedValue.trim())) {
          result.approved = true;
        } else if (
          /^(?:rejected|revise|revision required|fail|failed|false|no)$/i.test(
            approvedValue.trim()
          )
        ) {
          result.approved = false;
        }
      }
      const rawFeedback = firstMappedValue(payload, [
        "feedback",
        "issues",
        "corrections",
        "recommendations",
      ]);
      feedbackValues = Array.isArray(rawFeedback)
        ? rawFeedback.map(String)
        : typeof rawFeedback === "string"
          ? [rawFeedback]
          : [];
    } else {
      const scoreMatch = text.match(/\bscore\s*[:—-]?\s*(\d{1,3})\s*(?:\/\s*100|%)?/i);
      const atsMatch = text.match(
        /\b(?:supported\s+)?ATS\s+coverage\s*[:—-]?\s*(\d{1,3})\s*(?:%|\/\s*100)?/i
      );
      const fabricationMatch = text.match(
        /\bfabrications?(?:\s+count)?\s*[:—-]?\s*(\d+)/i
      );
      result.metrics.score = safePercentage(scoreMatch && scoreMatch[1]);
      result.metrics.atsCoverage = safePercentage(atsMatch && atsMatch[1]);
      result.metrics.fabrications = safeCount(
        fabricationMatch && fabricationMatch[1]
      );
      if (/^\s*APPROVED\b/i.test(text)) result.approved = true;
      if (/^\s*(?:SCORE|REJECTED|REVISION REQUIRED)\b/i.test(text)) {
        result.approved = false;
      }
    }

    const completeMetrics =
      result.metrics.score !== null &&
      result.metrics.atsCoverage !== null &&
      result.metrics.fabrications !== null;
    result.formatValid = completeMetrics && result.approved !== null;

    let findings = [];
    if (feedbackValues) {
      findings = feedbackValues.map((item) => cleanMarkdownText(item)).filter(Boolean);
    } else {
      let current = null;
      text.split("\n").forEach((line) => {
        const match = line.match(/^\s*(?:\d+[.)]|[-*+])\s+(.+)$/);
        if (match) {
          current = cleanMarkdownText(match[1]);
          findings.push(current);
        } else if (/^\s+/.test(line) && current && line.trim()) {
          current = `${current} ${cleanMarkdownText(line.trim())}`;
          findings[findings.length - 1] = current;
        }
      });
    }
    if (!findings.length && !result.approved) {
      const prose = text
        .split("\n")
        .filter(
          (line) =>
            line.trim() &&
            !/\bscore\s*[:—-]?\s*\d|ATS\s+coverage|fabrications?\s*:/i.test(line)
        )
        .join(" ")
        .trim();
      if (prose) findings.push(cleanMarkdownText(prose));
    }

    result.findings = findings.slice(0, MAX_ITEMS).map((message, index) => ({
      id: index + 1,
      message,
      ...classifyFinding(message),
    }));
    result.findings.forEach((finding) => {
      result.groups[finding.category].push(finding);
    });
    return result;
  }

  function strings(value) {
    if (!Array.isArray(value)) return [];
    const seen = new Set();
    return value
      .map((item) => String(item === null || item === undefined ? "" : item).trim())
      .filter((item) => {
        const key = item.normalize("NFKC").toLowerCase();
        if (!item || seen.has(key)) return false;
        seen.add(key);
        return true;
      });
  }

  function labeledItems(report, names, topLevelOnly) {
    return sectionItems(report, names, {
      includeNested: !topLevelOnly,
    }).map((item) => ({ ...splitLabelAndDetail(item.text), depth: item.depth }));
  }

  function buildInsightsViewModel(data) {
    const input = data && typeof data === "object" ? data : {};
    const artifacts =
      input.artifacts && typeof input.artifacts === "object" ? input.artifacts : {};
    const scores = input.scores && typeof input.scores === "object" ? input.scores : {};
    const jd = parseMarkdownReport(artifacts.jd_analysis || "");
    const candidate = parseMarkdownReport(artifacts.candidate_profile || "");
    const strategy = parseMarkdownReport(artifacts.match_strategy || "");
    const review = parseReviewFeedback(artifacts.review_feedback || "");
    const requirements = parseRequirementMap(strategy);
    const maxData =
      input.maximum_match && typeof input.maximum_match === "object"
        ? input.maximum_match
        : input.maximumMatch && typeof input.maximumMatch === "object"
          ? input.maximumMatch
          : {};
    const maximumRaw =
      maxData.insights_markdown || input.maximum_insights_markdown || "";
    const maximum = parseMarkdownReport(maximumRaw);

    const reviewVerified = input.review_valid === true;
    if (reviewVerified) {
      review.metrics.score = safePercentage(
        input.review_score !== undefined
          ? input.review_score
          : scores.quality_score
      );
    } else {
      review.metrics.score = null;
      review.metrics.atsCoverage = null;
      review.metrics.fabrications = null;
    }

    const workItems = sectionItems(candidate, ["Work History"]);
    const projectItems = sectionItems(candidate, ["Projects"]);
    const skillItems = labeledItems(candidate, ["Skills Inventory"], false);
    const quantified = sectionItems(candidate, ["Quantified Achievements Index"]);
    const education = sectionItems(candidate, ["Education & Certifications"]);
    const conflicts = sectionItems(candidate, ["Conflicts & Gaps"]);
    const gaps = sectionItems(strategy, [
      "Genuine Gaps (do not paper over)",
      "Genuine Gaps",
    ]).map((item) => item.text);
    const protectedTerms = strings(scores.unsupported_keywords);
    const fallbackProtected = sectionItems(strategy, [
      "Do-Not-Claim List",
      "Do Not Claim List",
    ]).map((item) => item.text);
    const claimable = strings(scores.claimable_keywords);
    const placed = strings(scores.placed_keywords);
    const missingSupported = strings(scores.missing_supported_keywords);
    const atsKeywords = sectionItems(jd, ["ATS Keywords (verbatim)", "ATS Keywords"]).map(
      (item) => item.text
    );

    const supportedAtsCoverage = safePercentage(scores.supported_ats_coverage);
    const overallRequirementMatch = safePercentage(
      scores.overall_requirement_match
    );
    const evidenceIntegrity = safePercentage(scores.evidence_integrity);
    const qualityScore = reviewVerified
      ? safePercentage(
          input.review_score !== undefined
            ? input.review_score
            : scores.quality_score
        )
      : null;

    const mustRequirements = requirements.filter(
      (item) => item.priority === "must"
    );
    const mappedMust = mustRequirements.filter(
      (item) => item.status === "supported"
    ).length;
    const reviewStatus = reviewVerified
      ? input.approved
        ? "approved"
        : "review"
      : "unverified";

    return {
      status: reviewStatus,
      metrics: [
        {
          key: "ats",
          label: "Supported ATS coverage",
          value: supportedAtsCoverage,
          unit: "%",
          help: "Evidence-backed keywords present in the resume",
        },
        {
          key: "requirements",
          label: "Overall requirement match",
          value: overallRequirementMatch,
          unit: "%",
          help: "Role requirements mapped to candidate evidence",
        },
        {
          key: "integrity",
          label: "Evidence integrity",
          value: evidenceIntegrity,
          unit: "%",
          help: "Reviewer assessment of supported final claims",
        },
        {
          key: "quality",
          label: "Quality review",
          value: qualityScore,
          unit: qualityScore === null ? "" : "/100",
          help: reviewVerified
            ? "Verified reviewer result"
            : "The model review could not be verified",
        },
      ],
      role: {
        company: sectionText(jd, ["Target Company"]),
        title: sectionText(jd, ["Target Role"]),
        summary: sectionText(jd, ["Role Summary"]),
        mustHave: sectionItems(jd, ["Must-Have Requirements"]).map(
          (item) => item.text
        ),
        niceToHave: sectionItems(jd, ["Nice-to-Have Requirements"]).map(
          (item) => item.text
        ),
        responsibilities: sectionItems(jd, ["Key Responsibilities"]).map(
          (item) => item.text
        ),
        hiddenPriorities: sectionItems(jd, ["Hidden Priorities"]).map(
          (item) => item.text
        ),
      },
      requirementSummary: {
        mappedMust,
        totalMust: mustRequirements.length,
        gaps: mustRequirements.length - mappedMust,
      },
      requirements,
      gaps,
      keywords: {
        claimable,
        placed,
        missingSupported,
        protected: protectedTerms.length ? protectedTerms : fallbackProtected,
        jd: atsKeywords,
      },
      strategy: {
        positioning: sectionText(strategy, ["Positioning Strategy"]),
        keywordPlan: sectionItems(strategy, ["Keyword Placement Plan"]).map(
          (item) => item.text
        ),
      },
      candidateEvidence: {
        roleCount: workItems.filter((item) => item.depth === 0).length,
        projectCount: projectItems.filter((item) => item.depth === 0).length,
        quantifiedCount: quantified.length,
        workItems,
        projectItems,
        skills: skillItems,
        quantified: quantified.map((item) => item.text),
        education: education.map((item) => item.text),
        conflicts: conflicts.map((item) => item.text),
      },
      review: {
        ...review,
        verified: reviewVerified,
        structureIssues: strings(scores.structure_issues),
        structureValid:
          typeof scores.structure_valid === "boolean"
            ? scores.structure_valid
            : null,
      },
      maximum: {
        available: Boolean(maximumRaw.trim()),
        report: maximum,
        raw: maximumRaw,
        scores:
          maxData.scores && typeof maxData.scores === "object"
            ? maxData.scores
            : {},
      },
      flow: [
        { title: "Role intelligence", agent: "JD Analyzer", complete: Boolean(jd.raw.trim()) },
        {
          title: "Evidence inventory",
          agent: "Profile Analyzer",
          complete: Boolean(candidate.raw.trim()),
        },
        {
          title: "Match strategy",
          agent: "Positioning Strategist",
          complete: Boolean(strategy.raw.trim()),
        },
        {
          title: "Resume drafting",
          agent: "Resume Writer",
          complete: Boolean(String(input.resume_markdown || "").trim()),
        },
        {
          title: "Quality review",
          agent: "ATS & Fact Reviewer",
          complete: Boolean(review.raw.trim()),
        },
        {
          title: "Cover letter",
          agent: "Letter Writer",
          complete: Boolean(String(input.cover_letter_markdown || "").trim()),
        },
      ],
      reports: { jd, candidate, strategy },
    };
  }

  function render(root, data) {
    if (!root || typeof root.replaceChildren !== "function") {
      throw new TypeError("RoleFitInsights.render requires a DOM element");
    }
    const doc = root.ownerDocument ||
      (typeof document !== "undefined" ? document : null);
    if (!doc) throw new TypeError("A document is required to render insights");
    const model = buildInsightsViewModel(data);

    function node(tagName, className, text) {
      const element = doc.createElement(tagName);
      if (className) element.className = className;
      if (text !== undefined && text !== null) element.textContent = String(text);
      return element;
    }

    function append(parent) {
      for (let index = 1; index < arguments.length; index += 1) {
        const child = arguments[index];
        if (child) parent.append(child);
      }
      return parent;
    }

    function emptyState(message) {
      return node("p", "insight-inline-empty", message);
    }

    function listCard(title, values, className) {
      const card = node(
        "section",
        `insight-section-card ${className || ""}`.trim()
      );
      append(card, node("h5", "", title));
      if (!values.length) {
        card.append(emptyState("No items were returned for this section."));
        return card;
      }
      const list = node("ul", "");
      values.slice(0, MAX_DISPLAY_ITEMS).forEach((value) => {
        list.append(node("li", "", value));
      });
      card.append(list);
      return card;
    }

    function chipCloud(title, values, tone) {
      const card = node(
        "section",
        "insight-section-card"
      );
      append(card, node("h5", "", title));
      if (!values.length) {
        card.append(emptyState("None."));
        return card;
      }
      const cloud = node("div", "insight-chip-cloud");
      values.slice(0, MAX_DISPLAY_ITEMS).forEach((value) => {
        cloud.append(
          node(
            "span",
            `insight-chip ${tone || ""}`.trim(),
            value
          )
        );
      });
      card.append(cloud);
      return card;
    }

    function rawDisclosure(title, raw, open) {
      const details = node(
        "details",
        "insight-disclosure insight-section-card--wide"
      );
      if (open) details.open = true;
      const summary = node("summary", "", title);
      const body = node("div", "insight-disclosure-body");
      const pre = node(
        "pre",
        "insight-raw-fallback",
        raw || "No raw artifact was returned."
      );
      body.append(pre);
      append(details, summary, body);
      return details;
    }

    function panel(id, label, description, badge) {
      const section = node("section", "insight-panel");
      section.id = `insights-panel-${id}`;
      section.setAttribute("role", "tabpanel");
      section.setAttribute("aria-labelledby", `insights-tab-${id}`);
      section.tabIndex = 0;
      section.dataset.panel = id;
      section.hidden = true;
      const header = node("header", "insight-panel-header");
      const heading = node("div", "");
      append(
        heading,
        node("h4", "", label),
        node("p", "", description)
      );
      append(
        header,
        heading,
        node("span", "insight-panel-badge", badge || "Agent output")
      );
      const content = node(
        "div",
        "insight-panel-body insight-section-grid"
      );
      section.insightsContent = content;
      append(section, header, content);
      return section;
    }

    const dashboard = node("div", "insights-dashboard-inner");
    const hero = node("header", "insights-hero");
    const heroCopy = node("div", "insights-hero-copy");
    append(
      heroCopy,
      node("span", "insights-eyebrow", "Agent decision dashboard"),
      node("h3", "", model.role.title || "Application intelligence"),
      node(
        "p",
        "",
        model.role.company
          ? `${model.role.company} · Trace every recommendation back to role requirements and candidate evidence.`
          : "Trace every recommendation back to role requirements and candidate evidence."
      )
    );
    const statusLabel =
      model.status === "approved"
        ? "Quality gate passed"
        : model.status === "review"
          ? "Review required"
          : "Model review not verified";
    const hasFabrications =
      model.review.metrics.fabrications !== null &&
      model.review.metrics.fabrications > 0;
    const statusTone =
      model.status === "approved"
        ? "is-good"
        : hasFabrications
          ? "is-danger"
          : model.status === "review"
          ? "is-warning"
          : "is-neutral";
    const status = node(
      "div",
      `insights-status status-${model.status} ${statusTone}`
    );
    status.dataset.tone =
      model.status === "approved"
        ? "good"
        : hasFabrications
          ? "danger"
          : model.status === "review"
          ? "warning"
          : "neutral";
    append(
      status,
      node(
        "span",
        "insights-status-icon",
        model.status === "approved" ? "✓" : "!"
      ),
      node("strong", "", statusLabel),
      node(
        "p",
        "",
        model.status === "approved"
          ? "The reviewer verified the final draft and its supporting evidence."
          : hasFabrications
            ? `${model.review.metrics.fabrications} unsupported claim(s) require attention before applying.`
            : model.status === "review"
              ? `${model.review.findings.length} reviewer finding(s) are organized in the quality view.`
              : "No false score is shown. Inspect the preserved reviewer response."
      )
    );
    append(hero, heroCopy, status);
    dashboard.append(hero);

    const metrics = node("div", "insights-metrics");
    metrics.setAttribute("role", "list");
    model.metrics.forEach((metric) => {
      const metricTone =
        metric.value === null
          ? "neutral"
          : metric.key === "integrity" && metric.value < 100
            ? "danger"
            : metric.value < 70
              ? "warning"
              : "good";
      const card = node(
        "article",
        `insight-metric metric-${metric.key} is-${metricTone}`
      );
      card.dataset.tone = metricTone;
      card.setAttribute("role", "listitem");
      card.setAttribute(
        "aria-label",
        `${metric.label}: ${
          metric.value === null ? "unavailable" : `${metric.value}${metric.unit}`
        }`
      );
      const ring = node(
        "div",
        `metric-ring${metric.value === null ? " is-unavailable" : ""}`
      );
      ring.style.setProperty(
        "--score",
        String(metric.value === null ? 0 : metric.value)
      );
      ring.dataset.available = metric.value === null ? "false" : "true";
      ring.setAttribute("aria-hidden", "true");
      ring.append(
        node(
          "strong",
          "metric-ring-value",
          metric.value === null ? "—" : `${metric.value}${metric.unit}`
        )
      );
      const copy = node("div", "insight-metric-copy");
      append(
        copy,
        node("strong", "", metric.label),
        node("small", "", metric.help)
      );
      if (metric.value !== null) {
        const meter = node("div", "metric-bar");
        meter.setAttribute("role", "progressbar");
        meter.setAttribute("aria-label", metric.label);
        meter.setAttribute("aria-valuemin", "0");
        meter.setAttribute("aria-valuemax", "100");
        meter.setAttribute("aria-valuenow", String(metric.value));
        meter.style.setProperty("--score", String(metric.value));
        const fill = node("span", "");
        meter.append(fill);
        copy.append(meter);
      }
      append(card, ring, copy);
      metrics.append(card);
    });
    dashboard.append(metrics);

    const flow = node("ol", "insights-flow");
    flow.setAttribute("aria-label", "Agent workflow");
    model.flow.forEach((step, index) => {
      const item = node(
        "li",
        `insight-flow-step ${
          step.complete ? "complete" : "attention"
        }`
      );
      append(
        item,
        node(
          "span",
          "flow-index",
          String(index + 1).padStart(2, "0")
        )
      );
      const copy = node("span", "flow-copy");
      append(
        copy,
        node("strong", "", step.title),
        node("small", "", step.agent),
        node(
          "span",
          "flow-state",
          step.complete ? "Done" : "Unavailable"
        )
      );
      item.append(copy);
      flow.append(item);
    });
    dashboard.append(flow);

    const workspace = node("div", "insights-workspace");
    const sidebar = node("nav", "insights-sidebar");
    sidebar.setAttribute("role", "tablist");
    sidebar.setAttribute("aria-label", "Agent insight views");
    sidebar.setAttribute("aria-orientation", "vertical");
    sidebar.append(node("span", "insights-sidebar-label", "Decision views"));
    const panels = node("div", "insights-panel-stack");
    const definitions = [
      {
        id: "overview",
        label: "Overview",
        agent: "Combined agent summary",
        description:
          "A compact decision brief combining role fit, evidence coverage, and review risk.",
        badge: "All agents",
      },
      {
        id: "role",
        label: "Role intelligence",
        agent: "JD Analyzer",
        description:
          "The requirements, priorities, and exact ATS language extracted from the job description.",
        badge: "Agent 01",
      },
      {
        id: "evidence",
        label: "Candidate evidence",
        agent: "Profile Analyzer",
        description:
          "Verified work history, projects, skills, achievements, education, and source conflicts.",
        badge: "Agent 02",
      },
      {
        id: "match",
        label: "Match strategy",
        agent: "Positioning Strategist",
        description:
          "A requirement-by-requirement evidence map with supported keywords and protected gaps.",
        badge: "Agent 03",
      },
      {
        id: "review",
        label: "Quality review",
        agent: "ATS & Fact Reviewer",
        description:
          "Reviewer findings grouped by evidence integrity, document structure, ATS fit, and craft.",
        badge: "Agent 05",
      },
    ];
    if (model.maximum.available) {
      definitions.push({
        id: "maximum",
        label: "Maximum match",
        agent: "Evidence audit branch",
        description:
          "The separately audited maximum-match report, including user-attested gap evidence.",
        badge: "Optional branch",
      });
    }

    const tabs = [];
    const panelElements = [];
    definitions.forEach((definition, index) => {
      const { id, label, agent, description, badge } = definition;
      const button = node(
        "button",
        `insight-nav-button ${index === 0 ? "active" : ""}`
      );
      button.type = "button";
      button.id = `insights-tab-${id}`;
      button.dataset.target = id;
      button.setAttribute("role", "tab");
      button.setAttribute("aria-controls", `insights-panel-${id}`);
      button.setAttribute("aria-selected", index === 0 ? "true" : "false");
      button.tabIndex = index === 0 ? 0 : -1;
      const buttonCopy = node("span", "insight-nav-copy");
      append(
        buttonCopy,
        node("strong", "", label),
        node("small", "", agent)
      );
      append(
        button,
        node(
          "span",
          "insight-nav-number",
          String(index + 1).padStart(2, "0")
        ),
        buttonCopy,
        node(
          "span",
          `insight-nav-state${
            id === "review" && model.status !== "approved" ? " attention" : ""
          }`
        )
      );
      sidebar.append(button);
      tabs.push(button);
      const content = panel(id, label, description, badge);
      content.hidden = index !== 0;
      content.setAttribute("aria-hidden", index === 0 ? "false" : "true");
      panels.append(content);
      panelElements.push(content);
    });

    function activateTab(index, focus) {
      tabs.forEach((tab, itemIndex) => {
        const selected = itemIndex === index;
        tab.setAttribute("aria-selected", selected ? "true" : "false");
        tab.tabIndex = selected ? 0 : -1;
        tab.classList.toggle("active", selected);
        panelElements[itemIndex].hidden = !selected;
        panelElements[itemIndex].setAttribute(
          "aria-hidden",
          selected ? "false" : "true"
        );
      });
      if (focus) tabs[index].focus();
    }

    tabs.forEach((tab, index) => {
      tab.addEventListener("click", () => activateTab(index, false));
      tab.addEventListener("keydown", (event) => {
        let next = null;
        if (event.key === "ArrowDown" || event.key === "ArrowRight") {
          next = (index + 1) % tabs.length;
        } else if (event.key === "ArrowUp" || event.key === "ArrowLeft") {
          next = (index - 1 + tabs.length) % tabs.length;
        } else if (event.key === "Home") {
          next = 0;
        } else if (event.key === "End") {
          next = tabs.length - 1;
        }
        if (next !== null) {
          event.preventDefault();
          activateTab(next, true);
        }
      });
    });

    const overview = panelElements[0].insightsContent;
    const summaryCard = node(
      "section",
      "insight-section-card wide insights-summary-card"
    );
    append(
      summaryCard,
      node("h5", "", "What the agents found"),
      node(
        "p",
        "",
        model.role.summary ||
          "The role-summary artifact was unavailable. Review the raw job analysis."
      )
    );
    const counts = node(
      "div",
      "insight-fact-grid insights-summary-counts"
    );
    [
      [
        model.requirementSummary.mappedMust,
        `${model.requirementSummary.totalMust} must-haves mapped`,
      ],
      [model.gaps.length, "genuine evidence gaps"],
      [model.keywords.placed.length, "supported keywords placed"],
      [model.review.findings.length, "review findings"],
    ].forEach(([value, label]) => {
      const item = node("div", "insight-fact insights-summary-count");
      append(item, node("span", "", label), node("strong", "", value));
      counts.append(item);
    });
    summaryCard.append(counts);
    overview.append(summaryCard);
    if (
      model.review.metrics.fabrications !== null &&
      model.review.metrics.fabrications > 0
    ) {
      const alert = node(
        "section",
        "review-alert is-danger wide"
      );
      append(
        alert,
        node("strong", "", "Evidence-integrity review required"),
        node(
          "p",
          "",
          `The reviewer identified ${model.review.metrics.fabrications} unsupported claim(s). High keyword coverage does not override evidence integrity.`
        )
      );
      overview.append(alert);
    }
    overview.append(
      listCard(
        "Hidden hiring priorities",
        model.role.hiddenPriorities,
        "insights-priority-card"
      ),
      rawDisclosure("View all raw agent artifacts", [
        "JOB ANALYSIS",
        model.reports.jd.raw,
        "",
        "CANDIDATE PROFILE",
        model.reports.candidate.raw,
        "",
        "MATCH STRATEGY",
        model.reports.strategy.raw,
        "",
        "REVIEW FEEDBACK",
        model.review.raw,
      ].join("\n"))
    );

    const rolePanel = panelElements[1].insightsContent;
    const roleSnapshot = node(
      "section",
      "insight-section-card wide insights-role-card is-highlight"
    );
    append(
      roleSnapshot,
      node("span", "insights-eyebrow", model.role.company || "Target company"),
      node("h3", "", model.role.title || "Target role unavailable"),
      node("p", "", model.role.summary || "No role summary was returned.")
    );
    rolePanel.append(
      roleSnapshot,
      listCard("Must-have requirements", model.role.mustHave, "is-must-have"),
      listCard("Nice-to-have requirements", model.role.niceToHave, "is-nice-to-have"),
      listCard("Key responsibilities", model.role.responsibilities),
      chipCloud("JD keyword inventory", model.keywords.jd, "verified"),
      rawDisclosure("Raw job analysis", model.reports.jd.raw)
    );

    const evidencePanel = panelElements[2].insightsContent;
    const evidenceStats = node(
      "div",
      "insight-fact-grid insights-evidence-stats wide"
    );
    [
      [model.candidateEvidence.roleCount, "roles"],
      [model.candidateEvidence.projectCount, "projects"],
      [model.candidateEvidence.quantifiedCount, "quantified wins"],
    ].forEach(([count, label]) => {
      const item = node("div", "insight-fact insights-evidence-stat");
      append(item, node("span", "", label), node("strong", "", count));
      evidenceStats.append(item);
    });
    evidencePanel.append(evidenceStats);
    const timeline = node(
      "section",
      "insight-section-card wide"
    );
    append(timeline, node("h5", "", "Work-history evidence"));
    const timelineList = node("ol", "insight-timeline insights-timeline");
    const roles = model.candidateEvidence.workItems.filter(
      (item) => item.depth === 0
    );
    if (roles.length) {
      roles.slice(0, MAX_DISPLAY_ITEMS).forEach((role) => {
        timelineList.append(
          node(
            "li",
            "timeline-item insights-timeline-item",
            role.text
          )
        );
      });
      timeline.append(timelineList);
    } else {
      timeline.append(emptyState("No work-history roles were parsed."));
    }
    evidencePanel.append(timeline);
    const skillsCard = node(
      "section",
      "insight-section-card wide"
    );
    append(skillsCard, node("h5", "", "Verified skill inventory"));
    if (model.candidateEvidence.skills.length) {
      model.candidateEvidence.skills
        .slice(0, MAX_DISPLAY_ITEMS)
        .forEach((skill) => {
          const group = node("div", "insights-skill-group");
          append(
            group,
            node("strong", "", skill.label || "Skills"),
            node("p", "", skill.detail || skill.label)
          );
          skillsCard.append(group);
        });
    } else {
      skillsCard.append(emptyState("No categorized skills were parsed."));
    }
    evidencePanel.append(
      skillsCard,
      listCard(
        "Projects and products",
        model.candidateEvidence.projectItems.map((item) => item.text)
      ),
      listCard("Quantified achievements", model.candidateEvidence.quantified),
      listCard("Education and certifications", model.candidateEvidence.education),
      listCard("Source conflicts and gaps", model.candidateEvidence.conflicts, "is-warning"),
      rawDisclosure("Raw candidate fact inventory", model.reports.candidate.raw)
    );

    const matchPanel = panelElements[3].insightsContent;
    const requirementCard = node(
      "section",
      "insight-section-card wide insights-requirements-card"
    );
    append(requirementCard, node("h5", "", "Requirement-to-evidence map"));
    if (model.requirements.length) {
      const filterRow = node("div", "insights-requirement-summary");
      append(
        filterRow,
        node(
          "span",
          "is-supported",
          `${model.requirements.filter((item) => item.status === "supported").length} mapped`
        ),
        node(
          "span",
          "is-gap",
          `${model.requirements.filter((item) => item.status === "gap").length} gaps`
        )
      );
      requirementCard.append(filterRow);
      const rows = node(
        "div",
        "requirement-list"
      );
      model.requirements.slice(0, MAX_DISPLAY_ITEMS).forEach((requirement) => {
        const row = node(
          "article",
          `requirement-row is-${requirement.status}`
        );
        const copy = node("div", "requirement-copy");
        append(
          row,
          node(
            "span",
            "requirement-status",
            requirement.status === "supported"
              ? "Evidence mapped"
              : requirement.status === "gap"
                ? "Evidence gap"
                : "Needs inspection"
          )
        );
        append(
          copy,
          node("strong", "", requirement.requirement),
          node(
            "span",
            "",
            requirement.priority === "must" ? "Must-have" : "Nice-to-have"
          ),
          node(
            "p",
            "",
            requirement.evidence ||
              "No parseable evidence statement was returned."
          )
        );
        row.append(copy);
        rows.append(row);
      });
      requirementCard.append(rows);
    } else {
      requirementCard.append(
        emptyState(
          "No structured requirement map was returned. Open the raw strategy below."
        )
      );
    }
    matchPanel.append(
      model.strategy.positioning
        ? (() => {
            const card = node(
              "section",
              "insight-section-card wide is-highlight"
            );
            append(
              card,
              node("h5", "", "Positioning strategy"),
              node("p", "", model.strategy.positioning)
            );
            return card;
          })()
        : null,
      requirementCard,
      listCard("Keyword placement plan", model.strategy.keywordPlan),
      listCard("Genuine gaps", model.gaps, "is-warning"),
      chipCloud(
        "Placed evidence-backed keywords",
        model.keywords.placed,
        "verified is-placed"
      ),
      chipCloud(
        "Missing but supported keywords",
        model.keywords.missingSupported,
        "gap is-missing"
      ),
      chipCloud(
        "Protected — do not claim without evidence",
        model.keywords.protected,
        "protected is-protected"
      ),
      rawDisclosure("Raw positioning strategy", model.reports.strategy.raw)
    );

    const reviewPanel = panelElements[4].insightsContent;
    const reviewTone =
      model.status === "approved"
        ? ""
        : hasFabrications
          ? "is-danger"
          : "is-warning";
    const reviewLead = node(
      "section",
      `review-alert wide ${reviewTone}`.trim()
    );
    append(
      reviewLead,
      node(
        "strong",
        "",
        model.review.verified
          ? model.status === "approved"
            ? "Quality gate passed"
            : "Corrections required before applying"
          : "Reviewer result could not be verified"
      ),
      node(
        "p",
        "",
        model.review.verified
          ? `${model.review.findings.length} actionable finding(s) were returned.`
          : "The draft was preserved and no missing reviewer value was treated as zero."
      )
    );
    reviewPanel.append(reviewLead);
    const findings = node(
      "section",
      "insight-section-card wide"
    );
    append(findings, node("h5", "", "Action queue"));
    const findingsBody = node("div", "review-findings");
    const groupDefinitions = [
      ["fabrication", "Unsupported claims", "critical"],
      ["structure", "Structure and format", "warning"],
      ["ats", "ATS improvements", "warning"],
      ["craft", "Writing and craft", "advisory"],
      ["general", "Other reviewer notes", "advisory"],
    ];
    let renderedFinding = false;
    let findingIndex = 0;
    groupDefinitions.forEach(([key, label, severity]) => {
      const values = model.review.groups[key];
      if (!values.length) return;
      renderedFinding = true;
      const group = node(
        "section",
        `review-finding-group severity-${severity}`
      );
      append(group, node("h6", "", `${label} (${values.length})`));
      values.slice(0, MAX_DISPLAY_ITEMS).forEach((finding) => {
        findingIndex += 1;
        const item = node(
          "article",
          `review-finding ${severity === "advisory" ? "craft" : severity}`
        );
        item.dataset.index = String(findingIndex).padStart(2, "0");
        append(
          item,
          node("strong", "", label),
          node("p", "", finding.message)
        );
        group.append(item);
      });
      findingsBody.append(group);
    });
    if (!renderedFinding) {
      findingsBody.append(
        emptyState(
          model.status === "approved"
            ? "No corrections were requested."
            : "No structured reviewer findings were available."
        )
      );
    }
    findings.append(findingsBody);
    reviewPanel.append(
      findings,
      listCard(
        model.review.structureValid === false
          ? "Deterministic structure issues"
          : "Document structure",
        model.review.structureIssues.length
          ? model.review.structureIssues
          : model.review.structureValid === true
            ? ["Passed required document-structure checks."]
            : []
      ),
      rawDisclosure("Raw reviewer response", model.review.raw)
    );

    if (model.maximum.available) {
      const maximumPanel = panelElements[5].insightsContent;
      model.maximum.report.sections.forEach((section) => {
        const card = node(
          "section",
          "insight-section-card insights-maximum-card"
        );
        append(card, node("h5", "", section.title));
        section.paragraphs.forEach((paragraph) =>
          card.append(node("p", "", paragraph))
        );
        const items = section.items.filter((item) => item.kind !== "subheading");
        if (items.length) {
          const list = node("ul", "insights-list");
          items.slice(0, MAX_DISPLAY_ITEMS).forEach((item) =>
            list.append(node("li", "", item.text))
          );
          card.append(list);
        }
        maximumPanel.append(card);
      });
      maximumPanel.append(
        rawDisclosure("Raw maximum-match evidence audit", model.maximum.raw)
      );
    }

    append(workspace, sidebar, panels);
    dashboard.append(workspace);
    root.replaceChildren(dashboard);
    return model;
  }

  return Object.freeze({
    canonicalHeading,
    cleanMarkdownText,
    parseMarkdownReport,
    parseRequirementMap,
    parseReviewFeedback,
    buildInsightsViewModel,
    render,
  });
});
