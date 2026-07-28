const form = document.querySelector("#generator-form");
const fileInput = document.querySelector("#file-input");
const dropzone = document.querySelector("#dropzone");
const fileList = document.querySelector("#file-list");
const resumeText = document.querySelector("#resume-text");
const jobDescription = document.querySelector("#job-description");
const contextCount = document.querySelector("#context-count");
const jdCount = document.querySelector("#jd-count");
const clearButton = document.querySelector("#clear-button");
const generateButton = document.querySelector("#generate-button");
const formError = document.querySelector("#form-error");
const processState = document.querySelector("#process-state");
const stages = [...document.querySelectorAll(".agent-stage")];
const results = document.querySelector("#results");
const engineInputs = [...document.querySelectorAll('input[name="engine"]')];
const engineOptions = [...document.querySelectorAll(".engine-option")];
const providerSettings = document.querySelector("#provider-settings");
const headerEngine = document.querySelector("#header-engine");
const openrouterKey = document.querySelector("#openrouter-key");
const modelName = document.querySelector("#model-name");
const modelOptions = document.querySelector("#model-options");
const modelHelp = document.querySelector("#model-help");
const loadModelsButton = document.querySelector("#load-models");
const recommendModelButton = document.querySelector("#recommend-model");
const modelRecommendation = document.querySelector("#model-recommendation");
const testOpenRouterButton = document.querySelector("#test-openrouter");
const openrouterStatus = document.querySelector("#openrouter-status");
const langsmithEnabled = document.querySelector("#langsmith-enabled");
const langsmithSettings = document.querySelector("#langsmith-settings");
const langsmithKey = document.querySelector("#langsmith-key");
const langsmithProject = document.querySelector("#langsmith-project");
const traceContent = document.querySelector("#trace-content");
const testLangsmithButton = document.querySelector("#test-langsmith");
const langsmithStatus = document.querySelector("#langsmith-status");
const runMetadata = document.querySelector("#run-metadata");
const maximumButton = document.querySelector("#generate-maximum");
const maximumIntro = document.querySelector("#maximum-intro");
const maximumResult = document.querySelector("#maximum-result");
const maximumError = document.querySelector("#maximum-error");
const maximumDownload = document.querySelector("#maximum-download");
const maximumInsightsCard = document.querySelector("#maximum-insights-card");
const maximumGapList = document.querySelector("#gap-list");
const maximumGapCount = document.querySelector("#gap-count");
const maximumGapEmpty = document.querySelector("#gap-empty");
const maximumCostNote = document.querySelector("#maximum-cost-note");
const editMaximumEvidenceButton = document.querySelector(
  "#edit-maximum-evidence"
);

let selectedFiles = [];
let progressTimer = null;
let availableModels = [];
let serverOpenRouterConfigured = false;
let serverLangsmithConfigured = false;
let currentGeneration = null;
let maximumMatchData = null;
let maximumGaps = [];

const selectedEngine = () =>
  document.querySelector('input[name="engine"]:checked').value;

function updateEngineUI() {
  const engine = selectedEngine();
  engineOptions.forEach((option) => {
    option.classList.toggle("active", option.querySelector("input").checked);
  });
  providerSettings.hidden = engine !== "langgraph_openrouter";
  headerEngine.textContent =
    engine === "google_adk" ? "Google ADK pipeline" : "LangGraph + OpenRouter";
}

engineInputs.forEach((input) => input.addEventListener("change", updateEngineUI));
langsmithEnabled.addEventListener("change", () => {
  langsmithSettings.hidden = !langsmithEnabled.checked;
});

document.querySelectorAll(".reveal-button").forEach((button) => {
  button.addEventListener("click", () => {
    const input = document.querySelector(`#${button.dataset.secret}`);
    const reveal = input.type === "password";
    input.type = reveal ? "text" : "password";
    button.textContent = reveal ? "Hide" : "Show";
  });
});

function setConnectionStatus(element, message, type = "") {
  element.textContent = message;
  element.classList.remove("success", "error");
  if (type) element.classList.add(type);
}

async function providerRequest(path, keyHeader, key, serverConfigured = false) {
  if (!key.trim() && !serverConfigured) {
    throw new Error("Add the API key to .env or enter it here first.");
  }
  const headers = {};
  if (key.trim()) headers[keyHeader] = key.trim();
  const response = await fetch(path, {
    headers,
  });
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error("Provider returned an unreadable response.");
  }
  if (!response.ok) throw new Error(data.detail || "Connection failed.");
  return data;
}

function normalizedKey(value, variableName) {
  let key = value.trim();
  if (key.toLowerCase().startsWith("bearer ")) key = key.slice(7).trim();
  const prefix = `${variableName}=`;
  if (key.toUpperCase().startsWith(prefix)) key = key.slice(prefix.length).trim();
  if (
    key.length >= 2 &&
    ((key.startsWith('"') && key.endsWith('"')) ||
      (key.startsWith("'") && key.endsWith("'")))
  ) {
    key = key.slice(1, -1).trim();
  }
  return key;
}

async function loadProviderConfiguration() {
  try {
    const response = await fetch("/api/providers");
    if (!response.ok) return;
    const providers = await response.json();
    const openrouter = providers.find(
      (provider) => provider.id === "langgraph_openrouter"
    );
    serverOpenRouterConfigured = Boolean(openrouter?.configured);
    serverLangsmithConfigured = Boolean(openrouter?.tracing_configured);
    if (serverOpenRouterConfigured) {
      openrouterKey.placeholder = "Configured in .env (optional override)";
      setConnectionStatus(openrouterStatus, "OpenRouter key loaded from .env", "success");
    }
    if (serverLangsmithConfigured) {
      langsmithKey.placeholder = "Configured in .env (optional override)";
      setConnectionStatus(langsmithStatus, "LangSmith key loaded from .env", "success");
    }
    if (openrouter?.langsmith_project) {
      langsmithProject.value = openrouter.langsmith_project;
    }
  } catch {
    // The generation endpoint still returns a precise configuration error.
  }
}

async function loadModels() {
  setConnectionStatus(openrouterStatus, "Loading models…");
  loadModelsButton.disabled = true;
  try {
    availableModels = await providerRequest(
      "/api/providers/openrouter/models?limit=200",
      "X-OpenRouter-Api-Key",
      openrouterKey.value,
      serverOpenRouterConfigured
    );
    modelOptions.innerHTML = "";
    availableModels.forEach((model) => {
      const option = document.createElement("option");
      option.value = model.id;
      option.label = `${model.name}${model.context_length ? ` · ${model.context_length.toLocaleString()} tokens` : ""}`;
      modelOptions.append(option);
    });
    setConnectionStatus(
      openrouterStatus,
      `${availableModels.length} models loaded`,
      "success"
    );
  } catch (error) {
    setConnectionStatus(openrouterStatus, error.message, "error");
    throw error;
  } finally {
    loadModelsButton.disabled = false;
  }
}

loadModelsButton.addEventListener("click", () => loadModels().catch(() => {}));

modelName.addEventListener("change", () => {
  const model = availableModels.find((item) => item.id === modelName.value.trim());
  if (!model) {
    modelHelp.textContent = "Enter any valid OpenRouter model slug.";
    return;
  }
  const capabilities = [];
  if (model.supported_parameters.includes("structured_outputs")) capabilities.push("Structured output");
  if (model.supported_parameters.includes("tools")) capabilities.push("Tools");
  modelHelp.textContent = [
    model.context_length ? `${model.context_length.toLocaleString()} token context` : "",
    ...capabilities,
  ].filter(Boolean).join(" · ");
});

testOpenRouterButton.addEventListener("click", async () => {
  setConnectionStatus(openrouterStatus, "Testing…");
  try {
    const key = normalizedKey(openrouterKey.value, "OPENROUTER_API_KEY");
    openrouterKey.value = key;
    if (key && !key.startsWith("sk-or-")) {
      throw new Error("Use an OpenRouter inference key beginning with sk-or-.");
    }
    const selectedModel = modelName.value.trim();
    if (!selectedModel) {
      throw new Error("Enter or select an OpenRouter model first.");
    }
    const data = await providerRequest(
      `/api/providers/openrouter/validate?model_name=${encodeURIComponent(selectedModel)}`,
      "X-OpenRouter-Api-Key",
      key,
      serverOpenRouterConfigured
    );
    setConnectionStatus(openrouterStatus, data.message, "success");
  } catch (error) {
    setConnectionStatus(openrouterStatus, error.message, "error");
  }
});

testLangsmithButton.addEventListener("click", async () => {
  setConnectionStatus(langsmithStatus, "Testing…");
  try {
    const key = normalizedKey(langsmithKey.value, "LANGSMITH_API_KEY");
    langsmithKey.value = key;
    const data = await providerRequest(
      "/api/providers/langsmith/validate",
      "X-LangSmith-Api-Key",
      key,
      serverLangsmithConfigured
    );
    setConnectionStatus(langsmithStatus, data.message, "success");
  } catch (error) {
    setConnectionStatus(langsmithStatus, error.message, "error");
  }
});

const readableSize = (bytes) => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
};

function renderFiles() {
  fileList.innerHTML = "";
  selectedFiles.forEach((file, index) => {
    const item = document.createElement("div");
    item.className = "file-chip";
    const name = document.createElement("strong");
    name.textContent = file.name;
    const size = document.createElement("span");
    size.textContent = readableSize(file.size);
    const remove = document.createElement("button");
    remove.type = "button";
    remove.setAttribute("aria-label", `Remove ${file.name}`);
    remove.textContent = "×";
    remove.addEventListener("click", () => {
      selectedFiles.splice(index, 1);
      renderFiles();
    });
    item.append(name, size, remove);
    fileList.append(item);
  });
}

function addFiles(files) {
  const allowed = [".pdf", ".docx", ".txt", ".md"];
  const incoming = [...files].filter((file) =>
    allowed.some((extension) => file.name.toLowerCase().endsWith(extension))
  );
  incoming.forEach((file) => {
    const duplicate = selectedFiles.some(
      (current) => current.name === file.name && current.size === file.size
    );
    if (!duplicate) selectedFiles.push(file);
  });
  renderFiles();
}

fileInput.addEventListener("change", () => {
  addFiles(fileInput.files);
  fileInput.value = "";
});

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
});
["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragging");
  });
});
dropzone.addEventListener("drop", (event) => addFiles(event.dataTransfer.files));

resumeText.addEventListener("input", () => {
  contextCount.textContent = `${resumeText.value.length.toLocaleString()} / 50,000`;
});
jobDescription.addEventListener("input", () => {
  jdCount.textContent = `${jobDescription.value.length.toLocaleString()} characters`;
});

function resetStages() {
  clearInterval(progressTimer);
  stages.forEach((stage) => {
    stage.classList.remove("active", "done");
    stage.querySelector(".stage-status").textContent = "Waiting";
  });
  processState.textContent = "Ready when you are";
}

function startProgress() {
  resetStages();
  let current = 0;
  const activate = (index) => {
    stages.forEach((stage, stageIndex) => {
      stage.classList.toggle("active", stageIndex === index);
      if (stageIndex < index) {
        stage.classList.add("done");
        stage.querySelector(".stage-status").textContent = "Done";
      }
    });
    stages[index].querySelector(".stage-status").textContent = "Working";
    processState.textContent = `Estimated progress · agent ${index + 1} of ${stages.length}`;
  };
  activate(current);
  progressTimer = setInterval(() => {
    if (current < stages.length - 1) {
      current += 1;
      activate(current);
    } else {
      processState.textContent = "Estimated progress · finalizing";
    }
  }, 8000);
}

function setRecommendedModel(model) {
  modelName.value = model;
  modelName.dispatchEvent(new Event("change"));
  setConnectionStatus(openrouterStatus, "Model changed — test it before generating.");
}

recommendModelButton.addEventListener("click", () => {
  const uploadBytes = selectedFiles.reduce((total, file) => total + file.size, 0);
  const sourceChars = resumeText.value.length + jobDescription.value.length;
  const largeContext = uploadBytes > 750_000 || sourceChars > 40_000 || selectedFiles.length > 1;
  const reliable = largeContext
    ? "google/gemini-2.5-flash-lite"
    : "openai/gpt-4.1-mini";
  const rationale = largeContext
    ? "Large multi-document context detected. This model has a 1M-token window, low cost, and is a better fit for the pipeline’s repeated long-context calls."
    : "Moderate context detected. This model offers strong professional writing quality and a 200K-token window.";

  modelRecommendation.hidden = false;
  modelRecommendation.innerHTML = "";
  const title = document.createElement("strong");
  title.textContent = `Recommended: ${reliable}`;
  const explanation = document.createElement("small");
  explanation.textContent = rationale;
  const options = document.createElement("div");
  options.className = "recommendation-options";
  [
    [reliable, "Use recommended"],
    ["openrouter/free", "Free · best effort"],
    ["openai/gpt-4.1-mini", "Reliable · 1M context"],
  ].forEach(([model, label]) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.addEventListener("click", () => setRecommendedModel(model));
    options.append(button);
  });
  modelRecommendation.append(title, explanation, options);
});

function finishProgress() {
  clearInterval(progressTimer);
  stages.forEach((stage) => {
    stage.classList.remove("active");
    stage.classList.add("done");
    stage.querySelector(".stage-status").textContent = "Done";
  });
  processState.textContent = "Application complete";
}

function showError(message) {
  formError.textContent = message;
  formError.classList.add("visible");
}

function looksLikeJobDescription(text) {
  const value = text.trim();
  if (value.length < 600) return false;
  const signals = [
    /\b(?:full\s+)?job description\b/i,
    /\bjob requisition(?:\s+id)?\b/i,
    /\bminimum qualifications?\b/i,
    /\bpreferred qualifications?\b/i,
    /^\s*you will\s*:/im,
    /^\s*you have\s*:/im,
    /\bequal employment opportunities?\b/i,
    /\bnot eligible for .{0,40}(?:sponsorship|immigration)\b/i,
    /\bwe(?:'re| are) looking for\b/i,
  ];
  const hits = signals
    .map((pattern, index) => (pattern.test(value) ? index : -1))
    .filter((index) => index >= 0);
  const explicitLabel = hits.includes(0) || hits.includes(1);
  return hits.length >= 3 || (explicitLabel && hits.length >= 2);
}

function scoreText(value) {
  return value === null || value === undefined ? "—" : `${value}%`;
}

function appendInlineMarkdown(element, text) {
  const pattern = /\*\*([^*]+)\*\*/g;
  let cursor = 0;
  let match;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > cursor) {
      element.append(document.createTextNode(text.slice(cursor, match.index)));
    }
    const strong = document.createElement("strong");
    strong.textContent = match[1];
    element.append(strong);
    cursor = pattern.lastIndex;
  }
  if (cursor < text.length) {
    element.append(document.createTextNode(text.slice(cursor)));
  }
}

function renderDocumentPreview(element, markdown) {
  element.innerHTML = "";
  const lines = String(markdown || "").replace(/\r\n?/g, "\n").split("\n");
  let list = null;
  let section = "";
  let sawName = false;

  const endList = () => {
    list = null;
  };
  const appendParagraph = (text, className = "") => {
    const paragraph = document.createElement("p");
    if (className) paragraph.className = className;
    appendInlineMarkdown(paragraph, text);
    element.append(paragraph);
  };

  lines.forEach((rawLine) => {
    const line = rawLine.trim();
    if (!line) {
      endList();
      return;
    }
    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      endList();
      const level = heading[1].length;
      const node = document.createElement(`h${level}`);
      appendInlineMarkdown(node, heading[2]);
      element.append(node);
      if (level === 1) sawName = true;
      if (level === 2) section = heading[2].toLowerCase();
      return;
    }
    const bullet = line.match(/^[-*]\s+(.+)$/);
    if (bullet) {
      if (!list) {
        list = document.createElement("ul");
        element.append(list);
      }
      const item = document.createElement("li");
      appendInlineMarkdown(item, bullet[1]);
      list.append(item);
      return;
    }
    endList();

    if (section === "skills") {
      const categories = line
        .split(/(?=\*\*[^*\n]+:\*\*)/)
        .map((part) => part.trim())
        .filter(Boolean);
      if (categories.length > 1) {
        categories.forEach((category) =>
          appendParagraph(category, "preview-skill")
        );
        return;
      }
    }
    appendParagraph(
      line,
      sawName && !section && element.querySelectorAll("p").length === 0
        ? "preview-contact"
        : section === "skills"
          ? "preview-skill"
          : ""
    );
  });
}

function resetMaximumMatch() {
  maximumMatchData = null;
  maximumGaps = [];
  maximumIntro.hidden = false;
  maximumResult.hidden = true;
  maximumError.textContent = "";
  maximumError.classList.remove("visible");
  maximumButton.disabled = true;
  maximumButton.querySelector("span").textContent = "Review evidence and generate";
  maximumGapList.innerHTML = "";
  maximumGapCount.textContent = "Loading gaps…";
  maximumGapEmpty.hidden = true;
  maximumCostNote.textContent =
    "Evidence is validated before the two specialist agents run.";
  maximumDownload.hidden = true;
  maximumDownload.removeAttribute("href");
  maximumInsightsCard.hidden = true;
  document.querySelector("#maximum-insights").textContent = "";
  renderDocumentPreview(document.querySelector("#maximum-resume-text"), "");
  document.querySelector("#maximum-ats-score").textContent = "—";
  document.querySelector("#maximum-role-score").textContent = "—";
  document.querySelector("#maximum-integrity-score").textContent = "—";
}

function maximumApiError(data, fallback) {
  if (typeof data?.detail === "string") return data.detail;
  if (Array.isArray(data?.detail)) {
    return data.detail
      .map((item) => item.msg || "Invalid evidence field")
      .join(" ");
  }
  return fallback;
}

function updateMaximumEvidenceCount() {
  const checked = maximumGapList.querySelectorAll(
    ".gap-resolution-toggle:checked"
  ).length;
  maximumButton.querySelector("span").textContent = checked
    ? `Validate ${checked} evidence ${checked === 1 ? "item" : "items"} and generate`
    : "Generate with current verified evidence";
  maximumCostNote.textContent = checked
    ? `${checked} user-attested evidence ${checked === 1 ? "item" : "items"} will be checked before the agents run.`
    : "No new evidence selected. Genuine gaps will remain protected.";
}

function gapField(labelText, field, options = {}) {
  const wrapper = document.createElement("div");
  wrapper.className = `gap-field${options.full ? " full" : ""}`;
  const label = document.createElement("label");
  label.textContent = labelText;
  let input;
  if (options.type === "select") {
    input = document.createElement("select");
    [
      ["work_experience", "Work experience"],
      ["product_project", "Product or project"],
    ].forEach(([value, labelValue]) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = labelValue;
      input.append(option);
    });
  } else if (options.type === "textarea") {
    input = document.createElement("textarea");
  } else {
    input = document.createElement("input");
    input.type = options.type || "text";
  }
  input.dataset.field = field;
  if (options.placeholder) input.placeholder = options.placeholder;
  if (options.maxLength) input.maxLength = options.maxLength;
  wrapper.append(label, input);
  return { wrapper, label, input };
}

function renderMaximumGaps() {
  maximumGapList.innerHTML = "";
  maximumGapCount.textContent = `${maximumGaps.length} unresolved ${
    maximumGaps.length === 1 ? "gap" : "gaps"
  }`;
  maximumGapEmpty.hidden = maximumGaps.length !== 0;
  maximumButton.disabled = false;

  maximumGaps.forEach((gap) => {
    const card = document.createElement("section");
    card.className = "gap-card";
    card.dataset.gapId = gap.id;

    const toggleLabel = document.createElement("label");
    toggleLabel.className = "gap-card-toggle";
    const toggle = document.createElement("input");
    toggle.type = "checkbox";
    toggle.className = "gap-resolution-toggle";
    toggle.setAttribute(
      "aria-label",
      `I have evidence for ${gap.skill}`
    );
    const title = document.createElement("span");
    title.className = "gap-card-title";
    const strong = document.createElement("strong");
    strong.textContent = gap.skill;
    const reason = document.createElement("small");
    reason.textContent = gap.reason;
    title.append(strong, reason);
    const origin = document.createElement("span");
    origin.className = "gap-origin";
    origin.textContent = gap.ats_keyword ? "JD keyword" : "Requirement gap";
    toggleLabel.append(toggle, title, origin);

    const fields = document.createElement("div");
    fields.className = "gap-evidence-fields";
    fields.hidden = true;
    const sourceType = gapField("Evidence source", "source_type", {
      type: "select",
    });
    const sourceName = gapField("Employer", "source_name", {
      placeholder: "Company or organization",
      maxLength: 160,
    });
    const contribution = gapField("Role or contribution", "role_or_contribution", {
      placeholder: "Your title or responsibility",
      maxLength: 180,
    });
    const dates = gapField("Dates", "dates", {
      placeholder: "Example: January 2025 – May 2025",
      maxLength: 100,
    });
    const evidence = gapField("What did you personally do with this skill?", "evidence_text", {
      type: "textarea",
      full: true,
      placeholder:
        "Describe the feature, implementation, technical decisions, and your personal contribution.",
      maxLength: 2000,
    });
    const outcome = gapField("Outcome or measurable result (optional)", "outcome", {
      type: "textarea",
      full: true,
      placeholder:
        "Add only a result you can defend, such as performance, users, reliability, or delivery impact.",
      maxLength: 500,
    });
    const reference = gapField("Supporting link (optional)", "reference_url", {
      type: "url",
      full: true,
      placeholder: "https://github.com/... or a product page",
      maxLength: 500,
    });
    const attestation = document.createElement("label");
    attestation.className = "gap-attestation";
    const attestationInput = document.createElement("input");
    attestationInput.type = "checkbox";
    attestationInput.dataset.field = "candidate_attested";
    const attestationText = document.createElement("span");
    attestationText.textContent =
      "I confirm this is accurate, belongs to the named work/product, and can be explained in an interview.";
    attestation.append(attestationInput, attestationText);
    const error = document.createElement("p");
    error.className = "gap-card-error";

    fields.append(
      sourceType.wrapper,
      sourceName.wrapper,
      contribution.wrapper,
      dates.wrapper,
      evidence.wrapper,
      outcome.wrapper,
      reference.wrapper,
      attestation,
      error
    );
    card.append(toggleLabel, fields);
    maximumGapList.append(card);

    toggle.addEventListener("change", () => {
      fields.hidden = !toggle.checked;
      card.classList.toggle("resolving", toggle.checked);
      card.classList.remove("invalid");
      error.textContent = "";
      updateMaximumEvidenceCount();
      if (toggle.checked) sourceName.input.focus();
    });
    sourceType.input.addEventListener("change", () => {
      const work = sourceType.input.value === "work_experience";
      sourceName.label.textContent = work ? "Employer" : "Product or project";
      sourceName.input.placeholder = work
        ? "Company or organization"
        : "Product or project name";
      contribution.label.textContent = work
        ? "Role or title"
        : "Your contribution";
    });
  });
  updateMaximumEvidenceCount();
}

async function loadMaximumGaps() {
  if (!currentGeneration?.session_id) return;
  maximumButton.disabled = true;
  maximumGapCount.textContent = "Loading gaps…";
  const response = await fetch(
    `/api/resume/maximum-match/${currentGeneration.session_id}/gaps`
  );
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error("The gap checklist returned an unreadable response.");
  }
  if (!response.ok) {
    throw new Error(
      maximumApiError(data, "The gap checklist could not be loaded.")
    );
  }
  maximumGaps = data.gaps || [];
  renderMaximumGaps();
}

function collectMaximumEvidence() {
  const collected = [];
  let firstInvalid = null;
  maximumGapList.querySelectorAll(".gap-card").forEach((card) => {
    card.classList.remove("invalid");
    const error = card.querySelector(".gap-card-error");
    error.textContent = "";
    const enabled = card.querySelector(".gap-resolution-toggle").checked;
    if (!enabled) return;
    const read = (field) =>
      card.querySelector(`[data-field="${field}"]`);
    const sourceName = read("source_name").value.trim();
    const contribution = read("role_or_contribution").value.trim();
    const dates = read("dates").value.trim();
    const evidenceText = read("evidence_text").value.trim();
    const outcome = read("outcome").value.trim();
    const referenceUrl = read("reference_url").value.trim();
    const attested = read("candidate_attested").checked;
    let message = "";
    if (!sourceName || !contribution || !dates) {
      message = "Add the employer/product, your role or contribution, and dates.";
    } else if (evidenceText.length < 40) {
      message =
        "Describe what you personally did in at least 40 characters.";
    } else if (referenceUrl && !/^https?:\/\//i.test(referenceUrl)) {
      message = "The supporting link must start with http:// or https://.";
    } else if (!attested) {
      message = "Confirm that the evidence is accurate before continuing.";
    }
    if (message) {
      card.classList.add("invalid");
      error.textContent = message;
      firstInvalid ||= card;
      return;
    }
    collected.push({
      gap_id: card.dataset.gapId,
      source_type: read("source_type").value,
      source_name: sourceName,
      role_or_contribution: contribution,
      dates,
      evidence_text: evidenceText,
      outcome,
      reference_url: referenceUrl,
      candidate_attested: attested,
    });
  });
  if (firstInvalid) {
    firstInvalid.scrollIntoView({ behavior: "smooth", block: "center" });
    throw new Error("Complete the highlighted evidence card before generating.");
  }
  return collected;
}

function addRunChip(label, value, id = "") {
  const chip = document.createElement("span");
  chip.className = "run-chip";
  if (id) chip.id = id;
  const strong = document.createElement("strong");
  strong.textContent = `${label}: `;
  chip.append(strong, document.createTextNode(value));
  runMetadata.append(chip);
}

function populateMaximumMatch(data) {
  maximumMatchData = data;
  maximumIntro.hidden = true;
  maximumResult.hidden = false;
  renderDocumentPreview(
    document.querySelector("#maximum-resume-text"),
    data.resume_markdown
  );
  document.querySelector("#maximum-ats-score").textContent =
    scoreText(data.scores.supported_ats_coverage);
  document.querySelector("#maximum-role-score").textContent =
    scoreText(data.scores.overall_requirement_match);
  document.querySelector("#maximum-integrity-score").textContent =
    scoreText(data.scores.evidence_integrity);
  document.querySelector("#maximum-status-title").textContent = data.approved
    ? "Evidence audit passed"
    : "Ready for your evidence review";
  document.querySelector("#maximum-status-copy").textContent = data.approved
    ? "Every final claim passed the agent audit and supported-keyword coverage target."
    : data.scores.score_status === "valid"
      ? "The resume is preserved, but the audit found items to inspect before applying."
      : "The resume is preserved. Some reviewer scores were unavailable, so inspect the evidence audit.";
  if (data.evidence_count) {
    document.querySelector("#maximum-status-copy").textContent +=
      ` ${data.evidence_count} user-attested gap ${
        data.evidence_count === 1 ? "item was" : "items were"
      } incorporated and audited.`;
  }
  maximumDownload.href = data.resume_pdf_url;
  maximumDownload.download = data.resume_filename;
  maximumDownload.hidden = false;
  document.querySelector("#maximum-insights").textContent = data.insights_markdown;
  maximumInsightsCard.hidden = false;

  document.querySelector("#maximum-run-chip")?.remove();
  const tokens = data.usage?.total_tokens
    ? `${data.usage.total_tokens.toLocaleString()} additional`
    : "Tracked by selected engine";
  addRunChip("Maximum-match tokens", tokens, "maximum-run-chip");
}

function editMaximumEvidence() {
  maximumMatchData = null;
  maximumIntro.hidden = false;
  maximumResult.hidden = true;
  maximumDownload.hidden = true;
  maximumDownload.removeAttribute("href");
  maximumInsightsCard.hidden = true;
  document.querySelector("#maximum-run-chip")?.remove();
  maximumError.textContent = "";
  maximumError.classList.remove("visible");
  maximumButton.disabled = false;
  updateMaximumEvidenceCount();
  maximumIntro.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function generateMaximumMatch() {
  if (!currentGeneration?.maximum_match_generate_url || maximumMatchData) return;
  maximumError.textContent = "";
  maximumError.classList.remove("visible");
  let evidence;
  try {
    evidence = collectMaximumEvidence();
  } catch (error) {
    maximumError.textContent = error.message;
    maximumError.classList.add("visible");
    return;
  }
  maximumButton.disabled = true;
  maximumButton.querySelector("span").textContent = "Validating your evidence…";

  const headers = { "Content-Type": "application/json" };
  if (currentGeneration.engine === "langgraph_openrouter") {
    const openrouterBrowserKey = normalizedKey(
      openrouterKey.value,
      "OPENROUTER_API_KEY"
    );
    if (openrouterBrowserKey) {
      headers["X-OpenRouter-Api-Key"] = openrouterBrowserKey;
    }
    if (currentGeneration.langsmith_enabled) {
      const langsmithBrowserKey = normalizedKey(
        langsmithKey.value,
        "LANGSMITH_API_KEY"
      );
      if (langsmithBrowserKey) {
        headers["X-LangSmith-Api-Key"] = langsmithBrowserKey;
      }
    }
  }

  try {
    const validationResponse = await fetch(
      `/api/resume/maximum-match/${currentGeneration.session_id}/evidence/validate`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ evidence }),
      }
    );
    let validationData;
    try {
      validationData = await validationResponse.json();
    } catch {
      throw new Error("The evidence validator returned an unreadable response.");
    }
    if (!validationResponse.ok) {
      throw new Error(
        maximumApiError(validationData, "The evidence could not be validated.")
      );
    }
    maximumCostNote.textContent = validationData.message;
    maximumButton.querySelector("span").textContent =
      "Maximum-match agents are working…";

    const response = await fetch(currentGeneration.maximum_match_generate_url, {
      method: "POST",
      headers,
      body: JSON.stringify({ evidence }),
    });
    let data;
    try {
      data = await response.json();
    } catch {
      throw new Error("The maximum-match service returned an unreadable response.");
    }
    if (!response.ok) {
      throw new Error(
        maximumApiError(
          data,
          "The maximum-match resume could not be generated."
        )
      );
    }
    populateMaximumMatch(data);
  } catch (error) {
    maximumError.textContent = error.message;
    maximumError.classList.add("visible");
    maximumButton.disabled = false;
    updateMaximumEvidenceCount();
  }
}

maximumButton.addEventListener("click", generateMaximumMatch);
editMaximumEvidenceButton.addEventListener("click", editMaximumEvidence);

function populateResults(data) {
  currentGeneration = data;
  resetMaximumMatch();
  activateResultTab(
    document.querySelector('.result-tab[data-target="resume-output"]')
  );
  renderDocumentPreview(
    document.querySelector("#resume-output"),
    data.resume_markdown
  );
  renderDocumentPreview(
    document.querySelector("#cover-output"),
    data.cover_letter_markdown ||
      data.warnings?.[0] ||
      "Cover letter was not generated."
  );
  document.querySelector("#jd-analysis").textContent = data.artifacts.jd_analysis;
  document.querySelector("#candidate-profile").textContent = data.artifacts.candidate_profile;
  document.querySelector("#match-strategy").textContent = data.artifacts.match_strategy;
  document.querySelector("#review-feedback").textContent = data.artifacts.review_feedback;
  document.querySelector("#approval-copy").textContent = data.approved
    ? "The quality agent approved this evidence-grounded draft."
    : data.review_valid
      ? "The draft reached its review limit. Please inspect the feedback before applying."
      : "The draft was preserved, but the reviewer score was unavailable—not zero.";
  document.querySelector("#resume-download").href = data.resume_pdf_url;
  document.querySelector("#resume-download").download = data.resume_filename;
  document.querySelector("#cover-download").href = data.cover_letter_pdf_url;
  document.querySelector("#cover-download").download = data.cover_letter_filename;
  document.querySelector("#cover-download").hidden = !data.cover_letter_pdf_url;
  const metadata = [
    ["Engine", data.engine === "google_adk" ? "Google ADK" : "LangGraph"],
    ["Model", data.model_name],
    data.review_valid && data.review_score !== null && data.review_score !== undefined
      ? ["Review", `${data.review_score}/100`]
      : ["Review", "Unavailable"],
    data.scores?.supported_ats_coverage !== null &&
      data.scores?.supported_ats_coverage !== undefined
      ? ["Supported ATS coverage", `${data.scores.supported_ats_coverage}%`]
      : null,
    data.revision_count ? ["Draft passes", data.revision_count] : null,
    data.usage?.total_tokens ? ["Tokens", data.usage.total_tokens.toLocaleString()] : null,
    data.langsmith_enabled ? ["LangSmith", data.langsmith_project] : null,
    data.warnings?.length ? ["Warning", data.warnings[0]] : null,
  ].filter(Boolean);
  runMetadata.innerHTML = "";
  metadata.forEach(([label, value]) => {
    addRunChip(label, value);
  });
  results.hidden = false;
  results.scrollIntoView({ behavior: "smooth", block: "start" });
  loadMaximumGaps().catch((error) => {
    maximumGapCount.textContent = "Gap list unavailable";
    maximumError.textContent = error.message;
    maximumError.classList.add("visible");
    maximumButton.disabled = false;
  });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  formError.classList.remove("visible");

  const jd = jobDescription.value.trim();
  const context = resumeText.value.trim();
  const engine = selectedEngine();
  if (!jd) {
    showError("Paste the job description before generating.");
    jobDescription.focus();
    return;
  }
  if (!selectedFiles.length && !context) {
    showError("Add at least one source document or paste candidate context.");
    return;
  }
  if (context && looksLikeJobDescription(context)) {
    showError(
      "Candidate context appears to contain another job description. Keep the target posting in Job Description, and use Candidate context only for your work, projects, skills, education, and achievements."
    );
    resumeText.focus();
    return;
  }
  if (engine === "langgraph_openrouter") {
    const key = normalizedKey(openrouterKey.value, "OPENROUTER_API_KEY");
    openrouterKey.value = key;
    if (!key && !serverOpenRouterConfigured) {
      showError("Add OPENROUTER to .env or enter your OpenRouter API key.");
      openrouterKey.focus();
      return;
    }
    if (!modelName.value.trim()) {
      showError("Enter or select an OpenRouter model.");
      modelName.focus();
      return;
    }
    if (
      langsmithEnabled.checked &&
      !langsmithKey.value.trim() &&
      !serverLangsmithConfigured
    ) {
      showError("Add LANGSMITH_API_KEY to .env, enter it here, or disable tracing.");
      langsmithKey.focus();
      return;
    }
  }

  const payload = new FormData();
  payload.append("job_description", jd);
  payload.append("engine", engine);
  if (engine === "langgraph_openrouter") {
    payload.append("model_name", modelName.value.trim());
    payload.append("langsmith_enabled", String(langsmithEnabled.checked));
    payload.append("langsmith_project", langsmithProject.value.trim() || "rolefit-resume-agent");
    payload.append("trace_content", String(traceContent.checked));
  }
  if (context) payload.append("resume_text", context);
  selectedFiles.forEach((file) => payload.append("files", file));

  generateButton.disabled = true;
  generateButton.querySelector("span").textContent = "Agents are working…";
  results.hidden = true;
  startProgress();

  try {
    const headers = {};
    if (engine === "langgraph_openrouter") {
      const openrouterBrowserKey = normalizedKey(
        openrouterKey.value,
        "OPENROUTER_API_KEY"
      );
      if (openrouterBrowserKey) {
        headers["X-OpenRouter-Api-Key"] = openrouterBrowserKey;
      }
      if (langsmithEnabled.checked) {
        const langsmithBrowserKey = normalizedKey(
          langsmithKey.value,
          "LANGSMITH_API_KEY"
        );
        if (langsmithBrowserKey) {
          headers["X-LangSmith-Api-Key"] = langsmithBrowserKey;
        }
      }
    }
    const response = await fetch("/api/resume/generate", {
      method: "POST",
      headers,
      body: payload,
    });
    let data;
    try {
      data = await response.json();
    } catch {
      throw new Error("The server returned an unreadable response.");
    }
    if (!response.ok) {
      throw new Error(data.detail || "The application could not be generated.");
    }
    finishProgress();
    populateResults(data);
  } catch (error) {
    resetStages();
    processState.textContent = "Generation stopped";
    showError(error.message);
  } finally {
    generateButton.disabled = false;
    generateButton.querySelector("span").textContent = "Generate tailored application";
  }
});

clearButton.addEventListener("click", () => {
  selectedFiles = [];
  form.reset();
  renderFiles();
  contextCount.textContent = "0 / 50,000";
  jdCount.textContent = "0 characters";
  formError.classList.remove("visible");
  results.hidden = true;
  currentGeneration = null;
  resetMaximumMatch();
  resetStages();
  updateEngineUI();
});

function activateResultTab(tab) {
  document.querySelectorAll(".result-tab").forEach((item) => {
    const active = item === tab;
    item.classList.toggle("active", active);
    item.setAttribute("aria-selected", String(active));
  });
  document.querySelectorAll(".result-view").forEach((item) => {
    item.classList.toggle("active", item.id === tab.dataset.target);
  });
}

document.querySelectorAll(".result-tab").forEach((tab) => {
  tab.addEventListener("click", () => activateResultTab(tab));
});

updateEngineUI();
loadProviderConfiguration();
