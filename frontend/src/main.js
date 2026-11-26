import { absoluteUrl, analyzeImage, deleteDetection, getApiBase, getCategories, getDetection, getDetections, getHealth, saveApiBase } from "./api.js";
import { addBatchFiles, processBatch, removeBatchItem } from "./batch.mjs";

const FALLBACK_CATEGORIES = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"];
const NAV = [
  ["dashboard", "Dashboard", "grid"], ["inspect", "Inspect Product", "scan"],
  ["history", "Inspection History", "clock"], ["categories", "Product Categories", "boxes"], ["status", "System Status", "pulse"],
];
const PAGE_META = {
  dashboard: ["Overview", "Inspection dashboard"], inspect: ["New analysis", "Inspect product"],
  history: ["Records", "Inspection history"], categories: ["Catalog", "Product categories"], status: ["Connectivity", "System status"],
};
const ICONS = {
  grid: '<rect x="3" y="3" width="7" height="7" rx="1.3"/><rect x="14" y="3" width="7" height="7" rx="1.3"/><rect x="3" y="14" width="7" height="7" rx="1.3"/><rect x="14" y="14" width="7" height="7" rx="1.3"/>',
  menu: '<path d="M4 6h16M4 12h16M4 18h16"/>',
  scan: '<path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3"/><path d="M7 12h10M12 7v10"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3.5 2"/>',
  boxes: '<path d="m12 3 9 5-9 5-9-5 9-5Z"/><path d="m3 12 9 5 9-5M3 16l9 5 9-5M12 13v8"/>',
  pulse: '<path d="M3 12h4l2.5-7 5 14 2.5-7h4"/>',
  search: '<circle cx="10.8" cy="10.8" r="6.8"/><path d="m16 16 4.5 4.5"/>',
  arrow: '<path d="M5 12h14m-6-6 6 6-6 6"/>',
  upload: '<path d="M12 16V4m-5 5 5-5 5 5"/><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/>',
  image: '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="8.5" cy="9" r="1.5"/><path d="m21 15-5-5L5 20"/>',
  trash: '<path d="M4 7h16m-10 4v6m4-6v6M5 7l1 14h12l1-14M9 7V4h6v3"/>',
  close: '<path d="m6 6 12 12M18 6 6 18"/>',
  refresh: '<path d="M20 7v5h-5M4 17v-5h5"/><path d="M5.5 9a7 7 0 0 1 11.9-2L20 12M4 12l2.6 5a7 7 0 0 0 11.9-2"/>',
  check: '<path d="m5 12 4 4L19 6"/>',
  warning: '<path d="M10.3 4.3 2.8 17.2A2 2 0 0 0 4.5 20h15a2 2 0 0 0 1.7-2.8L13.7 4.3a2 2 0 0 0-3.4 0Z"/><path d="M12 9v4m0 3h.01"/>',
  chevron: '<path d="m9 18 6-6-6-6"/>',
  settings: '<circle cx="12" cy="12" r="3"/><path d="m19.4 15 .1.1 1.1.9-1.3 2.2-1.4-.5a7.9 7.9 0 0 1-1.6.9l-.3 1.5h-2.6l-.3-1.5a7.9 7.9 0 0 1-1.6-.9l-1.4.5-1.3-2.2 1.1-.9a7.4 7.4 0 0 1 0-1.8l-1.1-.9 1.3-2.2 1.4.5a7.9 7.9 0 0 1 1.6-.9l.3-1.5h2.6l.3 1.5a7.9 7.9 0 0 1 1.6.9l1.4-.5 1.3 2.2-1.1.9a7.4 7.4 0 0 1 0 1.8Z"/>',
};

const state = {
  route: location.hash.replace(/^#\/?/, "") || "dashboard", categories: FALLBACK_CATEGORIES, categorySource: "fallback",
  file: null, previewUrl: "", selectedFiles: [], rejectedFiles: [], batchRunId: 0, selectedCategory: "", analysis: null, submitting: false,
  historyPage: 1, history: null, historyError: "", filterCategory: "", minScore: "", maxScore: "", sortOrder: "desc",
  health: null, healthLoading: false, modal: null, modalOpener: null, mobileMenuOpen: false, desktopSidebarCollapsed: false, apiNotice: "",
};
const app = document.querySelector("#app");

function icon(name, size = 20) { return `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name] || ICONS.grid}</svg>`; }
function esc(value) { return String(value ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
function titleCase(value) { return String(value || "").replaceAll("_", " ").replace(/\b\w/g, c => c.toUpperCase()); }
function score(value) { return Number.isFinite(Number(value)) ? Number(value).toFixed(4) : "—"; }
function dateTime(value) { const date = new Date(value); return Number.isNaN(date.getTime()) ? "—" : new Intl.DateTimeFormat("en", { dateStyle: "medium", timeStyle: "short" }).format(date); }
function activeMeta() { return PAGE_META[state.route] || PAGE_META.dashboard; }
function navLink([route, label, glyph]) { return `<a class="nav-link ${state.route === route ? "active" : ""}" href="#/${route}" aria-label="${label}" title="${label}" ${state.route === route ? 'aria-current="page"' : ""}><span class="nav-icon">${icon(glyph, 18)}</span><span>${label}</span>${state.route === route ? '<span class="nav-active-dot"></span>' : ""}</a>`; }
function isMobileViewport() { return window.matchMedia?.("(max-width: 760px)").matches ?? false; }
function syncNavigationToggle() {
  const mobile = isMobileViewport();
  const expanded = mobile ? state.mobileMenuOpen : !state.desktopSidebarCollapsed;
  const toggle = document.querySelector("#menu-toggle");
  toggle?.setAttribute?.("aria-expanded", String(expanded));
  toggle?.setAttribute?.("aria-label", mobile ? (expanded ? "Close navigation menu" : "Open navigation menu") : (expanded ? "Collapse navigation sidebar" : "Expand navigation sidebar"));
}

function shell(content) {
  const [eyebrow, heading] = activeMeta();
  app.innerHTML = `<div class="app-shell${state.mobileMenuOpen ? " menu-open" : ""}${state.desktopSidebarCollapsed ? " sidebar-collapsed" : ""}">
    <aside class="sidebar${state.mobileMenuOpen ? " sidebar-open" : ""}" id="sidebar" aria-label="Site navigation">
      <button class="mobile-nav-close icon-button" type="button" data-action="close-navigation" aria-label="Close navigation menu">${icon("close")}</button>
      <a href="#/dashboard" class="brand"><span class="brand-mark">I</span><span class="brand-copy"><b>INSPECTRA</b><small>VISION SYSTEMS</small></span></a>
      <div class="workspace-label">WORKSPACE</div><nav class="primary-nav" aria-label="Main navigation">${NAV.map(navLink).join("")}</nav>
      <div class="sidebar-bottom"><div class="sidebar-model"><span class="model-orbit">${icon("pulse", 16)}</span><div><b>Vision engine</b><small>DINOv2 · Exact kNN</small></div><span class="tiny-dot" aria-hidden="true"></span></div><div class="sidebar-foot">Industrial inspection<br/>Continuous anomaly scoring</div></div>
    </aside><button class="nav-backdrop" type="button" data-action="close-navigation" aria-label="Close navigation menu" tabindex="-1" aria-hidden="true"></button>
    <div class="main-column"><header class="topbar"><button class="mobile-menu icon-button" id="menu-toggle" data-action="menu-toggle" type="button" aria-label="${isMobileViewport() ? (state.mobileMenuOpen ? "Close navigation menu" : "Open navigation menu") : (state.desktopSidebarCollapsed ? "Expand navigation sidebar" : "Collapse navigation sidebar")}" aria-expanded="${isMobileViewport() ? state.mobileMenuOpen : !state.desktopSidebarCollapsed}" aria-controls="sidebar">${icon("menu")}</button><div class="breadcrumb"><span>${eyebrow}</span>${icon("chevron", 14)}<b>${heading}</b></div><div class="topbar-right"><div class="api-indicator ${state.health?.online ? "is-online" : state.health ? "is-offline" : ""}" role="status" aria-live="polite" aria-atomic="true"><span class="status-dot" aria-hidden="true"></span><span class="status-label">${healthLabel(state.health)}</span></div><button class="avatar" title="Industrial Vision workspace">IV</button></div></header>
    <main class="page-content" id="page-content">${content}</main><footer class="app-footer"><span>INDUSTRIAL VISION ANOMALY DETECTION</span><span>Continuous scoring · No binary threshold configured</span></footer></div>
    <div id="modal-root"></div>
  </div>`;
}

function setMobileMenu(open, restoreFocus = true) {
  state.mobileMenuOpen = open;
  document.querySelector(".app-shell")?.classList.toggle("menu-open", open);
  document.querySelector("#sidebar")?.classList.toggle("sidebar-open", open);
  const toggle = document.querySelector("#menu-toggle");
  syncNavigationToggle();
  if (open) document.querySelector("#sidebar .primary-nav a")?.focus();
  else if (restoreFocus) toggle?.focus();
}

function setDesktopSidebarCollapsed(collapsed) {
  state.desktopSidebarCollapsed = collapsed;
  document.querySelector(".app-shell")?.classList.toggle("sidebar-collapsed", collapsed);
  syncNavigationToggle();
}

function categoryOptions(selected = "", includeAll = false) {
  return `${includeAll ? '<option value="">All categories</option>' : '<option value="">Select a category</option>'}${state.categories.map(c => `<option value="${esc(c)}" ${c === selected ? "selected" : ""}>${esc(titleCase(c))}</option>`).join("")}`;
}
function errorPanel(message, retry = "") { return `<div class="inline-alert error-alert" role="alert">${icon("warning", 18)}<span>${esc(message)}</span>${retry ? `<button class="text-button" data-action="${retry}">Try again</button>` : ""}</div>`; }
function loadingBlock(label = "Loading live data…") { return `<div class="loading-block"><span class="spinner"></span><span>${esc(label)}</span></div>`; }
function emptyBlock(title, text, action = "") { return `<div class="empty-state"><span class="empty-icon">${icon("image", 22)}</span><b>${esc(title)}</b><p>${esc(text)}</p>${action}</div>`; }
function statCard(label, value, helper, glyph, statusBadge = "") { return `<article class="stat-card"><div class="stat-top"><span>${esc(label)}</span><span class="stat-icon">${icon(glyph, 18)}</span></div><div class="stat-value">${esc(value)}</div><div class="stat-helper"><span>${esc(helper)}</span>${statusBadge}</div></article>`; }
function healthBadge(health) {
  if (!health) return '<span class="badge badge-neutral">Not checked</span>';
  if (!health.online) return `<span class="badge ${health.errorType === "configuration" ? "badge-danger" : "badge-warning"}">${health.errorType === "configuration" ? "Invalid API URL" : health.errorType === "http" ? "Health check failed" : "Connection issue"}</span>`;
  const status = health.data?.status;
  return status === "ok" ? '<span class="badge badge-success">Online</span>' : '<span class="badge badge-warning">Degraded</span>';
}
function healthLabel(health) {
  if (!health) return "Checking API";
  if (health.online) return "API reachable";
  if (health.errorType === "configuration") return "Invalid API URL";
  return health.errorType === "http" ? "API health check failed" : "API connection issue";
}

async function readHealth() {
  try { state.health = { online: true, data: await getHealth() }; }
  catch (error) {
    const degraded = error.status === 503 && error.body?.status === "degraded" && typeof error.body.service === "string" && error.body.database === "disconnected";
    state.health = degraded
      ? { online: true, data: error.body, error: error.message }
      : { online: false, error: error.message, errorType: error.kind || (error.status ? "http" : "network") };
  }
}
async function readCategories() {
  try { const data = await getCategories(); state.categories = Array.isArray(data?.categories) ? data.categories : FALLBACK_CATEGORIES; state.categorySource = Array.isArray(data?.categories) ? "api" : "fallback"; }
  catch { state.categories = FALLBACK_CATEGORIES; state.categorySource = "fallback"; }
}

function detectionRows(items, compact = false, showFullIdentifier = true) {
  if (!items?.length) return emptyBlock("No inspections yet", "Your saved inspection records will appear here after the first analysis.", '<a class="button button-primary button-small" href="#/inspect">Inspect a product</a>');
  return `<div class="table-wrap"><table><thead><tr><th>INSPECTION</th><th>PRODUCT CATEGORY</th><th>ANOMALY SCORE</th><th>PROCESS TIME</th><th>DATE</th><th></th></tr></thead><tbody>${items.map(item => `<tr>
    <td><div class="inspection-id"><span class="id-mark">${icon("scan", 15)}</span><span><b>${esc(item.prediction_id.slice(0, 8).toUpperCase())}</b>${showFullIdentifier ? `<small>${esc(item.prediction_id)}</small>` : ""}</span></div></td>
    <td><span class="category-chip">${esc(titleCase(item.category))}</span></td><td><span class="score-number">${score(item.anomaly_score)}</span></td><td>${Number(item.processing_time_ms || 0).toLocaleString()} <span class="muted">ms</span></td><td>${esc(dateTime(item.created_at))}</td>
    <td><button class="row-action" data-action="view-detection" data-id="${esc(item.prediction_id)}" aria-label="${showFullIdentifier ? "View inspection details" : `View inspection ${esc(item.prediction_id.slice(0, 8))} details`}">${icon("chevron", 17)}</button></td>
  </tr>`).join("")}</tbody></table></div>`;
}

async function renderDashboard(container) {
  container.innerHTML = `<section class="welcome-row"><div><div class="eyebrow"><span class="eyebrow-line"></span>INSPECTION OVERVIEW</div><h1>Inspection activity</h1><p>Review saved inspection results or begin a new product inspection.</p></div><a class="button button-primary" href="#/inspect">${icon("scan", 17)} Inspect a product ${icon("arrow", 16)}</a></section>
    <section class="stats-grid" id="dashboard-stats">${loadingBlock("Fetching live metrics…")}</section>
    <section class="panel recent-panel"><div class="panel-heading"><div><span class="eyebrow">LATEST ACTIVITY</span><h2>Recent inspections</h2></div><a class="text-link" href="#/history">View all inspections ${icon("arrow", 15)}</a></div><div id="dashboard-history">${loadingBlock()}</div></section>
    <section class="dashboard-bottom"><article class="engine-card"><div class="engine-copy"><span class="eyebrow">SCORING ENGINE</span><h3>Frozen DINOv2 + exact kNN</h3><p>Patch-level distances are aggregated into a continuous anomaly score. Higher scores indicate greater deviation from the normal feature bank.</p><small>Model health is not reported by the API.</small></div></article><article class="quick-card"><span class="quick-icon">${icon("upload", 20)}</span><b>Ready for an inspection?</b><p>Upload a product image to receive its score and visual anomaly map.</p><a href="#/inspect" class="text-link">Start inspection ${icon("arrow", 15)}</a></article></section>`;
  const [healthResult, categoriesResult, historyResult] = await Promise.allSettled([readHealth(), getCategories(), getDetections({ page: 1, pageSize: 5 })]);
  if (healthResult.status === "fulfilled") {} else state.health = { online: false, error: "Health request failed." };
  if (categoriesResult.status === "fulfilled" && Array.isArray(categoriesResult.value?.categories)) { state.categories = categoriesResult.value.categories; state.categorySource = "api"; }
  const stats = document.querySelector("#dashboard-stats");
  if (stats) {
    const history = historyResult.status === "fulfilled" ? historyResult.value : null;
    stats.innerHTML = `<div class="stat-grid-inner">${statCard("TOTAL INSPECTIONS", history ? Number(history.total).toLocaleString() : "—", history ? "Saved inspection records" : "History unavailable", "scan")}${statCard("PRODUCT CATEGORIES", state.categories.length, state.categorySource === "api" ? "Returned by categories API" : "Fallback catalog · API unavailable", "boxes")}${statCard("RECENT ACTIVITY", history ? history.items.length : "—", history ? "Records on this page" : "History unavailable", "clock")}${statCard("BACKEND STATUS", state.health?.online ? state.health.data?.status || "Reachable" : state.health?.errorType === "configuration" ? "Invalid API URL" : state.health?.errorType === "http" ? "API error" : "Connection issue", state.health?.data?.database ? `MongoDB ${state.health.data.database}` : "Health detail unavailable", "pulse", healthBadge(state.health))}</div>`;
  }
  const recent = document.querySelector("#dashboard-history");
  if (recent) recent.innerHTML = historyResult.status === "fulfilled" ? detectionRows(historyResult.value.items, true) : errorPanel(historyResult.reason.message || "Inspection history is unavailable.");
  renderShellStatus();
}

function renderShellStatus() {
  const el = document.querySelector(".api-indicator");
  if (!el || !state.health) return;
  el.className = `api-indicator ${state.health.online ? "is-online" : "is-offline"}`;
  el.innerHTML = `<span class="status-dot" aria-hidden="true"></span><span class="status-label">${healthLabel(state.health)}</span>`;
}

async function renderInspect(container) {
  container.innerHTML = `<section class="page-intro"><div><div class="eyebrow"><span class="eyebrow-line"></span>IMAGE ANALYSIS</div><h1>Inspect a product</h1><p>Submit a product image to the frozen anomaly detection pipeline.</p></div><div class="protocol-note">${icon("pulse", 16)} Continuous score · No binary decision threshold</div></section>
  <div class="inspect-layout"><section class="panel inspect-form-panel"><div class="panel-heading compact-heading"><div><span class="eyebrow">NEW INSPECTION</span><h2>Analysis input</h2></div></div>
    <label class="field-label" for="category-select">Product category <span class="required">*</span></label><select class="form-control" id="category-select"><option value="">Loading categories…</option></select><p class="field-hint" id="category-hint">Choose the matching MVTec AD product category.</p>
    <div class="field-label upload-label">Product image(s) <span class="required">*</span></div><div class="dropzone" id="dropzone"><input type="file" id="file-input" accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp" multiple aria-label="Choose product images" hidden /><span class="upload-illustration">${icon("upload", 23)}</span><b>Drop images here</b><span class="drop-copy">or <button class="browse-button" type="button" data-action="browse-file">browse files</button></span><small>JPG, PNG or WEBP · up to 10 MB each · maximum 10 images</small></div><div id="file-feedback"></div>
    <button class="button button-primary analyze-button" id="analyze-button" data-action="analyze" disabled>${icon("scan", 17)} Analyze image ${icon("arrow", 16)}</button><p class="privacy-note">Each image is submitted separately. Successful inspections are saved as individual history records.</p>
  </section><section class="preview-column"><div class="panel preview-panel"><div class="panel-heading compact-heading"><div><span class="eyebrow">IMAGE PREVIEW</span><h2>Input image</h2></div><span class="preview-tag">BEFORE ANALYSIS</span></div><div class="preview-stage" id="preview-stage">${emptyBlock("Image preview", "Choose an image file to preview it here before analysis.")}</div><div class="preview-caption"><span>${icon("image", 15)} Original image</span><span id="preview-dimensions">—</span></div></div><div class="score-explainer"><div><b>Score interpretation</b><p>The anomaly score is continuous. A higher score means greater deviation from the category’s normal feature bank. No classification threshold is configured.</p></div></div></section></div>
  <section id="analysis-result" class="analysis-result" aria-live="polite"></section>`;
  const select = document.querySelector("#category-select");
  try { const data = await getCategories(); if (Array.isArray(data?.categories)) { state.categories = data.categories; state.categorySource = "api"; } }
  catch { state.categories = FALLBACK_CATEGORIES; state.categorySource = "fallback"; }
  if (select) { select.innerHTML = categoryOptions(state.selectedCategory); document.querySelector("#category-hint").textContent = state.categorySource === "api" ? "Categories loaded from the backend." : "Backend categories are unavailable; showing the MVTec fallback list."; }
  renderSelectedFiles();
  if (state.selectedFiles.length > 1 && state.selectedFiles.some(item => item.status !== "waiting")) renderBatchResults();
}

function renderSelectedFiles() {
  const stage = document.querySelector("#preview-stage");
  const feedback = document.querySelector("#file-feedback");
  const dimensions = document.querySelector("#preview-dimensions");
  const button = document.querySelector("#analyze-button");
  const input = document.querySelector("#file-input");
  const browse = document.querySelector('[data-action="browse-file"]');
  const category = document.querySelector("#category-select");
  if (input) input.disabled = state.submitting;
  if (browse) browse.disabled = state.submitting;
  if (category) category.disabled = state.submitting;
  const files = state.selectedFiles;
  if (stage) stage.innerHTML = files.length ? `<div class="batch-preview-grid">${files.map((item, index) => `<figure><img src="${esc(item.url)}" alt="Preview of ${esc(item.file.name)}" data-image-fallback="Preview unavailable" /><figcaption>${index + 1}. ${esc(item.file.name)}</figcaption></figure>`).join("")}</div>` : emptyBlock("Image preview", "Choose up to 10 images to preview them here before analysis.");
  if (stage) bindImageFallbacks(stage);
  if (dimensions) dimensions.textContent = files.length ? `${files.length} / 10 selected` : "—";
  if (feedback) feedback.innerHTML = `${files.map((item, index) => `<div class="batch-file ${item.status === "failed" ? "batch-file-failed" : ""}"><img src="${esc(item.url)}" alt="" data-image-fallback="" /><span class="file-name"><b>${esc(item.file.name)}</b><small>${(item.file.size / 1024 / 1024).toFixed(2)} MB · ${item.status === "waiting" ? "Waiting" : item.status === "processing" ? "Processing" : item.status === "completed" ? "Completed" : "Failed"}</small></span>${item.status === "completed" && item.result?.prediction_id ? `<a class="text-link" href="#/history" title="Inspection ${esc(item.result.prediction_id)}">History</a>` : ""}<button class="icon-button subtle" data-action="remove-file" data-index="${index}" aria-label="Remove ${esc(item.file.name)}" ${state.submitting ? "disabled" : ""}>${icon("close", 17)}</button></div>`).join("")}${state.rejectedFiles.map(item => `<div class="batch-rejected" role="alert"><b>${esc(item.name)}</b><span>${esc(item.reason)}</span></div>`).join("")}${files.length ? `<button class="text-button batch-clear" data-action="clear-file" ${state.submitting ? "disabled" : ""}>Clear selection</button>` : ""}`;
  if (button) {
    button.disabled = !files.length || !state.selectedCategory || state.submitting;
    button.innerHTML = state.submitting ? '<span class="spinner spinner-light"></span> Processing images…' : files.length > 1 ? `${icon("scan", 17)} Analyze All (${files.length}) ${icon("arrow", 16)}` : `${icon("scan", 17)} Analyze image ${icon("arrow", 16)}`;
  }
}

function showPreview(file, url) {
  const stage = document.querySelector("#preview-stage");
  if (!stage) return;
  stage.innerHTML = `<img src="${esc(url)}" alt="Selected product image preview" class="selected-preview" data-image-fallback="Input image unavailable" />`;
  bindImageFallbacks(stage);
  const img = stage.querySelector("img");
  img.onload = () => { const label = document.querySelector("#preview-dimensions"); if (label) label.textContent = `${img.naturalWidth} × ${img.naturalHeight} px`; };
  const feedback = document.querySelector("#file-feedback");
  if (feedback) feedback.innerHTML = `<div class="file-summary"><span class="file-type">${icon("image", 16)}</span><span class="file-name"><b>${esc(file.name)}</b><small>${(file.size / 1024 / 1024).toFixed(2)} MB</small></span><button class="icon-button subtle" data-action="clear-file" aria-label="Remove selected image">${icon("close", 17)}</button></div>`;
  const button = document.querySelector("#analyze-button");
  if (button) button.disabled = !state.selectedCategory || state.submitting;
}

function bindImageFallbacks(container) {
  container.querySelectorAll("img[data-image-fallback]").forEach(image => {
    image.addEventListener("error", () => {
      const fallback = document.createElement("div");
      fallback.className = "image-unavailable";
      fallback.setAttribute("role", "status");
      fallback.textContent = image.dataset.imageFallback;
      image.replaceWith(fallback);
    }, { once: true });
  });
}

function displayAnalysis(result) {
  const target = document.querySelector("#analysis-result");
  if (!target) return;
  const urls = result.result || {};
  target.innerHTML = `<section class="panel result-panel"><div class="panel-heading"><div><span class="eyebrow">ANALYSIS COMPLETE</span><h2>Inspection result</h2></div><span class="badge badge-neutral">Score returned</span></div><div class="result-summary"><div class="result-score"><span class="eyebrow">ANOMALY SCORE</span><b>${score(result.anomaly_score)}</b><small>Continuous model score · higher means more anomalous</small></div><div class="result-meta"><div><span>Category</span><b>${esc(titleCase(result.category))}</b></div><div><span>Processing time</span><b>${Number(result.processing_time_ms).toLocaleString()} ms</b></div><div><span>Inspection ID</span><b class="mono">${esc(result.prediction_id)}</b></div></div></div><div class="result-images"><figure><div class="result-image-wrap"><img src="${esc(absoluteUrl(urls.input_url))}" alt="Stored input image" data-image-fallback="Input image unavailable" /></div><figcaption><b>Input image</b><span>Stored request image</span></figcaption></figure><figure><div class="result-image-wrap"><img src="${esc(absoluteUrl(urls.heatmap_url))}" alt="Backend anomaly heatmap" data-image-fallback="Heatmap unavailable" /></div><figcaption><b>Anomaly heatmap</b><span>Backend visualization</span></figcaption></figure><figure><div class="result-image-wrap"><img src="${esc(absoluteUrl(urls.overlay_url))}" alt="Backend anomaly overlay" data-image-fallback="Overlay unavailable" /></div><figcaption><b>Overlay</b><span>Backend visualization</span></figcaption></figure></div><div class="result-footnote">The API returns a continuous anomaly score and visualizations only. This result is not a calibrated pass/fail classification.</div><div class="result-actions"><a href="#/history" class="text-link">View inspection history ${icon("arrow", 15)}</a><button class="button button-secondary button-small" data-action="new-analysis">New analysis</button></div></section>`;
  bindImageFallbacks(target);
  target.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function renderHistory(container) {
  container.innerHTML = `<section class="page-intro"><div><div class="eyebrow"><span class="eyebrow-line"></span>RECORDS</div><h1>Inspection history</h1><p>Browse saved analyses and open their original backend results.</p></div><a class="button button-primary" href="#/inspect">${icon("scan", 17)} New inspection</a></section>
    <section class="panel history-panel"><form class="filter-bar" id="history-filters"><div class="filter-control"><label for="filter-category">Category</label><select class="form-control" id="filter-category">${categoryOptions(state.filterCategory, true)}</select></div><div class="filter-control short"><label for="min-score">Min score</label><input class="form-control" id="min-score" type="number" step="any" min="0" value="${esc(state.minScore)}" placeholder="Any" /></div><div class="filter-control short"><label for="max-score">Max score</label><input class="form-control" id="max-score" type="number" step="any" min="0" value="${esc(state.maxScore)}" placeholder="Any" /></div><div class="filter-control"><label for="sort-order">Sort by time</label><select class="form-control" id="sort-order"><option value="desc" ${state.sortOrder === "desc" ? "selected" : ""}>Newest first</option><option value="asc" ${state.sortOrder === "asc" ? "selected" : ""}>Oldest first</option></select></div><button class="button button-secondary filter-button" type="submit">${icon("search", 16)} Apply filters</button></form><div id="history-content">${loadingBlock("Loading inspection records…")}</div><div id="history-pagination"></div></section>`;
  await loadHistory();
}

async function loadHistory() {
  const body = document.querySelector("#history-content"); if (!body) return;
  body.innerHTML = loadingBlock("Loading inspection records…");
  try {
    const data = await getDetections({ page: state.historyPage, pageSize: 10, category: state.filterCategory, minScore: state.minScore, maxScore: state.maxScore, sortOrder: state.sortOrder });
    state.history = data; state.historyError = ""; body.innerHTML = detectionRows(data.items, false, false);
    const pager = document.querySelector("#history-pagination");
    if (pager) pager.innerHTML = `<div class="pagination"><span>Showing ${data.total ? (data.page - 1) * data.page_size + 1 : 0}–${Math.min(data.page * data.page_size, data.total)} of ${data.total.toLocaleString()} records</span><div><button class="button button-secondary button-small" data-action="history-prev" ${data.page <= 1 ? "disabled" : ""}>Previous</button><span class="page-number">Page ${data.page} / ${Math.max(1, Math.ceil(data.total / data.page_size))}</span><button class="button button-secondary button-small" data-action="history-next" ${data.page * data.page_size >= data.total ? "disabled" : ""}>Next</button></div></div>`;
  } catch (error) {
    state.history = null; state.historyError = error.message;
    body.innerHTML = errorPanel(error.message, "history-retry");
    const pager = document.querySelector("#history-pagination"); if (pager) pager.innerHTML = "";
  }
}

async function renderCategories(container) {
  container.innerHTML = `<section class="page-intro"><div><div class="eyebrow"><span class="eyebrow-line"></span>PRODUCT CATALOG</div><h1>Product categories</h1><p>Inspection classes supported by the MVTec AD feature-bank pipeline.</p></div><a class="button button-secondary" href="#/inspect">${icon("scan", 17)} Inspect product</a></section><div id="category-source-note"></div><section class="category-grid" id="category-grid">${loadingBlock("Loading categories from backend…")}</section>`;
  try { const data = await getCategories(); if (Array.isArray(data?.categories)) { state.categories = data.categories; state.categorySource = "api"; } else throw new Error("Category response did not include a category list."); }
  catch { state.categories = FALLBACK_CATEGORIES; state.categorySource = "fallback"; }
  const note = document.querySelector("#category-source-note"); if (note && state.categorySource === "fallback") note.innerHTML = `<div class="inline-alert warning-alert">${icon("warning", 17)} Backend category list is unavailable. Showing the built-in MVTec AD fallback list; availability of individual banks is not exposed here.</div>`;
  const grid = document.querySelector("#category-grid");
  if (grid) grid.innerHTML = state.categories.map(category => `<article class="category-card"><div class="category-card-body"><h3>${esc(titleCase(category))}</h3><a href="#/inspect" data-category="${esc(category)}" class="category-link">Start inspection ${icon("arrow", 14)}</a></div></article>`).join("");
}

async function renderStatus(container) {
  container.innerHTML = `<section class="page-intro"><div><div class="eyebrow"><span class="eyebrow-line"></span>SYSTEM</div><h1>System status</h1><p>Health and connection details reported by the backend.</p></div><button class="button button-secondary" data-action="refresh-status">${icon("refresh", 16)} Refresh status</button></section><section class="status-section"><div class="status-section-heading"><span class="eyebrow">SERVICE HEALTH</span><h2>API and database</h2></div><div class="status-list" id="status-grid">${loadingBlock("Checking API and database health…")}</div></section><section class="status-section config-section"><div class="status-section-heading"><span class="eyebrow">API CONFIGURATION</span><h2>Backend address</h2></div><form id="api-config-form" class="api-config-form"><label for="api-base">API base URL</label><div class="api-config-row"><input id="api-base" class="form-control" type="url" value="${esc(getApiBase())}" placeholder="http://localhost:8000" /><button class="button button-primary button-small" type="submit">Save address</button></div><small>This address is stored in this browser only. No credentials are stored.</small><div id="api-config-message" aria-live="polite"></div></form></section><section class="status-section model-section"><div class="status-section-heading"><span class="eyebrow">MODEL INFORMATION</span><h2>What is exposed</h2></div><div class="status-note"><b>Model and feature-bank health</b><p>The API does not report model load state or per-category feature-bank availability. Inference readiness cannot be confirmed here.</p></div><div class="status-note"><b>Detection history</b><p>History endpoints require MongoDB-backed persistence.</p></div></section>`;
  await refreshStatusContent();
}

async function refreshStatusContent() {
  const el = document.querySelector("#status-grid"); if (el) el.innerHTML = loadingBlock("Checking API and database health…");
  await readHealth();
  if (!el) return;
  const health = state.health;
  if (!health.online) {
    const invalidAddress = health.errorType === "configuration";
    const httpFailure = health.errorType === "http";
    el.innerHTML = `<article class="health-row health-error"><div><span class="eyebrow">BACKEND API</span><h3>${invalidAddress ? "API address is invalid" : httpFailure ? "Health check failed" : "Could not connect to API"}</h3><p>${esc(health.error)}</p><small>${esc(getApiBase())}/api/health</small></div>${healthBadge(health)}</article><article class="health-row health-unknown"><div><span class="eyebrow">MONGODB</span><h3>Status unavailable</h3><p>A successful health response was not received, so database availability cannot be confirmed.</p></div><span class="badge badge-neutral">Unknown</span></article>`;
  } else {
    const data = health.data || {};
    const apiOk = data.status === "ok";
    const dbOk = data.database === "connected";
    el.innerHTML = `<article class="health-row ${apiOk ? "health-good" : "health-degraded"}"><div><span class="eyebrow">BACKEND API</span><h3>${apiOk ? "Service is online" : "Service is reachable"}</h3><p>Health response received from ${esc(data.service || "backend service")}.</p><small>HTTP ${health.error ? health.error.match(/HTTP (\d+)/)?.[1] || "503" : "200"} · ${esc(getApiBase())}/api/health</small></div>${healthBadge(health)}</article><article class="health-row ${dbOk ? "health-good" : "health-degraded"}"><div><span class="eyebrow">MONGODB</span><h3>${dbOk ? "Database connected" : "Degraded mode"}</h3><p>${dbOk ? "History persistence is available according to the health endpoint." : "The API reports database disconnected; history and persistence may be unavailable."}</p><small>Health response value: ${esc(data.database || "not supplied")}</small></div><span class="badge ${dbOk ? "badge-success" : "badge-warning"}">${dbOk ? "Connected" : "Disconnected"}</span></article>`;
  }
  renderShellStatus();
}

function renderModal(record) {
  const mount = document.querySelector("#modal-root"); if (!mount) return;
  if (!record) { mount.innerHTML = ""; return; }
  const urls = record.result || {};
  if (record.loading) { mount.innerHTML = `<div class="modal-backdrop"><section class="detail-modal" role="dialog" aria-modal="true" aria-label="Loading inspection details" tabindex="-1">${loadingBlock("Loading inspection details…")}</section></div>`; return; }
  mount.innerHTML = `<div class="modal-backdrop" data-action="close-modal"><section class="detail-modal" role="dialog" aria-modal="true" aria-labelledby="modal-title"><div class="modal-header"><div><span class="eyebrow">INSPECTION RECORD</span><h2 id="modal-title">${esc(titleCase(record.category))}</h2></div><button class="icon-button" data-action="close-modal" aria-label="Close dialog">${icon("close", 19)}</button></div><div class="modal-score"><span>ANOMALY SCORE</span><b>${score(record.anomaly_score)}</b><small>Continuous score · no threshold applied</small></div><div class="modal-detail-grid"><div><span>Inspection ID</span><b class="mono">${esc(record.prediction_id)}</b></div><div><span>Created</span><b>${esc(dateTime(record.created_at))}</b></div><div><span>Processing time</span><b>${Number(record.processing_time_ms || 0).toLocaleString()} ms</b></div></div><div class="modal-artifacts"><figure><img src="${esc(absoluteUrl(urls.input_url))}" alt="Input image" data-image-fallback="Input image unavailable"/><figcaption>Input</figcaption></figure><figure><img src="${esc(absoluteUrl(urls.heatmap_url))}" alt="Anomaly heatmap" data-image-fallback="Heatmap unavailable"/><figcaption>Heatmap</figcaption></figure><figure><img src="${esc(absoluteUrl(urls.overlay_url))}" alt="Anomaly overlay" data-image-fallback="Overlay unavailable"/><figcaption>Overlay</figcaption></figure></div><div id="delete-message"></div><div class="modal-actions"><button class="button button-danger button-small" data-action="delete-detection" data-id="${esc(record.prediction_id)}">${icon("trash", 15)} Delete record</button><button class="button button-secondary button-small" data-action="close-modal">Close</button></div></section></div>`;
  bindImageFallbacks(mount);
}

function closeModal(restoreFocus = true) {
  const opener = state.modalOpener;
  state.modal = null; state.modalOpener = null; renderModal(null);
  if (restoreFocus && opener?.isConnected && !opener.disabled) opener.focus();
}

async function openDetection(id, opener) {
  state.modalOpener = opener;
  state.modal = { loading: true }; renderModal(state.modal);
  document.querySelector("#modal-root [role='dialog']")?.focus();
  try { state.modal = await getDetection(id); renderModal(state.modal); document.querySelector("#modal-root button[data-action='close-modal']")?.focus(); }
  catch (error) { closeModal(); showToast(error.message, "error"); }
}

async function route() {
  const requested = location.hash.replace(/^#\/?/, "");
  state.route = NAV.some(([key]) => key === requested) ? requested : "dashboard";
  if (!location.hash || requested !== state.route) history.replaceState(null, "", `#/${state.route}`);
  shell('<div class="route-loading">Loading workspace…</div>');
  const container = document.querySelector("#page-content");
  if (state.route === "dashboard") await renderDashboard(container);
  else if (state.route === "inspect") await renderInspect(container);
  else if (state.route === "history") await renderHistory(container);
  else if (state.route === "categories") await renderCategories(container);
  else await renderStatus(container);
  renderModal(state.modal);
}

function showToast(message, kind = "success") {
  let toast = document.querySelector("#toast");
  if (!toast) { toast = document.createElement("div"); toast.id = "toast"; document.body.append(toast); }
  toast.className = `toast toast-${kind}`; toast.textContent = message; toast.classList.add("visible");
  clearTimeout(showToast.timer); showToast.timer = setTimeout(() => toast.classList.remove("visible"), 3600);
}

document.addEventListener("click", async event => {
  const anchor = event.target.closest("a[href^='#/']");
  if (anchor?.dataset.category) { state.selectedCategory = anchor.dataset.category; }
  if (event.target.closest(".primary-nav a") && state.mobileMenuOpen) setMobileMenu(false, false);
  const button = event.target.closest("[data-action]"); if (!button) return;
  const action = button.dataset.action;
  if (action === "browse-file") document.querySelector("#file-input")?.click();
  if (action === "clear-file") { event.preventDefault(); if (state.submitting) return; clearSelectedFiles(); state.analysis = null; const result = document.querySelector("#analysis-result"); if (result) result.innerHTML = ""; renderSelectedFiles(); }
  if (action === "remove-file") { if (state.submitting) return; removeSelectedFile(Number(button.dataset.index)); renderSelectedFiles(); const result = document.querySelector("#analysis-result"); if (state.selectedFiles.length > 1 && state.selectedFiles.some(item => item.status !== "waiting")) renderBatchResults(); else if (result) result.innerHTML = ""; }
  if (action === "analyze") await submitAnalysis();
  if (action === "new-analysis") { if (state.submitting) return; state.analysis = null; clearSelectedFiles(); await route(); }
  if (action === "history-prev" && state.historyPage > 1) { state.historyPage--; await loadHistory(); }
  if (action === "history-next" && state.history && state.historyPage * state.history.page_size < state.history.total) { state.historyPage++; await loadHistory(); }
  if (action === "history-retry") await loadHistory();
  if (action === "view-detection") await openDetection(button.dataset.id, button);
  if (action === "close-modal") { if (event.target === button || button === event.target.closest("button")) closeModal(); else if (event.target.classList.contains("modal-backdrop")) closeModal(); }
  if (action === "delete-detection") {
    if (!confirm("Delete this inspection record and its stored images? This cannot be undone.")) return;
    button.disabled = true;
    try { await deleteDetection(button.dataset.id); closeModal(state.route !== "history"); showToast("Inspection record deleted."); if (state.route === "history") await loadHistory(); }
    catch (error) { const message = document.querySelector("#delete-message"); if (message) message.innerHTML = errorPanel(error.message); button.disabled = false; }
  }
  if (action === "refresh-status") await refreshStatusContent();
  if (action === "menu-toggle") {
    if (isMobileViewport()) setMobileMenu(!state.mobileMenuOpen);
    else setDesktopSidebarCollapsed(!state.desktopSidebarCollapsed);
  }
  if (action === "close-navigation") setMobileMenu(false);
});

document.addEventListener("change", event => {
  if (event.target.id === "category-select") { state.selectedCategory = event.target.value; renderSelectedFiles(); }
  if (event.target.id === "file-input") { addSelectedFiles(Array.from(event.target.files || [])); event.target.value = ""; renderSelectedFiles(); }
});
document.addEventListener("submit", async event => {
  if (event.target.id === "history-filters") {
    event.preventDefault(); state.filterCategory = document.querySelector("#filter-category").value; state.minScore = document.querySelector("#min-score").value; state.maxScore = document.querySelector("#max-score").value; state.sortOrder = document.querySelector("#sort-order").value; state.historyPage = 1; await loadHistory();
  }
  if (event.target.id === "api-config-form") {
    event.preventDefault(); const target = document.querySelector("#api-config-message");
    try { saveApiBase(document.querySelector("#api-base").value); if (target) target.innerHTML = '<span class="config-saved">API address saved. Refreshing health…</span>'; await refreshStatusContent(); showToast("API address updated."); }
    catch (error) { if (target) target.innerHTML = `<span class="config-error">${esc(error.message)}</span>`; }
  }
});

function addSelectedFiles(files) {
  if (state.submitting) return;
  const selection = addBatchFiles(state.selectedFiles, files, file => URL.createObjectURL(file));
  state.selectedFiles = selection.items;
  state.rejectedFiles = selection.rejected;
  state.file = state.selectedFiles[0]?.file || null;
  state.previewUrl = state.selectedFiles[0]?.url || "";
  state.analysis = null;
  const result = document.querySelector("#analysis-result"); if (result) result.innerHTML = "";
}

function removeSelectedFile(index) {
  const result = removeBatchItem(state.selectedFiles, index);
  state.selectedFiles = result.items;
  if (result.removed) URL.revokeObjectURL(result.removed.url);
  state.file = state.selectedFiles[0]?.file || null;
  state.previewUrl = state.selectedFiles[0]?.url || "";
  state.analysis = null;
}

function clearSelectedFiles() {
  state.batchRunId++;
  for (const item of state.selectedFiles) URL.revokeObjectURL(item.url);
  state.selectedFiles = [];
  state.rejectedFiles = [];
  state.file = null;
  state.previewUrl = "";
}

async function submitAnalysis() {
  if (!state.selectedFiles.length || !state.selectedCategory || state.submitting) return;
  if (state.selectedFiles.length > 1) return submitBatch();
  state.file = state.selectedFiles[0].file;
  state.submitting = true;
  renderSelectedFiles();
  const button = document.querySelector("#analyze-button"); if (button) { button.disabled = true; button.innerHTML = '<span class="spinner spinner-light"></span> Analyzing image…'; }
  const target = document.querySelector("#analysis-result"); if (target) target.innerHTML = `<div class="analysis-progress">${loadingBlock("Uploading image and running anomaly analysis…")}<small>Inference may take several seconds while the category model loads.</small></div>`;
  try { state.analysis = await analyzeImage(state.selectedCategory, state.file); displayAnalysis(state.analysis); }
  catch (error) { if (target) target.innerHTML = errorPanel(error.message); }
  finally { state.submitting = false; renderSelectedFiles(); }
}

function renderBatchResults() {
  const target = document.querySelector("#analysis-result"); if (!target) return;
  const done = state.selectedFiles.filter(item => item.status === "completed" || item.status === "failed").length;
  target.innerHTML = `<section class="batch-results"><div class="panel-heading"><div><span class="eyebrow">BATCH INSPECTION</span><h2>${done} / ${state.selectedFiles.length} processed</h2></div><button class="button button-secondary button-small" data-action="new-analysis" ${state.submitting ? "disabled" : ""}>New batch</button></div><div class="batch-result-grid">${state.selectedFiles.map(item => `<article class="panel batch-result-card"><div class="batch-result-heading"><b>${esc(item.file.name)}</b><span class="badge ${item.status === "completed" ? "badge-success" : item.status === "failed" ? "badge-danger" : item.status === "processing" ? "badge-warning" : "badge-neutral"}">${item.status === "waiting" ? "Waiting" : item.status === "processing" ? "Processing" : item.status === "completed" ? "Completed" : "Failed"}</span></div>${item.status === "processing" ? loadingBlock("Running inspection…") : ""}${item.status === "failed" ? errorPanel(item.error) : ""}${item.result ? `<div class="batch-score"><span>Anomaly Score</span><b>${score(item.result.anomaly_score)}</b></div><div class="batch-result-meta">${item.result.processing_time_ms != null ? `<span>Processing time: ${Number(item.result.processing_time_ms).toLocaleString()} ms</span>` : ""}${item.result.prediction_id ? `<a class="text-link" href="#/history" title="${esc(item.result.prediction_id)}">Inspection ${esc(item.result.prediction_id.slice(0, 8).toUpperCase())} · History</a>` : ""}</div><div class="batch-artifacts">${[["Input image", item.result.result?.input_url, item.url], ["Heatmap", item.result.result?.heatmap_url], ["Overlay", item.result.result?.overlay_url]].map(([label, path, local]) => `<figure><div>${path ? `<img src="${esc(absoluteUrl(path))}" alt="${esc(label)} for ${esc(item.file.name)}" data-image-fallback="${esc(label)} unavailable" />` : local ? `<img src="${esc(local)}" alt="Selected input ${esc(item.file.name)}" data-image-fallback="Input image unavailable" />` : "Image unavailable"}</div><figcaption>${label}</figcaption></figure>`).join("")}</div>` : ""}</article>`).join("")}</div></section>`;
  bindImageFallbacks(target);
}

async function submitBatch() {
  if (!state.selectedFiles.length || !state.selectedCategory || state.submitting) return;
  const runId = ++state.batchRunId;
  state.submitting = true;
  state.analysis = null;
  for (const item of state.selectedFiles) { item.status = "waiting"; item.result = null; item.error = ""; }
  renderSelectedFiles(); renderBatchResults();
  try {
    await processBatch(state.selectedFiles, state.selectedCategory, analyzeImage, () => { renderSelectedFiles(); renderBatchResults(); }, () => runId === state.batchRunId);
  } finally {
    if (runId === state.batchRunId) { state.submitting = false; renderSelectedFiles(); renderBatchResults(); }
  }
}

document.addEventListener("click", event => { const dropzone = event.target.closest("#dropzone"); if (dropzone && !event.target.closest("button")) document.querySelector("#file-input")?.click(); });
document.addEventListener("dragover", event => { if (event.target.closest("#dropzone")) { event.preventDefault(); document.querySelector("#dropzone")?.classList.add("drag-active"); } });
document.addEventListener("dragleave", event => { if (event.target.closest("#dropzone")) document.querySelector("#dropzone")?.classList.remove("drag-active"); });
document.addEventListener("drop", event => { if (event.target.closest("#dropzone")) { event.preventDefault(); document.querySelector("#dropzone")?.classList.remove("drag-active"); addSelectedFiles(Array.from(event.dataTransfer?.files || [])); renderSelectedFiles(); } });
document.addEventListener("keydown", event => {
  if (event.key === "Escape" && state.mobileMenuOpen) { setMobileMenu(false); return; }
  if (event.key === "Escape" && state.modal) { closeModal(); return; }
  if (event.key === "Tab" && state.mobileMenuOpen) {
    const sidebar = document.querySelector("#sidebar");
    const items = Array.from(sidebar?.querySelectorAll("a[href],button:not([disabled])") || []).filter(item => item.getClientRects().length);
    const first = items[0], last = items[items.length - 1];
    if (!first) { event.preventDefault(); return; }
    if (!sidebar.contains(document.activeElement)) { event.preventDefault(); (event.shiftKey ? last : first).focus(); }
    else if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    return;
  }
  if (event.key !== "Tab" || !state.modal) return;
  const dialog = document.querySelector("#modal-root [role='dialog']");
  const focusable = Array.from(dialog?.querySelectorAll("a[href],button:not([disabled]),input:not([disabled]),select:not([disabled]),textarea:not([disabled]),[tabindex]:not([tabindex='-1'])") || []).filter(el => !el.hidden && el.getClientRects().length);
  if (!focusable.length) { event.preventDefault(); dialog?.focus(); return; }
  const first = focusable[0], last = focusable[focusable.length - 1];
  if (!dialog.contains(document.activeElement)) { event.preventDefault(); (event.shiftKey ? last : first).focus(); }
  else if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
  else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
});
window.addEventListener("hashchange", () => { if (state.modal) closeModal(false); if (state.mobileMenuOpen) setMobileMenu(false, false); route(); });
window.addEventListener("resize", () => {
  if (!isMobileViewport() && state.mobileMenuOpen) setMobileMenu(false, false);
  else syncNavigationToggle();
});
route();
