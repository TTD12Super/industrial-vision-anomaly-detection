/** @typedef {{ input_url: string, heatmap_url: string, overlay_url: string }} DetectionLinks */
/** @typedef {{ prediction_id: string, category: string, anomaly_score: number, processing_time_ms: number, result: DetectionLinks }} DetectionResponse */
/** @typedef {{ prediction_id: string, category: string, anomaly_score: number, processing_time_ms: number, created_at: string, result: DetectionLinks }} DetectionSummary */
/** @typedef {{ items: DetectionSummary[], total: number, page: number, page_size: number }} DetectionListResponse */
/** @typedef {{ status: string, service: string, database: string }} HealthResponse */

const DEFAULT_API_BASE = "http://localhost:8000";
const API_SETTING_KEY = "industrialVisionApiBaseUrl";

export function getApiBase() {
  try { return (localStorage.getItem(API_SETTING_KEY) || DEFAULT_API_BASE).replace(/\/$/, ""); }
  catch { return DEFAULT_API_BASE; }
}

function invalidApiBase() {
  const error = new Error("Enter a complete API base URL using HTTP or HTTPS, without a query or fragment.");
  error.kind = "configuration";
  return error;
}

function validateApiBase(value) {
  const clean = String(value ?? "").trim().replace(/\/$/, "");
  let parsed;
  try { parsed = new URL(clean); }
  catch { throw invalidApiBase(); }
  const authority = /^https?:\/\/([^/?#]+)/i.exec(clean)?.[1];
  if (!authority || !/^https?:\/\//i.test(clean) || !["http:", "https:"].includes(parsed.protocol) || !parsed.hostname || clean.includes("?") || clean.includes("#")) {
    throw invalidApiBase();
  }
  return clean;
}

export function saveApiBase(value) {
  const clean = validateApiBase(value);
  localStorage.setItem(API_SETTING_KEY, clean);
  return clean;
}

export function absoluteUrl(path) {
  if (!path) return "";
  try {
    const base = validateApiBase(getApiBase());
    const relativePath = /^[a-z][a-z\d+.-]*:\/\//i.test(path) ? path : String(path).replace(/^\/+/, "");
    return new URL(relativePath, `${base}/`).toString();
  }
  catch { return ""; }
}

async function request(path, options = {}) {
  const base = validateApiBase(getApiBase());
  let response;
  try {
    response = await fetch(`${base}${path}`, options);
  } catch {
    throw new Error("Could not connect to the API. Check the address and whether the backend is running.");
  }
  const contentType = response.headers.get("content-type") || "";
  let body = null;
  if (contentType.includes("application/json")) {
    try { body = await response.json(); } catch { body = null; }
  }
  if (!response.ok) {
    const detail = body?.detail;
    const message = response.status >= 500
      ? `The API returned a server error (HTTP ${response.status}).`
      : typeof detail === "string" ? detail : `Request failed with HTTP ${response.status}.`;
    const error = new Error(message);
    error.status = response.status;
    error.body = body;
    throw error;
  }
  if (!contentType.includes("application/json") || !body || typeof body !== "object" || Array.isArray(body)) {
    throw unexpectedResponse();
  }
  return body;
}

function unexpectedResponse() {
  return new Error("The API returned an unexpected response. Check that the configured address points to the expected backend.");
}
function requireString(value) { if (typeof value !== "string" || !value) throw unexpectedResponse(); }
function requireNumber(value) { if (typeof value !== "number" || !Number.isFinite(value)) throw unexpectedResponse(); }
function requireInteger(value) { if (!Number.isInteger(value)) throw unexpectedResponse(); }
function requireLinks(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw unexpectedResponse();
  requireString(value.input_url); requireString(value.heatmap_url); requireString(value.overlay_url);
}
function requireDetection(value, isSummary = false) {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw unexpectedResponse();
  requireString(value.prediction_id); requireString(value.category); requireNumber(value.anomaly_score); requireNumber(value.processing_time_ms);
  if (isSummary) requireString(value.created_at);
  requireLinks(value.result);
  return value;
}

/** @returns {Promise<HealthResponse>} */
export async function getHealth() {
  const data = await request("/api/health");
  requireString(data.status); requireString(data.service); requireString(data.database);
  return data;
}
/** @returns {Promise<{categories: string[]}>} */
export async function getCategories() {
  const data = await request("/api/categories");
  if (!Array.isArray(data.categories) || !data.categories.length || data.categories.some(category => typeof category !== "string" || !category)) throw unexpectedResponse();
  return data;
}

/** @returns {Promise<DetectionResponse>} */
export function analyzeImage(category, file) {
  const body = new FormData();
  body.append("category", category);
  body.append("file", file, file.name);
  return request("/api/detect", { method: "POST", body }).then(data => requireDetection(data));
}

/** @returns {Promise<DetectionListResponse>} */
export function getDetections({ page = 1, pageSize = 10, category = "", minScore = "", maxScore = "", sortOrder = "desc" } = {}) {
  const params = new URLSearchParams({ page: String(page), page_size: String(pageSize), sort_order: sortOrder });
  if (category) params.set("category", category);
  if (minScore !== "") params.set("min_score", minScore);
  if (maxScore !== "") params.set("max_score", maxScore);
  return request(`/api/detections?${params}`).then(data => {
    if (!Array.isArray(data.items) || data.items.some(item => { try { requireDetection(item, true); return false; } catch { return true; } })) throw unexpectedResponse();
    requireInteger(data.total); requireInteger(data.page); requireInteger(data.page_size);
    return data;
  });
}

/** @returns {Promise<DetectionSummary>} */
export function getDetection(id) { return request(`/api/detections/${encodeURIComponent(id)}`).then(data => requireDetection(data, true)); }
export function deleteDetection(id) {
  return request(`/api/detections/${encodeURIComponent(id)}`, { method: "DELETE" }).then(data => {
    requireString(data.message); requireString(data.prediction_id);
    return data;
  });
}
