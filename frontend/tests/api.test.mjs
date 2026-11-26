import test from "node:test";
import assert from "node:assert/strict";
import { absoluteUrl, analyzeImage, deleteDetection, getApiBase, getCategories, getDetection, getDetections, getHealth, saveApiBase } from "../src/api.js";

const health = { status: "ok", service: "test", database: "connected" };
const categories = { categories: ["wood"] };
const links = { input_url: "/api/detections/id/input", heatmap_url: "/api/detections/id/heatmap", overlay_url: "/api/detections/id/overlay" };
const summary = { prediction_id: "id", category: "wood", anomaly_score: 0.2, processing_time_ms: 12, created_at: "2026-10-31T00:00:00Z", result: links };
const detection = { ...summary };
const reply = (body, { status = 200, contentType = "application/json" } = {}) => ({
  ok: status >= 200 && status < 300,
  status,
  headers: { get: name => name === "content-type" ? contentType : null },
  async json() { if (body instanceof Error) throw body; return body; },
});
function mockStorage(value = null) {
  let stored = value;
  globalThis.localStorage = { getItem: () => stored, setItem: (_key, next) => { stored = next; } };
}

test("preserves URL path prefixes for API requests and artifact URLs", async () => {
  mockStorage();
  saveApiBase("https://example.test/service/");
  let requested = "";
  globalThis.fetch = async url => { requested = url; return reply(categories); };
  await getCategories();
  assert.equal(getApiBase(), "https://example.test/service");
  assert.equal(requested, "https://example.test/service/api/categories");
  assert.equal(absoluteUrl(links.input_url), "https://example.test/service/api/detections/id/input");
});

test("uses the default API URL if localStorage cannot be read", () => {
  globalThis.localStorage = { getItem() { throw new Error("storage blocked"); } };
  assert.equal(getApiBase(), "http://localhost:8000");
});

test("rejects invalid JSON and non-JSON success responses", async () => {
  mockStorage();
  globalThis.fetch = async () => reply(new Error("bad json"));
  await assert.rejects(getHealth(), /unexpected response/);
  globalThis.fetch = async () => reply("ok", { contentType: "text/plain" });
  await assert.rejects(getHealth(), /unexpected response/);
});

test("rejects structurally invalid successful responses instead of treating them as success", async () => {
  mockStorage();
  globalThis.fetch = async url => String(url).endsWith("/api/health") ? reply({ status: "ok" }) : reply({ items: null, total: 0 });
  await assert.rejects(getHealth(), /unexpected response/);
  await assert.rejects(getDetections(), /unexpected response/);
  globalThis.fetch = async () => reply({ ...detection, anomaly_score: "0.2" });
  await assert.rejects(getDetection("id"), /unexpected response/);
});

test("validates the actual health, categories, history, and detail response shapes", async () => {
  mockStorage();
  globalThis.fetch = async url => {
    const path = String(url);
    if (path.endsWith("/api/health")) return reply(health);
    if (path.endsWith("/api/categories")) return reply(categories);
    if (path.includes("/api/detections?")) return reply({ items: [summary], total: 1, page: 1, page_size: 10 });
    return reply(detection);
  };
  assert.deepEqual(await getHealth(), health);
  assert.deepEqual(await getCategories(), categories);
  assert.equal((await getDetections()).items[0].prediction_id, "id");
  assert.equal((await getDetection("id")).category, "wood");
});

test("preserves HTTP failure details and validates successful detection and deletion bodies", async () => {
  mockStorage();
  globalThis.fetch = async () => reply({ detail: "bad request" }, { status: 400 });
  await assert.rejects(getCategories(), error => error.status === 400 && error.message === "bad request");
  globalThis.fetch = async url => String(url).endsWith("/api/detect") ? reply(detection) : reply({ message: "deleted", prediction_id: "id" });
  const file = new File(["image"], "sample.png", { type: "image/png" });
  const result = await analyzeImage("wood", file);
  assert.equal(result.prediction_id, "id");
  assert.deepEqual(await deleteDetection("id"), { message: "deleted", prediction_id: "id" });
});
