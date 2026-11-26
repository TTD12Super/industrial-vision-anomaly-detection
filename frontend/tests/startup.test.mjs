import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

test("mobile close button is hidden except in the open mobile sidebar", async () => {
  const css = await readFile(new URL("../src/styles.css", import.meta.url), "utf8");
  assert.match(css, /\.sidebar \.mobile-nav-close\s*\{\s*display:\s*none;\s*\}/);
  assert.match(css, /@media \(max-width: 760px\)\s*\{[\s\S]*?\.sidebar\.sidebar-open \.mobile-nav-close\s*\{[^}]*display:\s*grid;/);
  assert.doesNotMatch(css, /@media \(min-width: 761px\)\s*\{[^}]*\.mobile-nav-close[^}]*display:\s*grid;/);
});

test("dashboard initializes with readable degraded UI when initial API requests fail", async () => {
  const app = { innerHTML: "" };
  const predictionId = "a12bc34d-5678-4abc-9def-0123456789ab";
  const detection = { prediction_id: predictionId, category: "wood", anomaly_score: 0.42, processing_time_ms: 18, created_at: "2026-10-31T00:00:00Z", result: { input_url: "/input", heatmap_url: "/heatmap", overlay_url: "/overlay" } };
  let listRequests = 0, categoryRequests = 0, healthResponse = null;
  const nodes = new Map();
  const listeners = new Map();
  const windowListeners = new Map();
  let mobileViewport = true;
  const reply = (body, status = 200) => ({ ok: status >= 200 && status < 300, status, headers: { get: () => "application/json" }, async json() { return body; } });
  let documentMock;
  const node = key => {
    if (!nodes.has(key)) {
      const classes = new Set(), attributes = {};
      nodes.set(key, { innerHTML: "", className: "", attributes,
        classList: { add: name => classes.add(name), remove: name => classes.delete(name), toggle: (name, force) => force ? classes.add(name) : classes.delete(name), contains: name => classes.has(name) },
        setAttribute(name, value) { attributes[name] = value; }, focus() { documentMock.activeElement = this; }, getClientRects() { return [1]; }, querySelectorAll() { return []; },
      });
    }
    return nodes.get(key);
  };
  globalThis.location = { hash: "" };
  globalThis.history = { replaceState() {} };
  globalThis.localStorage = { getItem() { return null; }, setItem() {} };
  globalThis.window = { addEventListener(type, listener) { windowListeners.set(type, listener); }, matchMedia() { return { matches: mobileViewport }; } };
  documentMock = globalThis.document = {
    querySelector(selector) { return selector === "#app" ? app : node(selector); },
    addEventListener(type, listener) { listeners.set(type, [...(listeners.get(type) || []), listener]); },
    createElement() { return node("created"); },
    body: { append() {} },
  };
  globalThis.fetch = async url => {
    const path = String(url);
    if (path.endsWith("/api/health")) { if (!healthResponse) throw new Error("mock backend unavailable"); return healthResponse; }
    if (path.includes("/api/detections?")) {
      listRequests++;
      return reply(listRequests === 1 ? { items: null, total: 0 } : { items: [detection], total: 1, page: 1, page_size: 10 });
    }
    if (path.endsWith("/api/categories")) {
      categoryRequests++;
      if (categoryRequests === 1) throw new Error("mock backend unavailable");
      return reply({ categories: ["wood", "cable"] });
    }
    if (path === `http://localhost:8000/api/detections/${predictionId}`) return reply(detection);
    throw new Error("mock backend unavailable");
  };

  await import("../src/main.js");
  await new Promise(resolve => setTimeout(resolve, 20));
  assert.match(app.innerHTML, /class="app-shell"/);
  assert.match(app.innerHTML, /class="api-indicator[^\"]*" role="status" aria-live="polite" aria-atomic="true"/);
  assert.doesNotMatch(app.innerHTML, /Loading inspection workspace/);
  assert.match(node("#dashboard-stats").innerHTML, /Fallback catalog/);
  assert.doesNotMatch(node("#dashboard-stats").innerHTML, /stat-mark|mark-green/);
  assert.match(node("#dashboard-history").innerHTML, /unexpected response/);
  assert.match(node("#page-content").innerHTML, /Model health is not reported by the API/);
  assert.doesNotMatch(node("#page-content").innerHTML, /engine-meta.*status-dot/);
  assert.match(node(".api-indicator").innerHTML, /API connection issue/);
  assert.match(node(".api-indicator").innerHTML, /class="status-dot" aria-hidden="true"/);
  assert.match(node(".api-indicator").innerHTML, /class="status-label"/);
  for (const label of ["Dashboard", "Inspect Product", "Inspection History", "Product Categories", "System Status"]) assert.match(app.innerHTML, new RegExp(label));
  assert.match(app.innerHTML, /aria-expanded="false" aria-controls="sidebar"/);
  assert.match(app.innerHTML, /data-action="close-navigation" aria-label="Close navigation menu"/);

  const clickTarget = (element, isNavigationLink = false) => ({ closest: selector => {
    if (selector === "[data-action]" && element.dataset?.action) return element;
    if (isNavigationLink && (selector === ".primary-nav a" || selector === "a[href^='#/']")) return element;
    return null;
  } });
  const menuToggle = node("#menu-toggle");
  menuToggle.dataset = { action: "menu-toggle" };
  const closeButton = { dataset: { action: "close-navigation" }, focus() { documentMock.activeElement = this; }, getClientRects() { return [1]; } };
  const firstLink = node("#sidebar .primary-nav a");
  const lastLink = { focus() { documentMock.activeElement = this; }, getClientRects() { return [1]; } };
  const sidebar = node("#sidebar");
  sidebar.querySelectorAll = () => [closeButton, firstLink, lastLink];
  sidebar.contains = element => [closeButton, firstLink, lastLink].includes(element);
  await listeners.get("click")[0]({ target: clickTarget(menuToggle) });
  assert.equal(node(".app-shell").classList.contains("menu-open"), true);
  assert.equal(node("#sidebar").classList.contains("sidebar-open"), true);
  assert.equal(menuToggle.attributes["aria-expanded"], "true");

  listeners.get("keydown")[0]({ key: "Escape" });
  assert.equal(node("#sidebar").classList.contains("sidebar-open"), false);
  assert.equal(menuToggle.attributes["aria-expanded"], "false");
  assert.equal(documentMock.activeElement, menuToggle);

  await listeners.get("click")[0]({ target: clickTarget(menuToggle) });
  await listeners.get("click")[0]({ target: clickTarget(closeButton) });
  assert.equal(node(".app-shell").classList.contains("menu-open"), false);

  await listeners.get("click")[0]({ target: clickTarget(menuToggle) });
  documentMock.activeElement = lastLink;
  let prevented = false;
  listeners.get("keydown")[0]({ key: "Tab", shiftKey: false, preventDefault() { prevented = true; } });
  assert.equal(prevented, true);
  assert.equal(documentMock.activeElement, closeButton);

  const backdrop = { dataset: { action: "close-navigation" } };
  await listeners.get("click")[0]({ target: clickTarget(backdrop) });
  assert.equal(node("#sidebar").classList.contains("sidebar-open"), false);

  await listeners.get("click")[0]({ target: clickTarget(menuToggle) });
  const navLink = { dataset: {} };
  await listeners.get("click")[0]({ target: clickTarget(navLink, true) });
  assert.equal(node(".app-shell").classList.contains("menu-open"), false);

  globalThis.location.hash = "#/inspect";
  windowListeners.get("hashchange")();
  await new Promise(resolve => setTimeout(resolve, 20));
  const inspectMarkup = node("#page-content").innerHTML;
  assert.doesNotMatch(inspectMarkup, /step-count|01\s*<i>\/\s*03/);
  assert.match(inspectMarkup, /id="category-select"/);
  assert.match(inspectMarkup, /id="file-input"[^>]*multiple/);
  assert.match(inspectMarkup, /id="analyze-button"/);
  assert.match(inspectMarkup, /Score interpretation/);

  globalThis.location.hash = "#/history";
  windowListeners.get("hashchange")();
  await new Promise(resolve => setTimeout(resolve, 20));
  const historyMarkup = node("#history-content").innerHTML;
  assert.match(historyMarkup, /A12BC34D/);
  assert.doesNotMatch(historyMarkup, new RegExp(`<small>${predictionId}</small>`));
  assert.match(historyMarkup, new RegExp(`data-id="${predictionId}"`));

  const viewButton = { dataset: { action: "view-detection", id: predictionId } };
  await listeners.get("click")[0]({ target: clickTarget(viewButton) });
  assert.match(node("#modal-root").innerHTML, new RegExp(`<b class="mono">${predictionId}</b>`));

  globalThis.location.hash = "#/categories";
  windowListeners.get("hashchange")();
  await new Promise(resolve => setTimeout(resolve, 20));
  const categoriesMarkup = node("#category-grid").innerHTML;
  assert.match(categoriesMarkup, /<h3>Wood<\/h3>/);
  assert.match(categoriesMarkup, /href="#\/inspect" data-category="wood"/);
  assert.doesNotMatch(categoriesMarkup, /category-tone-|MVTec AD · 01|Category entry from/);

  const categoryLink = { dataset: { category: "wood" } };
  await listeners.get("click")[0]({ target: clickTarget(categoryLink, true) });
  globalThis.location.hash = "#/inspect";
  windowListeners.get("hashchange")();
  await new Promise(resolve => setTimeout(resolve, 20));
  assert.match(node("#category-select").innerHTML, /value="wood" selected/);

  healthResponse = reply({ status: "ok", service: "Inspectra API", database: "connected" });
  globalThis.location.hash = "#/status";
  windowListeners.get("hashchange")();
  await new Promise(resolve => setTimeout(resolve, 20));
  assert.match(node("#page-content").innerHTML, /SERVICE HEALTH/);
  assert.match(node("#page-content").innerHTML, /API CONFIGURATION/);
  assert.match(node("#page-content").innerHTML, /MODEL INFORMATION/);
  assert.match(node("#page-content").innerHTML, /does not report model load state or per-category feature-bank availability/);
  assert.match(node("#status-grid").innerHTML, /Service is online/);
  assert.match(node("#status-grid").innerHTML, /Database connected/);
  assert.match(node("#page-content").innerHTML, /id="api-config-form"[\s\S]*Save address/);

  healthResponse = reply({ status: "degraded", service: "Inspectra API", database: "disconnected" }, 503);
  await listeners.get("click")[0]({ target: clickTarget({ dataset: { action: "refresh-status" } }) });
  assert.match(node("#status-grid").innerHTML, /Degraded/);
  assert.match(node("#status-grid").innerHTML, /Database connected|Degraded mode/);
  assert.match(node("#status-grid").innerHTML, /Disconnected/);

  healthResponse = null;
  await listeners.get("click")[0]({ target: clickTarget({ dataset: { action: "refresh-status" } }) });
  assert.match(node("#status-grid").innerHTML, /Could not connect to API/);
  assert.match(node("#status-grid").innerHTML, /Status unavailable/);
  assert.match(node("#status-grid").innerHTML, /Unknown/);

  const shell = node(".app-shell");
  mobileViewport = false;
  windowListeners.get("resize")();
  assert.equal(menuToggle.attributes["aria-label"], "Collapse navigation sidebar");
  assert.equal(menuToggle.attributes["aria-expanded"], "true");
  await listeners.get("click")[0]({ target: clickTarget(menuToggle) });
  assert.equal(shell.classList.contains("sidebar-collapsed"), true);
  assert.equal(menuToggle.attributes["aria-expanded"], "false");
  assert.equal(menuToggle.attributes["aria-label"], "Expand navigation sidebar");

  mobileViewport = true;
  windowListeners.get("resize")();
  assert.equal(shell.classList.contains("sidebar-collapsed"), true);
  assert.equal(shell.classList.contains("menu-open"), false);
  assert.equal(menuToggle.attributes["aria-expanded"], "false");
  await listeners.get("click")[0]({ target: clickTarget(menuToggle) });
  assert.equal(shell.classList.contains("menu-open"), true);
  assert.equal(shell.classList.contains("sidebar-collapsed"), true);

  mobileViewport = false;
  windowListeners.get("resize")();
  assert.equal(shell.classList.contains("menu-open"), false);
  assert.equal(shell.classList.contains("sidebar-open"), false);
  assert.equal(shell.classList.contains("sidebar-collapsed"), true);
  assert.equal(menuToggle.attributes["aria-expanded"], "false");

  await listeners.get("click")[0]({ target: clickTarget(menuToggle) });
  assert.equal(shell.classList.contains("sidebar-collapsed"), false);
  assert.equal(menuToggle.attributes["aria-expanded"], "true");
});
