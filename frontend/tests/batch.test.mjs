import test from "node:test";
import assert from "node:assert/strict";
import { addBatchFiles, processBatch, removeBatchItem } from "../src/batch.mjs";

const image = (name, overrides = {}) => ({ name, type: "image/png", size: 100, lastModified: 1, ...overrides });
const urls = file => `blob:${file.name}`;

test("accepts one image and keeps it available for the existing single-image path", () => {
  const { items, rejected } = addBatchFiles([], [image("one.png")], urls);
  assert.equal(items.length, 1);
  assert.equal(items[0].status, "waiting");
  assert.deepEqual(rejected, []);
});

test("accepts 10 images, rejects excess explicitly, and rejects unsupported files", () => {
  const first = addBatchFiles([], Array.from({ length: 10 }, (_, i) => image(`${i}.png`)), urls);
  assert.equal(first.items.length, 10);
  const second = addBatchFiles(first.items, [image("eleventh.png"), image("bad.gif", { type: "image/gif" })], urls);
  assert.equal(second.items.length, 10);
  assert.match(second.rejected[0].reason, /limit is 10/);
  assert.match(second.rejected[1].reason, /JPG, PNG or WEBP/);
});

test("rejects oversized and empty images using current upload limits", () => {
  const result = addBatchFiles([], [image("large.png", { size: 10 * 1024 * 1024 + 1 }), image("empty.png", { size: 0 })], urls);
  assert.equal(result.items.length, 0);
  assert.match(result.rejected[0].reason, /10 MB/);
  assert.match(result.rejected[1].reason, /empty/);
});

test("rejects duplicates and removes one selected item", () => {
  const file = image("same.png");
  const first = addBatchFiles([], [file], urls);
  const duplicate = addBatchFiles(first.items, [file], urls);
  assert.equal(duplicate.items.length, 1);
  assert.match(duplicate.rejected[0].reason, /already in the selection/);
  const removed = removeBatchItem(duplicate.items, 0);
  assert.equal(removed.items.length, 0);
  assert.equal(removed.removed.file, file);
});

test("continues after one failed request and records independent results", async () => {
  const { items } = addBatchFiles([], [image("ok.png"), image("fail.png"), image("ok2.png")], urls);
  const calls = [];
  await processBatch(items, "cable", async (_category, file) => {
    calls.push(file.name);
    if (file.name === "fail.png") throw new Error("mock failure");
    return { prediction_id: file.name, anomaly_score: 0.25, processing_time_ms: 4, result: {} };
  }, () => {});
  assert.deepEqual(calls, ["ok.png", "fail.png", "ok2.png"]);
  assert.deepEqual(items.map(item => item.status), ["completed", "failed", "completed"]);
  assert.equal(items[1].error, "mock failure");
  assert.equal(items[0].result.prediction_id, "ok.png");
});

test("marks every image failed if every mocked request fails", async () => {
  const { items } = addBatchFiles([], [image("a.png"), image("b.png")], urls);
  await processBatch(items, "cable", async () => { throw new Error("offline"); }, () => {});
  assert.deepEqual(items.map(item => item.status), ["failed", "failed"]);
});

test("allows a second batch after the first has finished", async () => {
  const { items } = addBatchFiles([], [image("again.png")], urls);
  let calls = 0;
  const analyze = async () => ({ prediction_id: String(++calls), anomaly_score: 0.1, result: {} });
  await processBatch(items, "wood", analyze, () => {});
  items[0].status = "waiting";
  items[0].result = null;
  await processBatch(items, "wood", analyze, () => {});
  assert.equal(calls, 2);
  assert.equal(items[0].status, "completed");
});

test("stops updating when the batch is no longer current", async () => {
  const { items } = addBatchFiles([], [image("a.png"), image("b.png")], urls);
  let current = true;
  let calls = 0;
  await processBatch(items, "wood", async () => { calls++; current = false; return {}; }, () => {}, () => current);
  assert.equal(calls, 1);
  assert.deepEqual(items.map(item => item.status), ["completed", "waiting"]);
});
