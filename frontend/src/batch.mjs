export const MAX_BATCH_IMAGES = 10;

export function validateImageFile(file) {
  if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) return "Choose a JPG, PNG or WEBP image.";
  if (file.size > 10 * 1024 * 1024) return "This file exceeds the 10 MB upload limit.";
  if (!file.size) return "This file is empty.";
  return "";
}

function fileKey(file) { return `${file.name}\u0000${file.size}\u0000${file.lastModified}`; }

export function addBatchFiles(selected, files, createPreviewUrl) {
  const items = [...selected];
  const rejected = [];
  const known = new Set(items.map(item => fileKey(item.file)));
  for (const file of files) {
    let reason = validateImageFile(file);
    if (!reason && known.has(fileKey(file))) reason = "This file is already in the selection.";
    if (!reason && items.length >= MAX_BATCH_IMAGES) reason = "The batch limit is 10 images. Remove an image before adding another.";
    if (reason) rejected.push({ name: file.name, reason });
    else {
      known.add(fileKey(file));
      items.push({ file, url: createPreviewUrl(file), status: "waiting", result: null, error: "" });
    }
  }
  return { items, rejected };
}

export function removeBatchItem(items, index) {
  const next = [...items];
  const [removed] = next.splice(index, 1);
  return { items: next, removed };
}

export async function processBatch(items, category, analyze, onUpdate, isCurrent = () => true) {
  for (const item of items) {
    if (!isCurrent()) return;
    item.status = "processing";
    onUpdate(item);
    try {
      item.result = await analyze(category, item.file);
      item.status = "completed";
    } catch (error) {
      item.status = "failed";
      item.error = error?.message || "Inspection failed.";
    }
    if (!isCurrent()) return;
    onUpdate(item);
  }
}
