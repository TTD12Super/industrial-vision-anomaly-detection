"""Download a small public image set and evaluate the product classifier.

The downloaded files are kept outside Git by default:

    DAenv/bin/python reports/external_classification_eval/test_external_images.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any
import urllib.parse
import urllib.request

from PIL import Image, ImageOps
import torch

from mvtec_system import SharedDINOv2, image_tensor, route_product


CATEGORIES = {
    "bottle": "Bottles",
    "cable": "Electrical cables",
    "capsule": "Capsules",
    "carpet": "Carpets",
    "grid": "Grids",
    "hazelnut": "Hazelnuts",
    "leather": "Leather",
    "metal_nut": "Nuts (hardware)",
    "pill": "Pills",
    "screw": "Screws",
    "tile": "Tiles",
    "toothbrush": "Toothbrushes",
    "transistor": "Transistors",
    "wood": "Wood",
    "zipper": "Zippers",
}
IMAGE_MIMES = {"image/jpeg", "image/png"}
USER_AGENT = "industrial-vision-anomaly-detection-external-eval/1.0"


def api_request(parameters: dict[str, Any]) -> dict[str, Any]:
    """Call the Wikimedia Commons API with a small retry backoff."""

    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(parameters)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.load(response)
            time.sleep(1.5)
            return data
        except Exception as error:  # network failures are retried, then reported
            last_error = error
            time.sleep(2**attempt)
    raise RuntimeError(f"Wikimedia API request failed: {last_error}")


def find_category_title(category_name: str) -> str:
    """Resolve a human category name to a Commons category page."""

    data = api_request(
        {
            "action": "query",
            "format": "json",
            "list": "search",
            "srnamespace": 14,
            "srsearch": category_name,
            "srlimit": 5,
        }
    )
    expected = "Category:" + category_name
    titles = [item["title"] for item in data.get("query", {}).get("search", [])]
    for title in titles:
        if title.casefold() == expected.casefold():
            return title
    if titles:
        return titles[0]
    raise ValueError(f"No Commons category found for {category_name!r}.")


def category_candidates(category_name: str, limit: int = 40) -> list[dict[str, Any]]:
    """List usable raster files from a Commons category."""

    category_title = find_category_title(category_name)
    category_data = api_request(
        {
            "action": "query",
            "format": "json",
            "generator": "categorymembers",
            "gcmtitle": category_title,
            "gcmnamespace": 6,
            "gcmtype": "file",
            "gcmlimit": limit,
            "prop": "imageinfo",
            "iiprop": "url|mime|size|extmetadata",
            "iiurlwidth": 640,
        }
    )
    candidates = []

    def add_pages(pages: Any) -> None:
        for page in pages:
            info = page.get("imageinfo", [{}])[0]
            title = page.get("title", "")
            mime = info.get("mime")
            if mime not in IMAGE_MIMES or info.get("width", 0) < 128 or info.get("height", 0) < 128:
                continue
            lowered = title.casefold()
            if any(word in lowered for word in ("diagram", "logo", "icon", "flag", "collage", "map")):
                continue
            metadata = info.get("extmetadata", {})
            candidates.append(
                {
                    "title": title,
                    "url": info.get("thumburl", info.get("url")),
                    "source_page": "https://commons.wikimedia.org/wiki/"
                    + urllib.parse.quote(title.replace(" ", "_")),
                    "mime": mime,
                    "license": metadata.get("LicenseShortName", {}).get("value", "unknown"),
                }
            )

    add_pages(category_data.get("query", {}).get("pages", {}).values())
    if len(candidates) < 2:
        search_data = api_request(
            {
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrsearch": category_name,
                "gsrnamespace": 6,
                "gsrlimit": limit,
                "prop": "imageinfo",
                "iiprop": "url|mime|size|extmetadata",
                "iiurlwidth": 640,
            }
        )
        add_pages(search_data.get("query", {}).get("pages", {}).values())
    unique = []
    seen_titles = set()
    for candidate in candidates:
        if candidate["title"] not in seen_titles:
            seen_titles.add(candidate["title"])
            unique.append(candidate)
    return unique


def download_image(candidate: dict[str, Any], output_path: Path) -> None:
    request = urllib.request.Request(candidate["url"], headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        output_path.write_bytes(response.read())


def prepare_image(source_path: Path, destination_path: Path) -> None:
    """Convert arbitrary source aspect ratios to the model's 256x256 input."""

    with Image.open(source_path) as image:
        prepared = ImageOps.fit(image.convert("RGB"), (256, 256), method=Image.Resampling.LANCZOS)
        prepared.save(destination_path, format="PNG")


def collect_images(output_dir: Path, per_category: int) -> list[dict[str, Any]]:
    """Download and normalize exactly the requested number per category."""

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for category, category_name in CATEGORIES.items():
        category_dir = images_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        candidates = category_candidates(category_name)
        if len(candidates) < per_category:
            raise RuntimeError(f"Only {len(candidates)} usable images found for {category}.")
        selected = candidates[:per_category]
        for index, candidate in enumerate(selected, start=1):
            raw_path = category_dir / f"{index:02d}_raw"
            prepared_path = category_dir / f"{index:02d}.png"
            download_image(candidate, raw_path)
            prepare_image(raw_path, prepared_path)
            raw_path.unlink()
            records.append(
                {
                    "category": category,
                    "path": str(prepared_path),
                    "source_title": candidate["title"],
                    "source_page": candidate["source_page"],
                    "source_url": candidate["url"],
                    "license": candidate["license"],
                }
            )
        print(category, len(selected), flush=True)
    (output_dir / "sources.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return records


def evaluate_images(output_dir: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Run raw classifier and calibrated router on downloaded images."""

    artifact_dir = Path("artifacts/product_classifier/v1")
    classifier = torch.load(artifact_dir / "classifier.pt", map_location="cpu", weights_only=True)
    reference = torch.load(artifact_dir / "train_embeddings.pt", map_location="cpu", weights_only=True)
    mapping = json.loads((artifact_dir / "class_mapping.json").read_text(encoding="utf-8"))["category_to_index"]
    calibration = json.loads((artifact_dir / "classification_calibration.json").read_text(encoding="utf-8"))
    encoder_metadata = json.loads((artifact_dir / "embedding_metadata.json").read_text(encoding="utf-8"))["encoder"]
    encoder = SharedDINOv2()
    results = []
    for record in records:
        tensor = image_tensor(record["path"], encoder.config)
        embedding = encoder.forward(tensor.unsqueeze(0)).global_embedding.cpu()[0]
        route = route_product(
            embedding,
            classifier,
            reference,
            mapping,
            calibration,
            "artifacts/dinov2_knn/final_run",
            encoder_metadata,
        )
        results.append(
            {
                **record,
                "predicted_category": route.predicted_category,
                "router_category": route.category,
                "confidence": route.confidence,
                "nearest_global_distance": route.nearest_global_distance,
                "rejected": route.rejected,
                "bank_status": route.bank_status,
                "raw_correct": route.predicted_category == record["category"],
                "router_correct": route.category == record["category"],
            }
        )
        print(record["category"], Path(record["path"]).name, route.category, flush=True)
    raw_accuracy = sum(item["raw_correct"] for item in results) / len(results)
    router_accuracy = sum(item["router_correct"] for item in results) / len(results)
    summary = {
        "image_count": len(results),
        "images_per_category": len(results) // len(CATEGORIES),
        "raw_classifier_accuracy": raw_accuracy,
        "router_accuracy_including_reject": router_accuracy,
        "reject_count": sum(item["rejected"] for item in results),
        "reject_rate": sum(item["rejected"] for item in results) / len(results),
        "source": "Wikimedia Commons category images",
        "note": "External images were evaluated only; they were not used for training or threshold selection.",
    }
    (output_dir / "results.json").write_text(
        json.dumps({"summary": summary, "images": results}, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/external_classification_eval"))
    parser.add_argument("--per-category", type=int, default=2)
    parser.add_argument("--redownload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.per_category < 1:
        raise ValueError("--per-category must be positive")
    sources_path = args.output_dir / "sources.json"
    if args.redownload or not sources_path.is_file():
        records = collect_images(args.output_dir, args.per_category)
    else:
        records = json.loads(sources_path.read_text(encoding="utf-8"))
    summary = evaluate_images(args.output_dir, records)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
