"""Small result renderer for the egg demo."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw

from .pipeline import EggInference


def draw_results(image_path: str | Path, results: Sequence[EggInference], output_path: str | Path) -> None:
    """Save source image with one box and anomaly label per detected egg."""

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    for result in results:
        left, top, right, bottom = result.detection.xyxy
        color = "red" if result.anomaly.label == "DEFECT" else "lime"
        draw.rectangle((left, top, right, bottom), outline=color, width=max(2, image.width // 500))
        text = f"{result.anomaly.label} {result.anomaly.score:.3f}"
        draw.text((left, max(0, top - 18)), text, fill=color)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
