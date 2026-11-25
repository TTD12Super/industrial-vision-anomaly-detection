from __future__ import annotations

import math
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import train_screw_rtdetr  # noqa: E402
from train_screw_rtdetr import corners_to_normalized_xywh  # noqa: E402


def test_obb_corners_convert_to_enclosing_normalized_aabb() -> None:
    xywh = corners_to_normalized_xywh(
        [(10, 20), (50, 10), (60, 40), (20, 50)],
        width=100,
        height=100,
    )

    assert all(math.isclose(actual, expected) for actual, expected in zip(xywh, (0.35, 0.30, 0.50, 0.40)))


def test_aabb_conversion_clips_outside_corners_to_image() -> None:
    xywh = corners_to_normalized_xywh(
        [(-10, -5), (110, -5), (110, 105), (-10, 105)],
        width=100,
        height=100,
    )

    assert xywh == (0.5, 0.5, 1.0, 1.0)


def test_aabb_conversion_rejects_invalid_corners() -> None:
    with pytest.raises(ValueError, match="four finite"):
        corners_to_normalized_xywh([(0, 0), (1, 1), (2, 2)], width=10, height=10)


def test_training_paths_are_absolute_and_checkpoint_consistent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    class FakeRTDETR:
        def __init__(self, model: str) -> None:
            observed["model"] = model

        def train(self, **options: object) -> SimpleNamespace:
            observed.update(options)
            save_directory = Path(str(options["project"])) / str(options["name"])
            checkpoint = save_directory / "weights" / "best.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            return SimpleNamespace(save_dir=save_directory)

    def fake_materialize(_config: object, output_root: str | Path) -> dict[str, str]:
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        return {"data_yaml": str(output_root / "data.yaml")}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train_screw_rtdetr, "materialize_rtdetr_dataset", fake_materialize)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(RTDETR=FakeRTDETR))

    result = train_screw_rtdetr.train_rtdetr(object(), Path("relative-output"), epochs=1, image_size=640, batch=1)

    expected_project = (tmp_path / "relative-output" / "training").resolve()
    expected_checkpoint = expected_project / "run" / "weights" / "best.pt"
    assert Path(str(observed["project"])) == expected_project
    assert result["checkpoint"] == str(expected_checkpoint)
