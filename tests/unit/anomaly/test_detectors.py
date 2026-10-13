"""Contract tests for the selectable anomaly detector registry."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch

from mvtec_system import (
    DINOv2Config,
    DINOv2Features,
    MultiLayerConfig,
    PaDiMDetector,
    PatchCoreDetector,
    create_detector,
    detector_names,
)


class TinyMultiLayerEncoder:
    def __init__(self) -> None:
        self.config = DINOv2Config(image_size=(16, 16), feature_layers=(6, 9, 12))
        self.patch_grid = (2, 2)
        self.padded_size = (16, 16)

    def forward(self, images: torch.Tensor) -> DINOv2Features:
        batch = images.shape[0]
        base = torch.arange(4 * 2, dtype=torch.float32).reshape(1, 4, 2)
        base = base.expand(batch, -1, -1)
        layers = {layer: base + layer for layer in (6, 9, 12)}
        return DINOv2Features(base[:, 0], base, layers)


class DetectorTests(unittest.TestCase):
    def paths(self) -> tuple[Path, list[Path]]:
        root = Path(tempfile.mkdtemp())
        paths = []
        for index in range(3):
            path = root / f"{index}.png"
            Image.new("RGB", (16, 16), color=(index * 20, 30, 40)).save(path)
            paths.append(path)
        return root, paths

    def test_registry_contains_all_three_detectors(self) -> None:
        self.assertEqual(detector_names(), ("exact_knn", "padim", "patchcore"))
        self.assertEqual(create_detector("patchcore").name, "patchcore")
        self.assertEqual(create_detector("padim").name, "padim")

    def test_patchcore_fit_predict_save_load(self) -> None:
        _, paths = self.paths()
        encoder = TinyMultiLayerEncoder()
        detector = PatchCoreDetector(
            MultiLayerConfig(
                layers=(6, 9, 12),
                neighborhood_size=3,
                coreset_ratio=0.5,
                coreset_max_points=10,
            ),
            k=1,
        )
        detector.fit("part", paths, encoder, batch_size=2)
        features = encoder.forward(torch.zeros(1, 3, 16, 16))
        result = detector.predict(features)
        self.assertEqual(result.anomaly_map.shape, (16, 16))
        self.assertEqual(result.patch_scores.shape, (4,))
        with tempfile.TemporaryDirectory() as directory:
            detector.save(directory)
            loaded = PatchCoreDetector(detector.config, k=1).load(directory)
            self.assertEqual(loaded.predict(features).patch_scores.shape, (4,))

    def test_padim_fit_predict_save_load(self) -> None:
        _, paths = self.paths()
        encoder = TinyMultiLayerEncoder()
        detector = PaDiMDetector(
            MultiLayerConfig(
                layers=(6, 9, 12),
                reduced_dim=3,
                covariance_regularization=0.01,
            )
        )
        detector.fit("part", paths, encoder, batch_size=2)
        features = encoder.forward(torch.zeros(1, 3, 16, 16))
        result = detector.predict(features)
        self.assertEqual(result.anomaly_map.shape, (16, 16))
        self.assertEqual(result.patch_scores.shape, (4,))
        with tempfile.TemporaryDirectory() as directory:
            detector.save(directory)
            loaded = PaDiMDetector(detector.config).load(directory)
            self.assertEqual(loaded.predict(features).patch_scores.shape, (4,))


if __name__ == "__main__":
    unittest.main()
