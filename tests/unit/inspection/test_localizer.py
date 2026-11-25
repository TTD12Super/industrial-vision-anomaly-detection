import unittest

import numpy as np

from inspection.backends import AnomalyClipLocalizer
from inspection.dino_refinement import propagate_scores
from inspection.evaluation import _postprocessed_metrics, postprocess_mask, select_mask_threshold, select_postprocess_threshold
from inspection.supervised_refinement import SupervisedPatchMaskRefiner, patch_inputs
from inspection.synthetic_segmentation import SyntheticDefectSegmenter, generate_synthetic_anomaly


class AnomalyClipLocalizerTests(unittest.TestCase):
    def test_missing_configuration_is_explicit_not_a_fake_map(self) -> None:
        result = AnomalyClipLocalizer({}).localize(np.zeros((8, 8, 3), dtype=np.uint8))

        self.assertEqual(result.status, "unavailable")
        self.assertEqual(result.reason, "ANOMALYCLIP_NOT_CONFIGURED")
        self.assertIsNone(result.heatmap)

    def test_threshold_selection_meets_recall_and_prefers_clean_good_pixels(self) -> None:
        samples = [
            {"heatmap": np.array([[0.1, 0.2], [0.9, 0.8]], dtype=np.float32), "truth": np.array([[False, False], [True, True]])},
            {"heatmap": np.array([[0.1, 0.3], [0.2, 0.4]], dtype=np.float32), "truth": np.zeros((2, 2), dtype=bool)},
        ]

        threshold, metrics = select_mask_threshold(samples, target_recall=1.0)

        self.assertGreaterEqual(threshold, 0.4)
        self.assertEqual(metrics["pixel_recall"], 1.0)
        self.assertEqual(metrics["good_false_positive_rate"], 0.0)

    def test_good_false_positive_rate_uses_good_crops_when_labels_exist(self) -> None:
        samples = [
            {"heatmap": np.ones((2, 2), dtype=np.float32), "truth": np.zeros((2, 2), dtype=bool), "label": "GOOD"},
            {"heatmap": np.ones((2, 2), dtype=np.float32), "truth": np.ones((2, 2), dtype=bool), "label": "DEFECT"},
        ]

        metrics = _postprocessed_metrics(samples, 0.5, "raw")

        self.assertEqual(metrics["good_false_positive_rate"], 1.0)
        self.assertEqual(metrics["good_image_false_positive_rate"], 1.0)
        self.assertIn("pixel_specificity", metrics)

    def test_postprocess_methods_remove_small_components(self) -> None:
        heatmap = np.zeros((9, 9), dtype=np.float32)
        heatmap[1, 1] = 1.0
        heatmap[4:7, 4:7] = 1.0

        mask = postprocess_mask(heatmap, 0.5, "components")

        self.assertFalse(mask[1, 1])
        self.assertTrue(mask[5, 5])

    def test_heatmap_open_method_removes_small_high_score_regions(self) -> None:
        heatmap = np.zeros((9, 9), dtype=np.float32)
        heatmap[1, 1] = 1.0
        heatmap[3:7, 3:7] = 1.0

        mask = postprocess_mask(heatmap, 0.5, "heatmap_open3")

        self.assertFalse(mask[1, 1])
        self.assertTrue(mask[5, 5])

    def test_larger_heatmap_open_methods_are_supported(self) -> None:
        heatmap = np.zeros((15, 15), dtype=np.float32)
        heatmap[2:11, 2:11] = 1.0

        for method in ("heatmap_open7", "heatmap_open9"):
            mask = postprocess_mask(heatmap, 0.5, method)
            self.assertTrue(mask.any(), method)
            self.assertFalse(mask[0, 0], method)

    def test_dino_graph_propagation_preserves_shape_and_finiteness(self) -> None:
        scores = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        features = np.eye(3, dtype=np.float32)

        refined = propagate_scores(scores, features, top_k=2, temperature=0.1, eta=1.0)

        self.assertEqual(refined.shape, scores.shape)
        self.assertTrue(np.isfinite(refined).all())

    def test_supervised_refiner_returns_a_finite_image_map(self) -> None:
        dino_features = np.eye(4, dtype=np.float32)
        heatmap = np.asarray([[0.0, 0.2], [0.8, 1.0]], dtype=np.float32)
        features = patch_inputs(dino_features, heatmap, side=2)
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)

        refiner = SupervisedPatchMaskRefiner.fit(features, labels)
        refined = refiner.predict_map(features, side=2, output_size=(8, 8))

        self.assertEqual(refined.shape, (8, 8))
        self.assertTrue(np.isfinite(refined).all())
        self.assertGreater(float(refined.max()), float(refined.min()))

    def test_spatial_patch_inputs_add_finite_local_context(self) -> None:
        dino_features = np.eye(4, dtype=np.float32)
        heatmap = np.asarray([[0.0, 0.2], [0.8, 1.0]], dtype=np.float32)

        features = patch_inputs(dino_features, heatmap, side=2, feature_mode="spatial")

        self.assertEqual(features.shape, (4, 11))
        self.assertTrue(np.isfinite(features).all())

        refiner = SupervisedPatchMaskRefiner.fit(
            features,
            np.asarray([0, 0, 1, 1], dtype=np.int64),
            feature_mode="spatial",
        )
        self.assertEqual(refiner.feature_mode, "spatial")

    def test_mlp_supervised_refiner_returns_a_finite_image_map(self) -> None:
        rng = np.random.default_rng(42)
        features = rng.normal(size=(24, 4)).astype(np.float32)
        labels = np.asarray([0] * 12 + [1] * 12, dtype=np.int64)
        refiner = SupervisedPatchMaskRefiner.fit(features, labels, model="mlp")
        refined = refiner.predict_map(features[:4], side=2, output_size=(8, 8))

        self.assertEqual(refined.shape, (8, 8))
        self.assertTrue(np.isfinite(refined).all())

    def test_spatial_decoder_keeps_patch_grid_and_returns_finite_map(self) -> None:
        rng = np.random.default_rng(42)
        feature_maps = [rng.normal(size=(9, 4)).astype(np.float32) for _ in range(3)]
        label_maps = [np.asarray([0, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.int64) for _ in range(3)]

        refiner = SupervisedPatchMaskRefiner.fit_spatial_decoder(feature_maps, label_maps, epochs=2)
        refined = refiner.predict_map(feature_maps[0], side=3, output_size=(12, 12))

        self.assertEqual(refined.shape, (12, 12))
        self.assertTrue(np.isfinite(refined).all())
        self.assertGreater(float(refined.max()), float(refined.min()))

    def test_synthetic_anomaly_generator_and_segmenter_contract(self) -> None:
        base = np.full((32, 32, 3), 120, dtype=np.uint8)
        donor = np.full((32, 32, 3), 220, dtype=np.uint8)
        corrupted, mask = generate_synthetic_anomaly(base, donor, seed=42)

        self.assertEqual(corrupted.shape, base.shape)
        self.assertEqual(mask.shape, base.shape[:2])
        self.assertTrue(np.isfinite(corrupted).all())
        self.assertTrue(np.isfinite(mask).all())

        refiner = SyntheticDefectSegmenter.fit([base, donor], image_size=32, epochs=1, batch_size=2)
        refined = refiner.predict_map(base)
        self.assertEqual(refined.shape, base.shape[:2])
        self.assertTrue(np.isfinite(refined).all())
