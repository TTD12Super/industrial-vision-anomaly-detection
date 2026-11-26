"""LOO calibration tests (require torch/Pillow/numpy; synthetic banks only).

Nothing here touches the real MVTec dataset or the real feature banks: every
fixture is built in a temporary directory from random tensors and tiny PNGs.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

ROOT = Path(__file__).resolve().parents[1]
PATCH_COUNT = 361
DIM = 8


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tiny_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8), 128, dtype=np.uint8), mode="L").save(path)


def _make_processed_root(root: Path, count: int = 3) -> list[str]:
    """Create ``bottle/train/good/*.png`` and return evaluator-style rel paths."""
    rel_paths = []
    for index in range(count):
        rel = Path("bottle") / "train" / "good" / f"{index:03d}.png"
        _tiny_png(root / rel)
        rel_paths.append(str(rel))
    return rel_paths


def _save_bank(root: Path, rel_paths, features: torch.Tensor, *, swap=None, paths=None) -> Path:
    """Persist a synthetic bank with metadata matching the real convention."""
    from model.inference.feature_bank import bank_metadata, save_feature_bank

    stored = list(paths) if paths is not None else list(rel_paths)
    if swap is not None:
        i, j = swap
        stored[i], stored[j] = stored[j], stored[i]
    metadata = bank_metadata(
        "bottle",
        embedding_dimension=int(features.shape[1]),
        patch_grid=(19, 19),
        padding=(10, 10),
        train_paths=stored,
    )
    destination = root / "banks" / "bottle" / "feature_bank.pt"
    save_feature_bank(destination, features, metadata)
    return destination


def _random_bank(m: int, patch_count: int = PATCH_COUNT, dim: int = DIM, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(m, patch_count, dim, generator=generator)
    return images.reshape(m * patch_count, dim), images


def _brute_patch_scores(bank: torch.Tensor, image_index: int, patch_count: int) -> torch.Tensor:
    start = image_index * patch_count
    stop = start + patch_count
    queries = bank[start:stop]
    allowed = torch.cat((bank[:start], bank[stop:]), dim=0)
    return torch.cdist(queries, allowed).min(dim=1).values


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "calibrate_thresholds_cli", ROOT / "scripts" / "calibrate_thresholds.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(not (HAS_TORCH and HAS_NUMPY), "torch/numpy required")
class TestLooExclusion(unittest.TestCase):
    """Excluded rows must never be selectable as nearest neighbours."""

    def test_excluded_rows_cannot_be_selected(self):
        from model.evaluation.calibration import loo_patch_scores

        generator = torch.Generator().manual_seed(1)
        base = torch.zeros(1, DIM)
        near = base + 0.01 * torch.randn(1, PATCH_COUNT, DIM, generator=generator)
        far = base + 5.0 + 1.0 * torch.randn(2, PATCH_COUNT, DIM, generator=generator)
        images = torch.cat((near, far), dim=0)
        bank = images.reshape(-1, DIM)

        scores = loo_patch_scores(bank, [0], PATCH_COUNT)
        self.assertEqual(tuple(scores.shape), (1, PATCH_COUNT))
        self.assertTrue(torch.isfinite(scores).all().item())
        # Its own rows are within ~0.05; every allowed row is far away.
        self.assertGreater(float(scores.min()), 5.0)
        unrestricted = torch.cdist(bank[:PATCH_COUNT], bank).min(dim=1).values
        self.assertLess(float(unrestricted.max()), 0.2)

    def test_all_361_own_rows_excluded_not_just_one_patch_row(self):
        from model.evaluation.calibration import loo_patch_scores

        bank, _ = _random_bank(3, PATCH_COUNT, DIM, seed=2)
        for image_index in range(3):
            actual = loo_patch_scores(bank, [image_index], PATCH_COUNT)[0]
            expected = _brute_patch_scores(bank, image_index, PATCH_COUNT)
            self.assertTrue(
                torch.allclose(actual, expected, atol=1e-5),
                f"image {image_index}: LOO patch scores must exclude the whole block",
            )
            # Every own row is the exact self-match (distance 0), so a score of 0
            # anywhere would mean a row of the image's own block was still eligible.
            self.assertGreater(float(actual.min()), 1e-4)

    def test_loo_matches_brute_force_small_synthetic_tensor(self):
        from model.evaluation.calibration import loo_image_scores

        generator = torch.Generator().manual_seed(3)
        m, patch_count, dim = 4, 9, 5
        bank = torch.randn(m * patch_count, dim, generator=generator)
        expected = []
        for i in range(m):
            query = bank[i * patch_count : (i + 1) * patch_count]
            allowed = torch.cat((bank[: i * patch_count], bank[(i + 1) * patch_count :]))
            expected.append(float(torch.cdist(query, allowed).min(dim=1).values.max()))
        for images_per_chunk, bank_chunk in ((1, 4), (3, 7), (4, 1000)):
            actual = loo_image_scores(
                bank,
                list(range(m)),
                patch_count,
                bank_chunk_size=bank_chunk,
            )
            np.testing.assert_allclose(actual, expected, atol=1e-5)

    def test_own_image_block_is_the_excluded_block(self):
        """Exclusion follows row mapping i -> [i*361, (i+1)*361)."""
        from model.evaluation.calibration import loo_patch_scores

        bank, _ = _random_bank(3, PATCH_COUNT, DIM, seed=4)
        for i in range(3):
            per_image = loo_patch_scores(bank, [i], PATCH_COUNT)
            # Scores must equal a reference that removes exactly rows
            # [i*361, (i+1)*361) and nothing else.
            kept = torch.cat((bank[: i * PATCH_COUNT], bank[(i + 1) * PATCH_COUNT :]), dim=0)
            reference = torch.cdist(bank[i * PATCH_COUNT : (i + 1) * PATCH_COUNT], kept)
            reference = reference.min(dim=1).values
            self.assertTrue(torch.allclose(per_image[0], reference, atol=1e-5))


@unittest.skipIf(not (HAS_TORCH and HAS_NUMPY), "torch/numpy required")
class TestMetadataAndMappingValidation(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="calib_meta_"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.processed = self.root / "processed"
        self.rel_paths = _make_processed_root(self.processed, count=3)

    def _validate(self, bank, metadata):
        from model.evaluation.calibration import validate_bank_for_calibration

        return validate_bank_for_calibration("bottle", bank, metadata, self.processed)

    def test_valid_metadata_returns_image_count(self):
        from model.inference.feature_bank import load_feature_bank

        bank, _ = _random_bank(3)
        destination = _save_bank(self.root, self.rel_paths, bank)
        loaded, metadata = load_feature_bank(destination)
        self.assertEqual(self._validate(loaded, metadata), 3)

    def test_reordered_train_paths_rejected(self):
        from model.inference.feature_bank import load_feature_bank

        bank, _ = _random_bank(3)
        destination = _save_bank(self.root, self.rel_paths, bank, swap=(0, 1))
        loaded, metadata = load_feature_bank(destination)
        with self.assertRaises(ValueError) as ctx:
            self._validate(loaded, metadata)
        self.assertIn("element by element", str(ctx.exception))

    def test_paths_outside_train_good_rejected(self):
        from model.evaluation.calibration import assert_paths_under_train_good

        with self.assertRaises(ValueError) as ctx:
            assert_paths_under_train_good(
                self.processed, "bottle", ["bottle/test/good/000.png"]
            )
        self.assertIn("test", str(ctx.exception))
        with self.assertRaises(ValueError):
            assert_paths_under_train_good(
                self.processed, "bottle", ["bottle/ground_truth/x_mask.png"]
            )
        with self.assertRaises(ValueError):
            assert_paths_under_train_good(self.processed, "bottle", ["../outside.png"])

    def test_row_count_must_be_images_times_361(self):
        from model.inference.feature_bank import bank_metadata

        good, _ = _random_bank(3)
        metadata = bank_metadata(
            "bottle",
            embedding_dimension=DIM,
            patch_grid=(19, 19),
            padding=(10, 10),
            train_paths=list(self.rel_paths),
        )
        short = good[:-5]
        with self.assertRaises(ValueError) as ctx:
            self._validate(short, metadata)
        self.assertIn("3 images x 361", str(ctx.exception))

    def test_wrong_model_revision_rejected(self):
        from model.inference.feature_bank import bank_metadata

        bank, _ = _random_bank(3)
        metadata = bank_metadata(
            "bottle",
            embedding_dimension=DIM,
            patch_grid=(19, 19),
            padding=(10, 10),
            train_paths=list(self.rel_paths),
        )
        metadata["model_revision"] = "not-the-pinned-revision"
        with self.assertRaises(ValueError) as ctx:
            self._validate(bank, metadata)
        self.assertIn("model_revision", str(ctx.exception))


@unittest.skipIf(not (HAS_TORCH and HAS_NUMPY), "torch/numpy required")
class TestThresholdFormula(unittest.TestCase):
    def test_index_for_m60_alpha_005(self):
        from model.evaluation.calibration import order_statistic_threshold

        scores = [float(v) for v in range(60)]
        result = order_statistic_threshold(scores, 0.05)
        self.assertEqual(result.index_k, 58)
        self.assertEqual(math.ceil((60 + 1) * (1 - 0.05)), 58)
        self.assertTrue(result.feasible)
        self.assertEqual(result.threshold, 57.0)  # 58th smallest, 1-based
        self.assertEqual(result.sample_count, 60)
        self.assertEqual(result.alpha, 0.05)
        payload = result.as_dict()
        self.assertIn("ceil((m + 1) * (1 - alpha))", payload["formula"])
        self.assertIn("score > threshold", payload["decision_rule"])

    def test_k_greater_than_m_gives_infinity_without_relaxing_alpha(self):
        from model.evaluation.calibration import order_statistic_threshold

        result = order_statistic_threshold([float(v) for v in range(10)], 0.05)
        self.assertEqual(result.index_k, 11)
        self.assertEqual(result.index_k, math.ceil((10 + 1) * (1 - 0.05)))
        self.assertTrue(math.isinf(result.threshold))
        self.assertFalse(result.feasible)
        self.assertEqual(result.alpha, 0.05)
        self.assertIn("alpha was not relaxed", result.note)
        self.assertIn("no rejections", result.note)

    def test_alpha_validation(self):
        from model.evaluation.calibration import order_statistic_threshold

        for alpha in (0.0, 1.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                order_statistic_threshold([1.0, 2.0], alpha)
        with self.assertRaises(ValueError):
            order_statistic_threshold([], 0.05)
        with self.assertRaises(ValueError):
            order_statistic_threshold([1.0, float("nan")], 0.05)

    def test_threshold_is_monotone_in_alpha(self):
        from model.evaluation.calibration import order_statistic_threshold

        scores = [float(v) for v in range(60)]
        loose = order_statistic_threshold(scores, 0.5)
        strict = order_statistic_threshold(scores, 0.05)
        self.assertEqual(loose.index_k, math.ceil(61 * 0.5))
        self.assertLessEqual(loose.threshold, strict.threshold)


@unittest.skipIf(not (HAS_TORCH and HAS_NUMPY), "torch/numpy required")
class TestMonotonicityAndDeterminism(unittest.TestCase):
    def test_adding_reference_rows_cannot_increase_score(self):
        from model.evaluation.calibration import loo_image_scores

        bank, _ = _random_bank(4, PATCH_COUNT, DIM, seed=5)
        generator = torch.Generator().manual_seed(6)
        extra = torch.randn(37, DIM, generator=generator)
        bigger = torch.cat((bank, extra), dim=0)
        base = loo_image_scores(bank, [0, 1, 2, 3], PATCH_COUNT, bank_chunk_size=128)
        grown = loo_image_scores(bigger, [0, 1, 2, 3], PATCH_COUNT, bank_chunk_size=128)
        self.assertTrue(
            np.all(grown <= base + 1e-6),
            "monotonicity lemma violated: extra reference rows raised a score",
        )

    def test_calibration_is_deterministic(self):
        from model.evaluation.calibration import loo_image_scores, order_statistic_threshold

        bank, _ = _random_bank(3, PATCH_COUNT, DIM, seed=7)
        first = loo_image_scores(bank, [0, 1, 2], PATCH_COUNT, bank_chunk_size=100)
        second = loo_image_scores(bank, [0, 1, 2], PATCH_COUNT, bank_chunk_size=100)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(
            order_statistic_threshold(first, 0.05).threshold,
            order_statistic_threshold(second, 0.05).threshold,
        )

    def test_image_score_is_max_over_patches(self):
        from model.evaluation.calibration import loo_image_scores, loo_patch_scores

        bank, _ = _random_bank(2, PATCH_COUNT, DIM, seed=8)
        images = loo_image_scores(bank, [0, 1], PATCH_COUNT)
        patches = loo_patch_scores(bank, [0, 1], PATCH_COUNT)
        np.testing.assert_allclose(images, patches.max(dim=1).values.numpy(), atol=1e-6)


@unittest.skipIf(not HAS_TORCH, "torch required")
class TestProductionScoringUnchanged(unittest.TestCase):
    def test_nearest_neighbor_scores_signature_and_numbers(self):
        import inspect

        from model.inference.knn import nearest_neighbor_scores

        signature = inspect.signature(nearest_neighbor_scores)
        self.assertEqual(
            list(signature.parameters),
            ["queries", "bank", "k", "query_chunk_size", "bank_chunk_size"],
        )
        generator = torch.Generator().manual_seed(9)
        bank = torch.randn(64, DIM, generator=generator)
        queries = torch.randn(17, DIM, generator=generator)
        expected = torch.cdist(queries, bank).topk(1, largest=False).values.flatten()
        for query_chunk, bank_chunk in ((512, 16384), (5, 7)):
            actual = nearest_neighbor_scores(
                queries, bank, 1, query_chunk_size=query_chunk, bank_chunk_size=bank_chunk
            )
            self.assertTrue(torch.allclose(actual, expected, atol=1e-5))

    def test_knn_module_has_no_exclusion_logic(self):
        source = (ROOT / "model" / "inference" / "knn.py").read_text(encoding="utf-8")
        self.assertNotIn("excluded", source)
        self.assertNotIn("loo", source.lower().replace("loop", ""))


class TestCliHasNoTestInputs(unittest.TestCase):
    def test_cli_exposes_no_test_or_evaluation_options(self):
        cli = _load_cli()
        args = cli.parse_args(["--categories", "bottle"])
        forbidden = ("test", "eval", "label", "mask", "ground", "score")
        for name in vars(args):
            self.assertFalse(
                any(token in name for token in forbidden),
                f"CLI option {name!r} looks like a test/evaluation input",
            )
        self.assertEqual(args.categories, ["bottle"])
        self.assertIsNone(args.alpha)
        self.assertTrue(cli.parse_args(["--all"]).all)

    def test_cli_rejects_test_and_evaluation_flags(self):
        cli = _load_cli()
        for argv in (
            ["--categories", "bottle", "--test-dir", "x"],
            ["--categories", "bottle", "--eval-run-dir", "x"],
            ["--categories", "bottle", "--labels", "x"],
            ["--categories", "bottle", "--ground-truth", "x"],
        ):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    cli.parse_args(argv)

    def test_cli_requires_category_selection(self):
        cli = _load_cli()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                cli.parse_args([])


@unittest.skipIf(not (HAS_TORCH and HAS_NUMPY), "torch/numpy required")
class TestArtifactsAndIntegrity(unittest.TestCase):
    """Full run_calibration on a synthetic fixture (no real dataset)."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="calib_run_"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.processed = self.root / "processed"
        self.rel_paths = _make_processed_root(self.processed, count=3)
        features, _ = _random_bank(3)
        self.bank_path = _save_bank(self.root, self.rel_paths, features)
        self.bank_dir = self.root / "banks"
        self.run_root = self.root / "runs"

    def _run(self, alpha: float, name: str) -> tuple[Path, list]:
        from model.evaluation.calibration import run_calibration

        return run_calibration(
            categories=["bottle"],
            processed_root=self.processed,
            bank_dir=self.bank_dir,
            run_root=self.run_root,
            alpha=alpha,
        )

    def test_run_writes_required_artifacts_without_touching_bank(self):
        before = _sha256(self.bank_path)
        bank_dir_files = sorted(p.name for p in (self.bank_dir / "bottle").iterdir())
        run_dir, results = self._run(0.5, "feasible")
        self.assertEqual(_sha256(self.bank_path), before, "bank file was modified")
        self.assertEqual(
            sorted(p.name for p in (self.bank_dir / "bottle").iterdir()), bank_dir_files
        )
        for name in (
            "calibration_config.json",
            "thresholds.csv",
            "bank_hashes_before.json",
            "bank_hashes_after.json",
            "assumptions_and_proof.md",
            "calibration_scores/bottle.csv",
        ):
            self.assertTrue((run_dir / name).is_file(), f"missing {name}")
        self.assertEqual(len(results), 1)
        threshold = results[0].threshold
        self.assertTrue(threshold.feasible)
        self.assertEqual(threshold.sample_count, 3)
        self.assertEqual(threshold.index_k, math.ceil((3 + 1) * (1 - 0.5)))

        before_hashes = json.loads((run_dir / "bank_hashes_before.json").read_text())
        after_hashes = json.loads((run_dir / "bank_hashes_after.json").read_text())
        self.assertEqual(before_hashes, after_hashes)
        self.assertEqual(before_hashes["bottle"], _sha256(self.bank_path))
        self.assertEqual(len(before_hashes), 15)
        self.assertIsNone(before_hashes["cable"])

    def test_artifacts_document_bound_and_assumptions(self):
        run_dir, _ = self._run(0.05, "infeasible")  # m=3 -> k=4 > 3 -> +inf
        config = json.loads((run_dir / "calibration_config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["alpha"], 0.05)
        self.assertEqual(config["model_revision"], "7764ea0f912e53c92e82eb78a2a1631e92725fc8")
        self.assertIn("ceil((m + 1) * (1 - alpha))", config["threshold_formula"])
        self.assertIn("score > threshold", config["decision_rule"])
        self.assertFalse(config["bound"]["is_split_conformal"])
        self.assertFalse(config["bound"]["is_conditional"])
        self.assertEqual(len(config["bound"]["assumptions"]), 7)
        self.assertTrue(config["bank_hashes_unchanged"])
        self.assertIn("torch", config["packages"])
        self.assertIn("numpy", config["packages"])
        self.assertTrue(config["config_hash"])
        self.assertIn("train/good", config["calibration_data_statement"])
        row = config["calibration_categories"][0]
        self.assertEqual(row["index_k"], 4)
        self.assertFalse(row["alpha_feasible"])
        self.assertTrue(math.isinf(row["threshold"]))
        self.assertIn("alpha was not relaxed", row["note"])

        doc = (run_dir / "assumptions_and_proof.md").read_text(encoding="utf-8").lower()
        for phrase in (
            "marginal",
            "not conditional",
            "not split-conformal",
            "exchangeable",
            "monotonicity",
            "score > threshold",
            "normal images only",
            "no guarantee under distribution shift",
        ):
            self.assertIn(phrase, doc, f"proof document must state: {phrase}")

        thresholds = (run_dir / "thresholds.csv").read_text(encoding="utf-8")
        self.assertIn("score > threshold", thresholds)
        self.assertIn("bottle,3,0.05,4,inf,false", thresholds.replace("\r\n", "\n"))

    def test_artifacts_contain_no_test_paths_or_labels(self):
        run_dir, _ = self._run(0.5, "paths")
        for path in run_dir.rglob("*"):
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            self.assertNotIn("ground_truth", text, str(path))
            self.assertNotIn("/test/", text, str(path))
            self.assertNotIn("\\test\\", text, str(path))
            self.assertNotIn("is_anomaly", text, str(path))
        scores_csv = (run_dir / "calibration_scores" / "bottle.csv").read_text(encoding="utf-8")
        for line in scores_csv.splitlines()[1:]:
            image_path = line.split(",")[1]
            self.assertTrue(image_path.startswith("bottle"), image_path)
            self.assertIn("train", image_path.split("/")[1:] + image_path.split("\\"))

    def test_repeated_runs_are_deterministic(self):
        first, _ = self._run(0.5, "a")
        second, _ = self._run(0.5, "b")
        self.assertNotEqual(first, second, "each run must get a unique directory")
        self.assertEqual(
            (first / "calibration_scores" / "bottle.csv").read_bytes(),
            (second / "calibration_scores" / "bottle.csv").read_bytes(),
        )
        first_rows = (first / "thresholds.csv").read_text().splitlines()
        second_rows = (second / "thresholds.csv").read_text().splitlines()
        # Only the measured seconds column may differ.
        strip = lambda rows: [",".join(row.split(",")[:-1]) for row in rows]
        self.assertEqual(strip(first_rows), strip(second_rows))

    def test_run_refuses_to_overwrite_existing_directory(self):
        from unittest import mock

        import model.evaluation.calibration as calibration

        from model.evaluation.calibration import run_calibration

        real_datetime = calibration.datetime

        class FixedDateTime:
            @classmethod
            def now(cls, tz=None):
                return real_datetime(2026, 1, 2, 3, 4, 5, 6, tzinfo=tz)

        with mock.patch.object(calibration, "datetime", FixedDateTime):
            run_dir, _ = self._run(0.5, "fixed")
            self.assertTrue(run_dir.is_dir())
            with self.assertRaises(FileExistsError):
                run_calibration(
                    categories=["bottle"],
                    processed_root=self.processed,
                    bank_dir=self.bank_dir,
                    run_root=run_dir.parent,
                    alpha=0.5,
                )
        self.assertTrue((run_dir / "calibration_config.json").is_file())


if __name__ == "__main__":
    unittest.main()
