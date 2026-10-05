"""Tests for non-invasive performance helpers."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from mvtec_system import available_backends, file_size_mb, measure_callable, memory_usage_mb


class PerformanceTests(unittest.TestCase):
    def test_backend_report_has_expected_keys(self) -> None:
        self.assertEqual(set(available_backends()), {"cpu", "mps", "cuda"})
        self.assertTrue(available_backends()["cpu"])

    def test_timer_and_file_size_are_finite(self) -> None:
        value, seconds = measure_callable(lambda: 2 + 2)
        self.assertEqual(value, 4)
        self.assertGreaterEqual(seconds, 0.0)
        self.assertGreater(memory_usage_mb(), 0.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "small.bin"
            path.write_bytes(b"x" * 1024)
            self.assertGreater(file_size_mb(path), 0.0)

    def test_device_synchronization_helper_accepts_cpu(self) -> None:
        _, seconds = measure_callable(lambda: torch.tensor([1.0]).sum(), torch.device("cpu"))
        self.assertGreaterEqual(seconds, 0.0)


if __name__ == "__main__":
    unittest.main()
