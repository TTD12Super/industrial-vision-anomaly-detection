from __future__ import annotations

import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from inspection.reporting import AsyncReportManager, LocalVLMReporter


class _Response:
    def __init__(self, payload: dict) -> None:
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


class _CaptureReporter:
    model_version = "capture-v1"

    def __init__(self) -> None:
        self.context: dict | None = None
        self.calls = 0

    def generate(self, context: dict) -> dict:
        self.context = context
        self.calls += 1
        return {
            "visible_defect_type": "crack",
            "defect_location": "head",
            "description": "A visible line is present.",
            "visible_evidence": ["line on head"],
            "heatmap_consistent_with_visible_evidence": "yes",
            "other_possible_explanation": "none",
            "needs_review": False,
        }


class ReportingTests(unittest.TestCase):
    def test_local_vlm_parses_openai_compatible_json_and_sends_images(self) -> None:
        reporter = LocalVLMReporter("http://127.0.0.1:8080/v1", "qwen3-vl")
        response = {
            "choices": [{"message": {"content": "```json\n{\"visible_defect_type\": \"crack\"}\n```"}}]
        }
        with patch("inspection.reporting.urllib_request.urlopen", return_value=_Response(response)) as urlopen:
            result = reporter.generate(
                {
                    "inspection_id": "i1",
                    "object_id": "screw_01",
                    "_evidence": [{"mime_type": "image/jpeg", "data": "ZmFrZQ=="}],
                }
            )

        self.assertEqual(result["visible_defect_type"], "crack")
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "qwen3-vl")
        self.assertIn("data:image/jpeg;base64,ZmFrZQ==", payload["messages"][0]["content"][1]["image_url"]["url"])

    def test_async_manager_passes_mime_types_and_persists_validated_report(self) -> None:
        reporter = _CaptureReporter()
        manager = AsyncReportManager(reporter)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            pending = manager.submit(
                {
                    "inspection_id": "i1",
                    "object_id": "screw_01",
                    "_evidence_mime_types": ["image/jpeg"],
                },
                [b"fake"],
                path,
            )
            pending.future.result(timeout=2)  # type: ignore[union-attr]
            self.assertEqual(manager.status(pending.report_id), "completed")
            self.assertEqual(reporter.calls, 1)
            assert reporter.context is not None
            self.assertEqual(reporter.context["_evidence"][0]["mime_type"], "image/jpeg")
            self.assertEqual(
                base64.b64decode(reporter.context["_evidence"][0]["data"]),
                b"fake",
            )
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["classifier_label"], "DEFECT")

    def test_async_manager_reuses_existing_report_cache(self) -> None:
        reporter = _CaptureReporter()
        manager = AsyncReportManager(reporter)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            context = {"inspection_id": "i1", "object_id": "screw_01"}
            first = manager.submit(context, [b"same-evidence"], path)
            first.future.result(timeout=2)  # type: ignore[union-attr]
            second = manager.submit(context, [b"same-evidence"], path)
            self.assertEqual(second.status, "completed")
            self.assertIsNone(second.future)
            self.assertEqual(reporter.calls, 1)

    def test_sync_manager_finishes_report_before_submit_returns(self) -> None:
        reporter = _CaptureReporter()
        manager = AsyncReportManager(reporter, async_enabled=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            pending = manager.submit(
                {"inspection_id": "i1", "object_id": "screw_01"},
                [b"same-evidence"],
                path,
            )
            self.assertEqual(pending.status, "completed")
            self.assertIsNone(pending.future)
            self.assertTrue(path.is_file())
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["report_status"], "completed")


if __name__ == "__main__":
    unittest.main()
