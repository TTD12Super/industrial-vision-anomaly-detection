"""Validated asynchronous English VLM reports and compact inspection PDFs."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import base64
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol
from urllib import request as urllib_request

REPORT_FIELDS = {
    "report_version",
    "inspection_id",
    "object_id",
    "language",
    "classifier_label",
    "visible_defect_type",
    "defect_location",
    "description",
    "visible_evidence",
    "heatmap_consistent_with_visible_evidence",
    "other_possible_explanation",
    "needs_review",
    "report_status",
}


class Reporter(Protocol):
    model_version: str

    def generate(self, context: dict[str, Any]) -> dict[str, Any]: ...


class ReportValidationError(ValueError):
    """The provider output cannot be stored as trusted report evidence."""


class MockReporter:
    """Offline provider used by tests; it deliberately makes no visual assertion."""

    model_version = "mock-vlm-v1"

    def generate(self, context: dict[str, Any]) -> dict[str, Any]:
        return {
            "visible_defect_type": "unknown",
            "defect_location": "unknown",
            "description": "Mock VLM report; no visual defect description was generated.",
            "visible_evidence": [],
            "heatmap_consistent_with_visible_evidence": "unknown",
            "other_possible_explanation": "unknown",
            "needs_review": True,
        }


class UnavailableCloudReporter:
    """Fails explicitly until a deployed provider, model and credential policy are configured."""

    model_version = "cloud-vlm-unconfigured"

    def generate(self, context: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("CLOUD_VLM_NOT_CONFIGURED")


class LocalVLMReporter:
    """Small OpenAI-compatible adapter for a local multimodal server.

    It intentionally owns no model runtime: llama.cpp and Ollama both expose
    the same local chat-completions boundary.  The deterministic pipeline
    remains authoritative; this class only turns saved evidence into text.
    """

    _PROMPT_VERSION = "screw-defect-report-v1"

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float = 120.0,
        max_tokens: int = 512,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("local_vlm.base_url must be an http(s) URL")
        if not model.strip():
            raise ValueError("local_vlm.model is required")
        if timeout_seconds <= 0 or max_tokens < 32:
            raise ValueError("local_vlm timeout and max_tokens must be positive")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = float(timeout_seconds)
        self.max_tokens = int(max_tokens)
        prompt_hash = hashlib.sha256(self._PROMPT_VERSION.encode()).hexdigest()[:12]
        self.model_version = f"local-vlm:{model}:{prompt_hash}"

    @staticmethod
    def _json_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict) and item.get("type") in {None, "text"}
            )
        raise RuntimeError("LOCAL_VLM_INVALID_MESSAGE_CONTENT")

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].lstrip()
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end < start:
            raise RuntimeError("LOCAL_VLM_RESPONSE_IS_NOT_JSON")
        payload = json.loads(cleaned[start : end + 1])
        if not isinstance(payload, dict):
            raise RuntimeError("LOCAL_VLM_RESPONSE_JSON_MUST_BE_OBJECT")
        return payload

    def _prompt(self, context: dict[str, Any]) -> str:
        facts = {key: value for key, value in context.items() if not key.startswith("_")}
        return (
            "You are an inspection assistant for screw images. The deterministic pipeline has already "
            "decided that this object is DEFECT. Inspect only the supplied images and facts. "
            "Do not change the classifier decision. If the crack or defect is not visibly supported, "
            "say unknown and set needs_review to true. Do not invent a defect type or location. "
            "Return JSON only with exactly these fields: visible_defect_type, defect_location, "
            "description, visible_evidence (array of short strings), "
            "heatmap_consistent_with_visible_evidence (yes/no/unknown), "
            "other_possible_explanation, needs_review (boolean).\n"
            f"Deterministic facts: {json.dumps(facts, sort_keys=True)}"
        )

    def generate(self, context: dict[str, Any]) -> dict[str, Any]:
        evidence = context.get("_evidence", [])
        content: list[dict[str, Any]] = [{"type": "text", "text": self._prompt(context)}]
        for item in evidence:
            if not isinstance(item, dict) or not isinstance(item.get("data"), str):
                raise RuntimeError("LOCAL_VLM_INVALID_EVIDENCE")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{item.get('mime_type', 'image/png')};base64,{item['data']}"
                    },
                }
            )
        payload = json.dumps(
            {
                "model": self.model,
                "temperature": 0,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": content}],
            }
        ).encode("utf-8")
        request = urllib_request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib_request.urlopen(request, timeout=self.timeout_seconds) as response:
            raw = json.loads(response.read().decode("utf-8"))
        try:
            text = self._json_content(raw["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("LOCAL_VLM_INVALID_RESPONSE") from error
        return self._parse_json(text)


def validate_report(raw: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Inject non-negotiable pipeline fields and reject malformed provider data."""

    if not isinstance(raw, dict):
        raise ReportValidationError("Provider output must be a JSON object")
    report = {
        "report_version": "1.0",
        "inspection_id": context["inspection_id"],
        "object_id": context["object_id"],
        "language": "en",
        "classifier_label": "DEFECT",
        "report_status": "completed",
        **raw,
        "classifier_label": "DEFECT",
    }
    missing = REPORT_FIELDS - report.keys()
    if missing:
        raise ReportValidationError(f"Report is missing fields: {sorted(missing)}")
    if report["language"] != "en" or report["classifier_label"] != "DEFECT":
        raise ReportValidationError("Report language and classifier label are fixed by the pipeline")
    if report["visible_defect_type"] is None:
        report["visible_defect_type"] = "unknown"
    if report["defect_location"] is None:
        report["defect_location"] = "unknown"
    if report["heatmap_consistent_with_visible_evidence"] not in {"yes", "no", "unknown"}:
        raise ReportValidationError("heatmap_consistent_with_visible_evidence must be yes, no or unknown")
    if not isinstance(report["visible_evidence"], list) or not all(isinstance(item, str) for item in report["visible_evidence"]):
        raise ReportValidationError("visible_evidence must be a list of strings")
    for key in ("visible_defect_type", "defect_location", "description", "other_possible_explanation"):
        if not isinstance(report[key], str) or len(report[key]) > 1000:
            raise ReportValidationError(f"{key} must be a string no longer than 1000 characters")
    if not isinstance(report["needs_review"], bool):
        raise ReportValidationError("needs_review must be boolean")
    report["report_status"] = "completed"
    return {key: report[key] for key in sorted(REPORT_FIELDS)}


@dataclass
class PendingReport:
    report_id: str
    status: str
    future: Future[dict[str, Any]] | None
    path: Path


class AsyncReportManager:
    """A bounded local queue with byte-exact cache keys and durable JSON evidence."""

    def __init__(self, reporter: Reporter, workers: int = 1, async_enabled: bool = True) -> None:
        self.reporter = reporter
        self.async_enabled = async_enabled
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="inspection-vlm") if async_enabled else None
        self.pending: dict[str, PendingReport] = {}

    def submit(self, context: dict[str, Any], evidence: list[bytes], path: Path) -> PendingReport:
        digest = hashlib.sha256()
        digest.update(self.reporter.model_version.encode())
        digest.update(json.dumps(context, sort_keys=True).encode())
        for item in evidence:
            digest.update(item)
        report_id = digest.hexdigest()
        if path.is_file():
            pending = PendingReport(report_id, "completed", None, path)
            self.pending[report_id] = pending
            return pending

        provider_context = dict(context)
        mime_types = context.get("_evidence_mime_types", [])
        provider_context["_evidence"] = [
            {
                "mime_type": mime_types[index] if index < len(mime_types) else "image/png",
                "data": base64.b64encode(item).decode("ascii"),
            }
            for index, item in enumerate(evidence)
        ]

        def work() -> dict[str, Any]:
            try:
                report = validate_report(self.reporter.generate(provider_context), context)
            except Exception as error:
                report = {
                    "report_version": "1.0",
                    "inspection_id": context["inspection_id"],
                    "object_id": context["object_id"],
                    "language": "en",
                    "classifier_label": "DEFECT",
                    "report_status": "failed",
                    "diagnostic": f"{type(error).__name__}: {error}",
                }
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            return report

        if not self.async_enabled:
            report = work()
            pending = PendingReport(report_id, str(report.get("report_status", "failed")), None, path)
            self.pending[report_id] = pending
            return pending

        assert self.executor is not None
        future = self.executor.submit(work)
        pending = PendingReport(report_id, "pending", future, path)
        self.pending[report_id] = pending
        return pending

    def status(self, report_id: str) -> str:
        pending = self.pending[report_id]
        if pending.future is None:
            return pending.status
        if not pending.future.done():
            return "pending"
        try:
            report = pending.future.result()
        except Exception:
            return "failed"
        return str(report.get("report_status", "failed"))


def render_summary_pdf(path: Path, inspection: dict[str, Any]) -> None:
    """Write a compact table that complements, rather than replaces, inspection.json."""

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(path), pagesize=A4, rightMargin=14 * mm, leftMargin=14 * mm, topMargin=12 * mm, bottomMargin=12 * mm
    )
    styles = getSampleStyleSheet()
    summary = inspection["summary"]
    title = Paragraph("Industrial inspection summary", styles["Title"])
    detail = Paragraph(
        f"Inspection: {inspection['inspection_id']} &nbsp; Product: {inspection['product_key']} &nbsp; "
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        styles["BodyText"],
    )
    rows: list[list[str]] = [["Object", "Status", "p(defect)", "VLM report"]]
    for item in inspection.get("objects", []):
        rows.append(
            [
                str(item["object_id"]),
                str(item["operational_status"]),
                f"{float(item['p_defect']):.4f}",
                str(item.get("vlm_status", "not_requested")),
            ]
        )
    if len(rows) == 1:
        rows.append(["-", "NO_DETECTION", "-", "not_requested"])
    table = Table(rows, colWidths=(40 * mm, 45 * mm, 32 * mm, 50 * mm), repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E78")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#AAB7C4")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7FA")]),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    note = Paragraph(
        f"Counts: GOOD {summary['good_count']}, DEFECT {summary['defect_count']}, "
        f"REVIEW_REQUIRED {summary['review_required_count']}. Pending reports remain in inspection.json.",
        styles["BodyText"],
    )
    doc.build([title, Spacer(1, 4 * mm), detail, Spacer(1, 5 * mm), table, Spacer(1, 4 * mm), note])
