"""Small, model-free ledger and selector for static-image experiments."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping


SPLIT_ROLES = ("train", "validation", "diagnostic", "independent_holdout")
METRIC_UNITS = ("image", "object", "crop", "pixel", "session")
REQUIRED_METRICS = ("defect_recall", "good_false_positive_rate")
REQUIRED_DENOMINATORS = ("defect_count", "good_count")
REQUIRED_NUMERATORS = ("defect_true_positive", "good_false_alarm_count")
DEFAULT_TIE_BREAK = (
    "lower_good_false_positive_rate",
    "lower_latency_ms",
    "lower_name",
)
MAX_RECALL_TIE_BREAK = (
    "maximize_defect_recall",
    "lower_good_false_positive_rate",
    "higher_threshold",
)
SUPPORTED_TIE_BREAKS = (DEFAULT_TIE_BREAK, MAX_RECALL_TIE_BREAK)
METRIC_TOLERANCE = 1e-3
NON_RATE_METRICS = {"latency_ms", "memory_mb"}


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one existing file."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_non_negative(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite non-negative number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite non-negative number") from error
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return number


def _unit_interval(value: Any, label: str) -> float:
    number = _finite_non_negative(value, label)
    if number > 1.0:
        raise ValueError(f"{label} must be in the range [0, 1]")
    return number


def _path(value: Any, base_dir: Path, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty file path")
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return str(path)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _selection_mentions_holdout(selection: Mapping[str, Any]) -> bool:
    guarded_keys = {
        "split_role",
        "selection_split_role",
        "metric_split_role",
        "metrics_split_role",
        "source_split_role",
    }
    for key, value in selection.items():
        if key in guarded_keys and value == "independent_holdout":
            return True
        if "holdout" in key.lower() and value not in (False, None):
            return True
    return False


def _verify_hash(expected: Any, actual: str, label: str) -> None:
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
        raise ValueError(f"{label} must be a 64-character SHA-256 hex digest")
    if expected != actual:
        raise ValueError(f"{label} does not match the current file")


def _strict_count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer count")
    if value < 0:
        raise ValueError(f"{label} must be a non-negative integer count")
    return value


def _validate_metric_contract(
    candidate: Mapping[str, Any], index: int
) -> tuple[str, dict[str, float], dict[str, int | float], dict[str, int | float]]:
    label = f"candidates[{index}]"
    metric_unit = candidate.get("metric_unit")
    if metric_unit not in METRIC_UNITS:
        raise ValueError(f"{label}.metric_unit must be one of {METRIC_UNITS}")

    metrics = dict(_mapping(candidate.get("metrics"), f"{label}.metrics"))
    for metric_name, metric_value in list(metrics.items()):
        if metric_name in NON_RATE_METRICS:
            metrics[metric_name] = _finite_non_negative(metric_value, f"{label}.metrics.{metric_name}")
        else:
            metrics[metric_name] = _unit_interval(metric_value, f"{label}.metrics.{metric_name}")
    for metric_name in REQUIRED_METRICS:
        if metric_name not in metrics:
            raise ValueError(f"{label}.metrics.{metric_name} is required")

    denominators = dict(_mapping(candidate.get("metric_denominators"), f"{label}.metric_denominators"))
    for metric_name, denominator in list(denominators.items()):
        if metric_name in REQUIRED_DENOMINATORS:
            value = _strict_count(denominator, f"{label}.metric_denominators.{metric_name}")
        else:
            value = _finite_non_negative(denominator, f"{label}.metric_denominators.{metric_name}")
        if value <= 0.0:
            raise ValueError(f"{label}.metric_denominators.{metric_name} must be positive")
        denominators[metric_name] = value
    for metric_name in REQUIRED_DENOMINATORS:
        if metric_name not in denominators:
            raise ValueError(f"{label}.metric_denominators.{metric_name} is required")

    numerators = dict(_mapping(candidate.get("metric_numerators"), f"{label}.metric_numerators"))
    for metric_name, numerator in list(numerators.items()):
        if metric_name in REQUIRED_NUMERATORS:
            numerators[metric_name] = _strict_count(
                numerator, f"{label}.metric_numerators.{metric_name}"
            )
        else:
            numerators[metric_name] = _finite_non_negative(
                numerator, f"{label}.metric_numerators.{metric_name}"
            )
    for metric_name in REQUIRED_NUMERATORS:
        if metric_name not in numerators:
            raise ValueError(f"{label}.metric_numerators.{metric_name} is required")

    defect_count = denominators["defect_count"]
    good_count = denominators["good_count"]
    defect_true_positive = numerators["defect_true_positive"]
    good_false_alarm_count = numerators["good_false_alarm_count"]
    if defect_true_positive > defect_count:
        raise ValueError(f"{label}.metric_numerators.defect_true_positive exceeds defect_count")
    if good_false_alarm_count > good_count:
        raise ValueError(f"{label}.metric_numerators.good_false_alarm_count exceeds good_count")
    expected_recall = defect_true_positive / defect_count
    expected_fp = good_false_alarm_count / good_count
    if abs(metrics["defect_recall"] - expected_recall) > METRIC_TOLERANCE:
        raise ValueError(f"{label}.metrics.defect_recall does not match TP/defect_count")
    if abs(metrics["good_false_positive_rate"] - expected_fp) > METRIC_TOLERANCE:
        raise ValueError(f"{label}.metrics.good_false_positive_rate does not match false_alarms/good_count")
    return metric_unit, metrics, numerators, denominators


def normalize_sweep_spec(
    spec: Mapping[str, Any],
    base_dir: str | Path = ".",
    *,
    spec_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and normalize a JSON sweep spec without running an experiment."""

    if not isinstance(spec, Mapping):
        raise ValueError("Sweep spec must be a JSON object")
    if spec.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    experiment = spec.get("experiment")
    if not isinstance(experiment, str) or not experiment.strip():
        raise ValueError("experiment must be a non-empty string")
    if spec.get("independent_holdout_locked") is not True:
        raise ValueError("independent_holdout_locked must be true")
    base_dir = Path(base_dir).resolve()

    raw_splits = _mapping(spec.get("split_roles"), "split_roles")
    missing_roles = [role for role in SPLIT_ROLES if role not in raw_splits]
    if missing_roles:
        raise ValueError(f"split_roles must explicitly contain: {', '.join(missing_roles)}")
    split_roles: dict[str, dict[str, str]] = {}
    for role in SPLIT_ROLES:
        value = raw_splits[role]
        if not isinstance(value, Mapping):
            raise ValueError(f"split_roles.{role} must include manifest and sha256")
        manifest_path = _path(value.get("manifest"), base_dir, f"split_roles.{role}.manifest")
        actual_hash = sha256_file(manifest_path)
        _verify_hash(value.get("sha256"), actual_hash, f"split_roles.{role}.sha256")
        split_roles[role] = {
            "manifest": manifest_path,
            "sha256": actual_hash,
            "expected_sha256": value["sha256"],
        }

    raw_lineage = _mapping(spec.get("lineage", {}), "lineage")
    diagnostic_alias = raw_lineage.get("diagnostic_alias_of")
    if diagnostic_alias is not None and diagnostic_alias != "test":
        raise ValueError("lineage.diagnostic_alias_of must be 'test'")
    if diagnostic_alias == "test":
        alias_manifest = _path(
            raw_lineage.get("diagnostic_alias_manifest"),
            base_dir,
            "lineage.diagnostic_alias_manifest",
        )
        alias_hash = sha256_file(alias_manifest)
        _verify_hash(
            raw_lineage.get("diagnostic_alias_sha256"),
            alias_hash,
            "lineage.diagnostic_alias_sha256",
        )
        if alias_manifest != split_roles["diagnostic"]["manifest"] or alias_hash != split_roles["diagnostic"]["sha256"]:
            raise ValueError("diagnostic alias manifest must match the locked diagnostic role")

    raw_selection = _mapping(spec.get("selection"), "selection")
    if _selection_mentions_holdout(raw_selection):
        raise ValueError("independent_holdout metrics cannot be used for selection")
    allowed_selection_keys = {
        "target_defect_recall",
        "good_false_positive_cap",
        "candidate_names",
        "selection_split_role",
        "tie_break",
    }
    unknown_selection_keys = sorted(set(raw_selection) - allowed_selection_keys)
    if unknown_selection_keys:
        raise ValueError(f"Unsupported selection fields: {unknown_selection_keys}")
    if raw_selection.get("selection_split_role", "validation") != "validation":
        raise ValueError("selection.selection_split_role must be validation")
    target_recall = _unit_interval(
        raw_selection.get("target_defect_recall"),
        "selection.target_defect_recall",
    )
    fp_cap = _unit_interval(
        raw_selection.get("good_false_positive_cap"),
        "selection.good_false_positive_cap",
    )
    tie_break = raw_selection.get("tie_break", list(DEFAULT_TIE_BREAK))
    if tie_break not in [list(rule) for rule in SUPPORTED_TIE_BREAKS]:
        raise ValueError(
            "selection.tie_break must match one of "
            + repr([list(rule) for rule in SUPPORTED_TIE_BREAKS])
        )
    candidate_names = raw_selection.get("candidate_names")
    if candidate_names is not None:
        if not isinstance(candidate_names, list) or not candidate_names or any(
            not isinstance(name, str) or not name for name in candidate_names
        ):
            raise ValueError("selection.candidate_names must be a non-empty list of names")
        if len(set(candidate_names)) != len(candidate_names):
            raise ValueError("selection.candidate_names must not contain duplicates")

    lineage = dict(raw_lineage)
    config_path = lineage.get("config_path")
    if config_path is None:
        raise ValueError("lineage.config_path is required")
    normalized_config = _path(config_path, base_dir, "lineage.config_path")
    config_actual_hash = sha256_file(normalized_config)
    _verify_hash(lineage.get("config_sha256"), config_actual_hash, "lineage.config_sha256")
    lineage["config_path"] = normalized_config
    lineage["config_expected_sha256"] = lineage["config_sha256"]
    lineage["config_sha256"] = config_actual_hash
    if spec_path is not None:
        normalized_spec_path = Path(spec_path).resolve()
        lineage["spec_path"] = str(normalized_spec_path)
        lineage["spec_sha256"] = sha256_file(normalized_spec_path)
    elif "spec_sha256" not in lineage:
        canonical = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
        lineage["spec_sha256"] = hashlib.sha256(canonical).hexdigest()

    raw_candidates = spec.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("candidates must be a non-empty list")
    candidates: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw_candidate in enumerate(raw_candidates):
        candidate = _mapping(raw_candidate, f"candidates[{index}]")
        label = f"candidates[{index}]"
        name = candidate.get("name")
        method = candidate.get("method")
        split_role = candidate.get("split_role")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{label}.name must be a non-empty string")
        if name in names:
            raise ValueError(f"Duplicate candidate name: {name}")
        names.add(name)
        if not isinstance(method, str) or not method.strip():
            raise ValueError(f"{label}.method must be a non-empty string")
        if split_role not in SPLIT_ROLES:
            raise ValueError(f"{label}.split_role is invalid: {split_role}")
        if candidate.get("holdout_used_for_selection") is not False:
            raise ValueError(f"{label}.holdout_used_for_selection must be explicitly false")

        metric_unit, metrics, numerators, denominators = _validate_metric_contract(candidate, index)

        raw_artifacts = candidate.get("artifact_paths")
        if isinstance(raw_artifacts, str):
            raw_artifacts = [raw_artifacts]
        if not isinstance(raw_artifacts, list) or not raw_artifacts:
            raise ValueError(f"{label}.artifact_paths must be a non-empty path list")
        artifact_paths = [_path(value, base_dir, f"{label}.artifact_paths") for value in raw_artifacts]
        raw_artifact_hashes = _mapping(candidate.get("artifact_sha256"), f"{label}.artifact_sha256")
        if not raw_artifact_hashes:
            raise ValueError(f"{label}.artifact_sha256 must pin every artifact path")
        expected_artifact_hashes: dict[str, str] = {}
        for declared_path, expected_hash in raw_artifact_hashes.items():
            normalized_declared_path = _path(
                declared_path, base_dir, f"{label}.artifact_sha256 path"
            )
            expected_artifact_hashes[normalized_declared_path] = expected_hash
        if set(expected_artifact_hashes) != set(artifact_paths):
            raise ValueError(f"{label}.artifact_sha256 must pin exactly artifact_paths")
        artifact_hashes = {}
        for path in artifact_paths:
            actual_hash = sha256_file(path)
            _verify_hash(
                expected_artifact_hashes[path],
                actual_hash,
                f"{label}.artifact_sha256[{path}]",
            )
            artifact_hashes[path] = actual_hash

        source_report_path = _path(
            candidate.get("source_report_path"), base_dir, f"{label}.source_report_path"
        )
        source_report_hash = sha256_file(source_report_path)
        declared_source_hash = candidate.get("source_report_sha256")
        _verify_hash(declared_source_hash, source_report_hash, f"{label}.source_report_sha256")

        normalized = {
            "name": name,
            "method": method,
            "metric_unit": metric_unit,
            "artifact_paths": artifact_paths,
            "artifact_sha256": expected_artifact_hashes,
            "artifact_hashes": artifact_hashes,
            "source_report_path": source_report_path,
            "source_report_expected_sha256": declared_source_hash,
            "source_report_sha256": source_report_hash,
            "source_report_actual_sha256": source_report_hash,
            "metrics": metrics,
            "metric_numerators": numerators,
            "metric_denominators": denominators,
            "split_role": split_role,
            "holdout_used_for_selection": False,
        }
        for key in ("seed", "device", "environment", "command"):
            if key in candidate:
                normalized[key] = candidate[key]
        candidates.append(normalized)

    if tie_break == list(MAX_RECALL_TIE_BREAK):
        missing_threshold = [
            candidate["name"]
            for candidate in candidates
            if "threshold" not in candidate["metrics"]
        ]
        if missing_threshold:
            raise ValueError(
                "maximize_defect_recall tie-break requires metrics.threshold for every candidate: "
                + ", ".join(missing_threshold)
            )

    if candidate_names is not None:
        missing_names = sorted(set(candidate_names) - names)
        if missing_names:
            raise ValueError(f"selection.candidate_names refer to unknown candidates: {missing_names}")
        selected_rows = [candidate for candidate in candidates if candidate["name"] in candidate_names]
        invalid_roles = sorted(
            {candidate["split_role"] for candidate in selected_rows if candidate["split_role"] != "validation"}
        )
        if invalid_roles:
            raise ValueError(
                "Only validation candidates may be selected; received split roles: "
                + ", ".join(invalid_roles)
            )

    return {
        "schema_version": 1,
        "experiment": experiment,
        "independent_holdout_locked": True,
        "split_roles": split_roles,
        "selection": {
            "target_defect_recall": target_recall,
            "good_false_positive_cap": fp_cap,
            "candidate_names": candidate_names,
            "selection_split_role": "validation",
            "tie_break": list(tie_break),
        },
        "lineage": lineage,
        "candidates": candidates,
    }


def validate_sweep_spec(
    spec: Mapping[str, Any], base_dir: str | Path = "."
) -> dict[str, Any]:
    """Alias kept explicit for callers that only need validation."""

    return normalize_sweep_spec(spec, base_dir)


def load_sweep_spec(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Sweep spec does not exist: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON sweep spec: {path}") from error
    return normalize_sweep_spec(raw, path.parent, spec_path=path)


def select_candidate(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Select validation-only candidates; diagnostic/holdout rows are evidence only."""

    selection = _mapping(spec["selection"], "selection")
    if _selection_mentions_holdout(selection):
        raise ValueError("independent_holdout metrics cannot be used for selection")
    tie_break = tuple(selection.get("tie_break", DEFAULT_TIE_BREAK))
    if tie_break not in SUPPORTED_TIE_BREAKS:
        raise ValueError("selection.tie_break is not supported")
    candidates = list(spec["candidates"])
    names = selection.get("candidate_names")
    if names is not None:
        by_name = {candidate["name"]: candidate for candidate in candidates}
        selected_rows = [by_name[name] for name in names]
        invalid_roles = sorted({row["split_role"] for row in selected_rows if row["split_role"] != "validation"})
        if invalid_roles:
            raise ValueError(
                "Only validation candidates may be selected; received split roles: "
                + ", ".join(invalid_roles)
            )
        validation_candidates = selected_rows
    else:
        validation_candidates = [candidate for candidate in candidates if candidate["split_role"] == "validation"]

    target_recall = float(selection["target_defect_recall"])
    fp_cap = float(selection["good_false_positive_cap"])
    eligible = [
        candidate
        for candidate in validation_candidates
        if candidate["metrics"]["defect_recall"] >= target_recall
        and candidate["metrics"]["good_false_positive_rate"] <= fp_cap
    ]
    result = {
        "selection_split_role": "validation",
        "target_defect_recall": target_recall,
        "good_false_positive_cap": fp_cap,
        "eligible_candidate_count": len(eligible),
    }
    if not eligible:
        return {"status": "not_passed", "selected_candidate": None, **result}

    def sort_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
        metrics = candidate["metrics"]
        if tie_break == MAX_RECALL_TIE_BREAK:
            return (
                -float(metrics["defect_recall"]),
                float(metrics["good_false_positive_rate"]),
                -float(metrics["threshold"]),
                str(candidate["name"]),
            )
        return (
            -float(metrics["defect_recall"]),
            float(metrics["good_false_positive_rate"]),
            float(metrics.get("latency_ms", math.inf)),
            str(candidate["name"]),
        )

    selected = min(eligible, key=sort_key)
    return {"status": "passed", "selected_candidate": selected, **result}


def _write_report_file(path: Path, content: str) -> None:
    """Small seam used by tests to exercise cleanup after a partial write."""

    path.write_text(content, encoding="utf-8")


def write_sweep_report(spec: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Atomically publish one immutable ledger run; refuse reuse of a directory."""

    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing sweep run directory: {output_dir}")
    normalized = normalize_sweep_spec(spec, Path.cwd())
    selection = select_candidate(normalized)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=str(output_dir.parent))
    )
    published = False
    try:
        manifest = {
            "schema_version": "1.1",
            "experiment": normalized["experiment"],
            "spec_sha256": normalized["lineage"]["spec_sha256"],
            "lineage": normalized["lineage"],
            "independent_holdout_locked": True,
            "split_roles": normalized["split_roles"],
            "selection_rule": normalized["selection"],
            "candidate_count": len(normalized["candidates"]),
            "candidate_provenance": {
                candidate["name"]: {
                    "source_report_path": candidate["source_report_path"],
                    "source_report_sha256": {
                        "expected": candidate["source_report_expected_sha256"],
                        "actual": candidate["source_report_actual_sha256"],
                    },
                    "artifact_hashes": {
                        "expected": candidate["artifact_sha256"],
                        "actual": candidate["artifact_hashes"],
                    },
                }
                for candidate in normalized["candidates"]
            },
            "independent_holdout_used_for_selection": False,
            "metric_provenance_independently_audited": False,
            "protocol_guard": (
                "Selector protocol guard only; metric provenance was not independently audited."
            ),
        }
        _write_report_file(temporary_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")
        candidates_jsonl = "".join(
            json.dumps(candidate, sort_keys=True) + "\n" for candidate in normalized["candidates"]
        )
        _write_report_file(temporary_dir / "candidates.jsonl", candidates_jsonl)

        validation_candidates = [
            candidate for candidate in normalized["candidates"] if candidate["split_role"] == "validation"
        ]
        validation_summary = {
            "schema_version": "1.1",
            "experiment": normalized["experiment"],
            "selection_split_role": "validation",
            "candidate_count": len(validation_candidates),
            "selection": selection,
            "metric_provenance_independently_audited": False,
            "candidates": [
                {
                    "name": candidate["name"],
                    "method": candidate["method"],
                    "metric_unit": candidate["metric_unit"],
                    "metrics": candidate["metrics"],
                    "metric_numerators": candidate["metric_numerators"],
                    "metric_denominators": candidate["metric_denominators"],
                    "source_report_sha256": candidate["source_report_sha256"],
                }
                for candidate in validation_candidates
            ],
        }
        _write_report_file(
            temporary_dir / "validation_summary.json",
            json.dumps(validation_summary, indent=2) + "\n",
        )
        _write_report_file(
            temporary_dir / "selected_candidate.json",
            json.dumps(selection, indent=2) + "\n",
        )
        os.rename(temporary_dir, output_dir)
        published = True
        return manifest
    finally:
        if not published and temporary_dir.exists():
            shutil.rmtree(temporary_dir, ignore_errors=True)
