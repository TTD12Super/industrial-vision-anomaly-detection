"""Small, calibrated score-fusion cascade for the screw experiment."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = (
    "p_global_dino",
    "cosine_distance_good",
    "cosine_distance_defect",
    "anomalyclip_score",
    "heatmap_max",
    "heatmap_p99",
    "heatmap_p95",
    "heatmap_mean",
    "heatmap_std",
    "heatmap_high_area_ratio",
    "stage1_candidate",
)


def heatmap_statistics(heatmap: np.ndarray) -> np.ndarray:
    """Compress a continuous anomaly map into stable image-level signals."""

    values = np.asarray(heatmap, dtype=np.float32)
    if values.ndim != 2 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("heatmap must be a finite, non-empty 2-D array")
    p99, p95 = np.percentile(values, [99.0, 95.0])
    return np.asarray(
        [
            float(values.max()),
            float(p99),
            float(p95),
            float(values.mean()),
            float(values.std()),
            float((values >= p95).mean()),
        ],
        dtype=np.float32,
    )


def _cosine_distance(embeddings: np.ndarray, prototype: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    prototype = np.asarray(prototype, dtype=np.float32).reshape(1, -1)
    normalized = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)
    normalized_prototype = prototype / max(float(np.linalg.norm(prototype)), 1e-8)
    return 1.0 - (normalized @ normalized_prototype.T).reshape(-1)


class CascadeScoreFusion:
    """Fuse DINO and AnomalyCLIP image-level evidence with hard-negative weighting."""

    def __init__(self, scaler: StandardScaler, classifier: LogisticRegression, good_prototype: np.ndarray, defect_prototype: np.ndarray, candidate_threshold: float, final_threshold: float) -> None:
        self.scaler = scaler
        self.classifier = classifier
        self.good_prototype = np.asarray(good_prototype, dtype=np.float32)
        self.defect_prototype = np.asarray(defect_prototype, dtype=np.float32)
        self.candidate_threshold = float(candidate_threshold)
        self.final_threshold = float(final_threshold)

    @staticmethod
    def candidate_signal(p_global: np.ndarray, anomalyclip_score: np.ndarray) -> np.ndarray:
        p_global = np.asarray(p_global, dtype=np.float32).reshape(-1)
        anomalyclip_score = np.asarray(anomalyclip_score, dtype=np.float32).reshape(-1)
        if p_global.shape != anomalyclip_score.shape:
            raise ValueError("candidate signals must have equal shapes")
        return np.maximum(p_global, anomalyclip_score)

    @classmethod
    def fit(
        cls,
        embeddings: np.ndarray,
        labels: np.ndarray,
        p_global: np.ndarray,
        anomalyclip_score: np.ndarray,
        heatmap_stats: np.ndarray,
        candidate_threshold: float,
        final_threshold: float,
    ) -> "CascadeScoreFusion":
        embeddings = np.asarray(embeddings, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        p_global = np.asarray(p_global, dtype=np.float32).reshape(-1)
        anomalyclip_score = np.asarray(anomalyclip_score, dtype=np.float32).reshape(-1)
        heatmap_stats = np.asarray(heatmap_stats, dtype=np.float32)
        if embeddings.ndim != 2 or heatmap_stats.ndim != 2 or len(labels) != len(embeddings):
            raise ValueError("cascade training arrays have incompatible shapes")
        if heatmap_stats.shape[0] != len(labels) or len(p_global) != len(labels) or len(anomalyclip_score) != len(labels):
            raise ValueError("cascade training arrays must have equal row counts")
        if set(labels.tolist()) != {0, 1}:
            raise ValueError("cascade training labels must contain GOOD and DEFECT")
        good_prototype = embeddings[labels == 0].mean(axis=0)
        defect_prototype = embeddings[labels == 1].mean(axis=0)
        stage1 = (cls.candidate_signal(p_global, anomalyclip_score) >= candidate_threshold).astype(np.float32)
        features = np.column_stack(
            [
                p_global,
                _cosine_distance(embeddings, good_prototype),
                _cosine_distance(embeddings, defect_prototype),
                anomalyclip_score,
                heatmap_stats,
                stage1,
            ]
        )
        scaler = StandardScaler().fit(features)
        classifier = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=42)
        sample_weight = np.ones(len(labels), dtype=np.float32)
        sample_weight[(labels == 0) & (stage1 == 1)] = 3.0
        classifier.fit(scaler.transform(features), labels, sample_weight=sample_weight)
        return cls(scaler, classifier, good_prototype, defect_prototype, candidate_threshold, final_threshold)

    def features(self, embeddings: np.ndarray, p_global: np.ndarray, anomalyclip_score: np.ndarray, heatmap_stats: np.ndarray) -> np.ndarray:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        p_global = np.asarray(p_global, dtype=np.float32).reshape(-1)
        anomalyclip_score = np.asarray(anomalyclip_score, dtype=np.float32).reshape(-1)
        heatmap_stats = np.asarray(heatmap_stats, dtype=np.float32)
        stage1 = (self.candidate_signal(p_global, anomalyclip_score) >= self.candidate_threshold).astype(np.float32)
        return np.column_stack(
            [
                p_global,
                _cosine_distance(embeddings, self.good_prototype),
                _cosine_distance(embeddings, self.defect_prototype),
                anomalyclip_score,
                heatmap_stats,
                stage1,
            ]
        )

    def predict_score(self, embeddings: np.ndarray, p_global: np.ndarray, anomalyclip_score: np.ndarray, heatmap_stats: np.ndarray) -> np.ndarray:
        features = self.features(embeddings, p_global, anomalyclip_score, heatmap_stats)
        return self.classifier.predict_proba(self.scaler.transform(features))[:, 1].astype(np.float32)

    def predict(self, embeddings: np.ndarray, p_global: np.ndarray, anomalyclip_score: np.ndarray, heatmap_stats: np.ndarray) -> np.ndarray:
        signal = self.candidate_signal(p_global, anomalyclip_score)
        score = self.predict_score(embeddings, p_global, anomalyclip_score, heatmap_stats)
        return ((signal >= self.candidate_threshold) & (score >= self.final_threshold)).astype(np.int64)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump({"feature_names": FEATURE_NAMES, "model": self}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "CascadeScoreFusion":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        if payload.get("feature_names") != FEATURE_NAMES or not isinstance(payload.get("model"), cls):
            raise ValueError("invalid cascade fusion artifact")
        return payload["model"]
