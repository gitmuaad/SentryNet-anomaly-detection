from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from sentrynet import features as feat
from sentrynet.baseline import StatisticalBaseline
from sentrynet.isolation_forest import anomaly_score as if_score
from sentrynet.one_class_svm import anomaly_score as ocsvm_score
from sentrynet.preprocessing import FittedPreprocessor

DECISION_NORMAL = "Normal"
DECISION_SUSPICIOUS = "Suspicious"

KIND_BASELINE = "statistical_baseline"
KIND_IFOREST = "isolation_forest"
KIND_OCSVM = "one_class_svm"


@dataclass
class ScoringPipeline:
    """A frozen detector: feature engineering + preprocessing + model + operating threshold."""

    kind: str
    model: Any
    threshold: float
    preprocessor: FittedPreprocessor | None = None
    numeric_features: Sequence[str] = feat.SOURCE_NUMERIC + feat.DERIVED_NUMERIC
    categorical_features: Sequence[str] = feat.SOURCE_CATEGORICAL
    safe_denominator_floor: float = feat.SAFE_DENOMINATOR_FLOOR
    metadata: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return self.kind

    def required_columns(self) -> list[str]:
        """Raw source columns an uploaded CSV must provide."""
        return list(feat.SOURCE_NUMERIC) + list(feat.SOURCE_CATEGORICAL)

    def score(self, raw: pd.DataFrame) -> np.ndarray:
        """Anomaly scores for raw flow rows; higher means more anomalous."""
        missing = [c for c in self.required_columns() if c not in raw.columns]
        if missing:
            raise ValueError(
                f"Input is missing required network-flow columns: {missing}. "
                f"Expected: {self.required_columns()}"
            )

        if self.kind == KIND_BASELINE:
            enriched = feat.add_derived_features(raw, floor=self.safe_denominator_floor)
            return np.asarray(self.model.score(enriched), dtype="float64")

        matrix = feat.build_feature_frame(
            raw,
            numeric=feat.SOURCE_NUMERIC,
            derived=feat.DERIVED_NUMERIC,
            categorical=self.categorical_features,
            floor=self.safe_denominator_floor,
        )
        if self.preprocessor is None:
            raise RuntimeError(f"Detector {self.kind!r} requires a fitted preprocessor.")
        transformed = self.preprocessor.transform(matrix)
        if self.kind == KIND_IFOREST:
            return if_score(self.model, transformed)
        if self.kind == KIND_OCSVM:
            return ocsvm_score(self.model, transformed)
        raise ValueError(f"Unknown detector kind: {self.kind!r}")

    def decide(self, scores: np.ndarray) -> np.ndarray:
        """Binary decision at the frozen operating threshold."""
        flagged = np.asarray(scores, dtype="float64") >= self.threshold
        return np.where(flagged, DECISION_SUSPICIOUS, DECISION_NORMAL)

    def score_frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Return ``raw`` with ``anomaly_score`` and ``decision`` appended."""
        scores = self.score(raw)
        out = raw.copy()
        out["anomaly_score"] = scores
        out["decision"] = self.decide(scores)
        return out


def build_pipeline(
    kind: str,
    model: Any,
    threshold: float,
    preprocessor: FittedPreprocessor | None = None,
    metadata: dict[str, Any] | None = None,
    safe_denominator_floor: float = feat.SAFE_DENOMINATOR_FLOOR,
) -> ScoringPipeline:
    """Construct a :class:`ScoringPipeline`, validating the detector/preprocessor pairing."""
    if kind in (KIND_IFOREST, KIND_OCSVM) and preprocessor is None:
        raise ValueError(f"{kind} requires a fitted preprocessor.")
    if kind == KIND_BASELINE and not isinstance(model, StatisticalBaseline):
        raise TypeError("statistical_baseline requires a fitted StatisticalBaseline instance.")
    return ScoringPipeline(
        kind=kind,
        model=model,
        threshold=float(threshold),
        preprocessor=preprocessor,
        metadata=metadata or {},
        safe_denominator_floor=safe_denominator_floor,
    )
