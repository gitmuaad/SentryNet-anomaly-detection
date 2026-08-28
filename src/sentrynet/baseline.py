"""Transparent statistical anomaly baseline.

Uses the five raw source features in their original units rather than the scaled model
matrix, so an alert can always be explained in plain terms ("this flow used more packets
than any Normal flow we trained on").

Statistics come from Normal training rows only. Two rule families are supported, and the
choice between them is a hyperparameter tuned on the validation set:

- percentile: distance outside a [lo, hi] band taken at percentiles p and 100-p
- zscore: |x - mean| / std

Both divide by max(spread, MIN_BAND_WIDTH). failed_logins is exactly 0 for every Normal row
here, so its std and band width are both zero without that floor -- and a tiny epsilon
instead of 1.0 would let that one feature dominate every score.

The final score is the max across features, so it's always traceable to one feature. Higher
means more anomalous.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

MIN_BAND_WIDTH = 1.0
RULES = ("percentile", "zscore")


@dataclass
class StatisticalBaseline:
    """Percentile / z-score anomaly baseline fit on Normal training rows only."""

    core_features: Sequence[str]
    rule: str = "percentile"
    percentile: float = 0.5
    min_band_width: float = MIN_BAND_WIDTH
    stats_: dict[str, dict[str, float]] = field(default_factory=dict)
    fitted_: bool = False

    name = "statistical_baseline"

    def fit(self, normal_train: pd.DataFrame) -> "StatisticalBaseline":
        """Estimate per-feature Normal statistics. ``normal_train`` must be Normal rows only."""
        if self.rule not in RULES:
            raise ValueError(f"rule must be one of {RULES}, got {self.rule!r}")
        stats: dict[str, dict[str, float]] = {}
        for col in self.core_features:
            values = normal_train[col].to_numpy(dtype="float64")
            lo = float(np.percentile(values, self.percentile))
            hi = float(np.percentile(values, 100.0 - self.percentile))
            stats[col] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "lo": lo,
                "hi": hi,
                "band_width": float(max(hi - lo, self.min_band_width)),
                "std_floored": float(max(values.std(ddof=0), self.min_band_width)),
                "min": float(values.min()),
                "max": float(values.max()),
                "zero_variance": bool(values.std(ddof=0) == 0.0),
            }
        self.stats_ = stats
        self.fitted_ = True
        return self

    def per_feature_scores(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Per-feature excursion values — the explanation behind the aggregate score."""
        self._check_fitted()
        out = {}
        for col in self.core_features:
            s = self.stats_[col]
            x = frame[col].to_numpy(dtype="float64")
            if self.rule == "percentile":
                excursion = np.maximum.reduce([np.zeros_like(x), x - s["hi"], s["lo"] - x])
                out[col] = excursion / s["band_width"]
            else:
                out[col] = np.abs(x - s["mean"]) / s["std_floored"]
        return pd.DataFrame(out, index=frame.index)

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        """Anomaly score per row; higher means more anomalous."""
        scores = self.per_feature_scores(frame).to_numpy(dtype="float64")
        return np.nan_to_num(scores.max(axis=1), nan=0.0, posinf=0.0, neginf=0.0)

    def explain(self, frame: pd.DataFrame) -> pd.Series:
        """Name of the feature responsible for each row's score."""
        per_feature = self.per_feature_scores(frame)
        return per_feature.idxmax(axis=1)

    def get_params(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "percentile": self.percentile,
            "min_band_width": self.min_band_width,
            "core_features": list(self.core_features),
        }

    def _check_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError("StatisticalBaseline must be fit on Normal training rows first.")


def baseline_grid(grid_cfg: dict[str, Iterable[Any]]) -> list[dict[str, Any]]:
    """Expand the configured baseline grid into an ordered list of parameter dicts."""
    configs: list[dict[str, Any]] = []
    for rule in grid_cfg["rule"]:
        if rule == "percentile":
            for pct in grid_cfg["percentile"]:
                configs.append({"rule": rule, "percentile": float(pct)})
        else:
            # The z-score rule does not use the percentile parameter.
            configs.append({"rule": rule, "percentile": float(list(grid_cfg["percentile"])[0])})
    return configs


def fit_baseline(
    normal_train: pd.DataFrame,
    core_features: Sequence[str],
    params: dict[str, Any],
    min_band_width: float = MIN_BAND_WIDTH,
) -> StatisticalBaseline:
    """Fit one baseline configuration on Normal training rows."""
    model = StatisticalBaseline(
        core_features=list(core_features),
        rule=str(params["rule"]),
        percentile=float(params["percentile"]),
        min_band_width=float(min_band_width),
    )
    return model.fit(normal_train)
