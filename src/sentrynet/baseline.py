"""Transparent statistical anomaly baseline.

Owner in the Action Plan: **Faris Alsharaan**.

This detector deliberately uses the five *raw* source features in their original units
(``duration``, ``src_bytes``, ``dst_bytes``, ``packet_count``, ``failed_logins``) rather than
the scaled model matrix, because its whole purpose is to be explainable to a human reviewer:
"this flow used more packets than any Normal flow we trained on".

Statistics are estimated from **Normal training rows only**. Two rule families are supported
and the choice between them is a hyperparameter selected on the approved validation set
(Normal + DDoS + BruteForce, never PortScan).

Percentile rule
    Per feature, a Normal band ``[lo, hi]`` is taken at percentiles ``p`` and ``100 - p``.
    A row's excursion is how far outside the band it falls, expressed in band widths::

        d = max(0, x - hi, lo - x) / max(hi - lo, MIN_BAND_WIDTH)

Z-score rule
    Per feature::

        d = |x - mean| / max(std, MIN_BAND_WIDTH)

Both divide by ``max(..., MIN_BAND_WIDTH)`` with ``MIN_BAND_WIDTH = 1.0``. This matters:
``failed_logins`` is **exactly 0 for every Normal row** in this dataset, so its standard
deviation and its percentile band width are both zero. An unfloored denominator would be a
division by zero; an arbitrarily tiny epsilon would make that one feature saturate the score.
Flooring at one integer unit is meaningful because all five features are integer counts or
whole seconds.

The final score is the **maximum** per-feature excursion, so the score always has a direct
human explanation: the single feature that is most out of range. Higher = more anomalous.
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
        """Continuous anomaly score. **Higher = more anomalous.**"""
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
