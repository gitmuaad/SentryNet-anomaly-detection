"""Population Stability Index (PSI) drift monitoring.

The reference distribution is the Normal training partition. A new CSV, or any block of
one, is compared against that baseline: PSI under 0.10 is a non-issue, 0.10-0.25 is worth
watching, and above 0.25 flags a retraining review. Nothing here retrains automatically --
it just raises the flag.

failed_logins is exactly 0 for every Normal row, so quantile binning can't split it into
more than one bin. The fallback uses three bins (below / exactly / above the constant)
instead of two, since a naive two-bin split would lump the constant in with every larger
value and report zero drift no matter how far new data moved.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

PSI_REVIEW_THRESHOLD = 0.25
PSI_MODERATE_THRESHOLD = 0.10
PSI_EPSILON = 1e-6


def _bin_edges(reference: np.ndarray, n_bins: int) -> tuple[np.ndarray, str]:
    """Quantile bin edges from the reference distribution, with a constant-value fallback."""
    quantiles = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if edges.size < 2:
        # The reference is a single constant value v. Bins must isolate "exactly v" from
        # "above v": with edges [-inf, v, inf] numpy would place v and everything greater in
        # the same bin, making drift away from a constant reference invisible.
        constant = float(edges[0]) if edges.size else 0.0
        lower = np.nextafter(constant, -np.inf)
        upper = np.nextafter(constant, np.inf)
        return np.array([-np.inf, lower, upper, np.inf]), "constant_reference_three_bin"
    edges = edges.astype("float64")
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges, "quantile"


def _proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(np.asarray(values, dtype="float64"), bins=edges)
    total = counts.sum()
    if total == 0:
        return np.full(len(counts), PSI_EPSILON)
    props = counts / total
    return np.where(props <= 0.0, PSI_EPSILON, props)


def build_psi_reference(
    normal_train: pd.DataFrame, features: Sequence[str], n_bins: int = 10
) -> dict[str, Any]:
    """Build the PSI baseline profile from Normal training rows only."""
    profile: dict[str, Any] = {
        "n_bins": int(n_bins),
        "reference_partition": "train (Normal rows only)",
        "n_reference_rows": int(len(normal_train)),
        "features": {},
    }
    for col in features:
        values = normal_train[col].to_numpy(dtype="float64")
        edges, mode = _bin_edges(values, n_bins)
        profile["features"][col] = {
            "edges": edges.tolist(),
            "binning_mode": mode,
            "reference_proportions": _proportions(values, edges).tolist(),
            "reference_mean": float(values.mean()),
            "reference_std": float(values.std(ddof=0)),
        }
    return profile


def psi_value(expected: np.ndarray, actual: np.ndarray) -> float:
    """PSI between two proportion vectors: ``sum((a - e) * ln(a / e))``."""
    e = np.asarray(expected, dtype="float64")
    a = np.asarray(actual, dtype="float64")
    e = np.where(e <= 0.0, PSI_EPSILON, e)
    a = np.where(a <= 0.0, PSI_EPSILON, a)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_report_from_frame(
    frame: pd.DataFrame,
    reference: Mapping[str, Any],
    threshold: float = PSI_REVIEW_THRESHOLD,
) -> dict[str, Any]:
    """Compare a new frame against the stored reference profile."""
    per_feature: dict[str, Any] = {}
    for col, spec in reference["features"].items():
        if col not in frame.columns:
            per_feature[col] = {"psi": None, "status": "MISSING COLUMN", "triggered": False}
            continue
        edges = np.asarray(spec["edges"], dtype="float64")
        actual = _proportions(frame[col].to_numpy(dtype="float64"), edges)
        value = psi_value(np.asarray(spec["reference_proportions"], dtype="float64"), actual)
        per_feature[col] = {
            "psi": value,
            "status": (
                "REVIEW TRIGGERED"
                if value > threshold
                else ("MODERATE SHIFT" if value > PSI_MODERATE_THRESHOLD else "STABLE")
            ),
            "triggered": bool(value > threshold),
            "actual_proportions": actual.tolist(),
            "binning_mode": spec["binning_mode"],
        }

    triggered = sorted(c for c, r in per_feature.items() if r.get("triggered"))
    values = [r["psi"] for r in per_feature.values() if r.get("psi") is not None]
    return {
        "n_rows_scored": int(len(frame)),
        "threshold": float(threshold),
        "per_feature": per_feature,
        "max_psi": max(values) if values else None,
        "features_triggering_review": triggered,
        "retraining_review_recommended": bool(triggered),
        "decision": (
            f"PSI > {threshold} on {triggered} - schedule a human retraining review."
            if triggered
            else f"No feature exceeds PSI {threshold}. Continue the scheduled review cadence."
        ),
        "disclaimer": (
            "This is a prototype drift-and-retraining plan. No automated production "
            "retraining is performed or claimed."
        ),
    }


def psi_summary_table(report: Mapping[str, Any]) -> pd.DataFrame:
    """Small table for the report and for the Gradio drift panel."""
    rows = [
        {
            "feature": col,
            "psi": None if res.get("psi") is None else round(float(res["psi"]), 6),
            "status": res.get("status"),
        }
        for col, res in report["per_feature"].items()
    ]
    return pd.DataFrame(rows)
