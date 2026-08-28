"""Prevalence sensitivity analysis (the 95/5 case).

The held-out test slice is attack-heavy, the reverse of real traffic. This re-samples
held-out rows only into a chosen Normal/Attack mix so the alert burden can be reported for a
more realistic low-prevalence regime.

Normal rows are never pulled from training -- only from the frozen test partition. Any
"implied daily alert volume" is a scenario driven by a configurable assumption
(sensitivity.assumed_daily_flow_volume), since the dataset has no timestamps and no real
daily volume can be measured from it.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from sentrynet.evaluation import NORMAL_LABEL, binary_labels, confusion_counts

SCENARIO_LABEL = "SCENARIO / SENSITIVITY ANALYSIS - not an observed measurement"


def build_prevalence_slice(
    held_out: pd.DataFrame,
    normal_fraction: float,
    seed: int,
    label_column: str = "attack_type",
    normal_label: str = NORMAL_LABEL,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Re-sample the held-out partition to a target Normal fraction, without replacement."""
    if not 0.0 < normal_fraction < 1.0:
        raise ValueError("normal_fraction must be strictly between 0 and 1")

    rng = np.random.default_rng(seed)
    normal_pool = held_out.loc[held_out[label_column] == normal_label]
    attack_pool = held_out.loc[held_out[label_column] != normal_label]
    if normal_pool.empty or attack_pool.empty:
        raise ValueError("Held-out data must contain both Normal and attack rows.")

    attack_fraction = 1.0 - normal_fraction
    # Try using every available Normal row first.
    n_attack_if_all_normal = int(round(len(normal_pool) * attack_fraction / normal_fraction))
    if n_attack_if_all_normal <= len(attack_pool):
        n_normal, n_attack = len(normal_pool), n_attack_if_all_normal
        binding = "normal_pool"
    else:
        n_attack = len(attack_pool)
        n_normal = int(round(len(attack_pool) * normal_fraction / attack_fraction))
        n_normal = min(n_normal, len(normal_pool))
        binding = "attack_pool"

    normal_idx = rng.choice(len(normal_pool), size=n_normal, replace=False)
    attack_idx = rng.choice(len(attack_pool), size=n_attack, replace=False)
    slice_df = pd.concat(
        [normal_pool.iloc[np.sort(normal_idx)], attack_pool.iloc[np.sort(attack_idx)]]
    )
    # Deterministic interleaving so that 1,000-row windows are not "all Normal then all attack".
    order = rng.permutation(len(slice_df))
    slice_df = slice_df.iloc[order].reset_index(drop=True)

    record = {
        "label": SCENARIO_LABEL,
        "requested_normal_fraction": float(normal_fraction),
        "achieved_normal_fraction": float(n_normal / (n_normal + n_attack)),
        "n_rows": int(n_normal + n_attack),
        "n_normal": int(n_normal),
        "n_attack": int(n_attack),
        "source": "held-out FINAL TEST partition only",
        "training_rows_used": 0,
        "sampling": "without replacement",
        "seed": int(seed),
        "binding_constraint": binding,
        "class_counts": {k: int(v) for k, v in slice_df[label_column].value_counts().items()},
    }
    return slice_df, record


def evaluate_slice(
    scores: np.ndarray,
    attack_type: pd.Series,
    threshold: float,
    normal_label: str = NORMAL_LABEL,
) -> dict[str, Any]:
    """Alert-burden metrics for a prevalence slice."""
    scores = np.asarray(scores, dtype="float64")
    y = binary_labels(attack_type, normal_label)
    y_pred = (scores >= threshold).astype(int)
    cm = confusion_counts(y, y_pred)

    tp, fp, fn = cm["true_positives"], cm["false_positives"], cm["false_negatives"]
    n_rows = int(len(y))
    n_normal = cm["true_negatives"] + fp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    alerts = int(y_pred.sum())

    return {
        "n_rows": n_rows,
        "n_normal_rows": int(n_normal),
        "alerts": alerts,
        "alert_rate": alerts / n_rows if n_rows else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0,
        "false_positives": fp,
        "false_positive_rate_among_normal": fp / n_normal if n_normal else 0.0,
        "false_positives_per_1000_rows": 1000.0 * fp / n_rows if n_rows else 0.0,
        "expected_alerts_per_10000_rows": 10000.0 * alerts / n_rows if n_rows else 0.0,
        "confusion_matrix": cm,
    }


def implied_daily_alerts(
    slice_metrics: Mapping[str, Any], assumed_daily_flow_volume: int
) -> dict[str, Any]:
    """Project daily alert volume from an assumed flow rate -- a scenario, not a measurement."""
    rate = float(slice_metrics["alert_rate"])
    fp_rate_all_rows = (
        float(slice_metrics["false_positives"]) / float(slice_metrics["n_rows"])
        if slice_metrics["n_rows"]
        else 0.0
    )
    return {
        "label": SCENARIO_LABEL,
        "assumption_name": "assumed_daily_flow_volume",
        "assumed_daily_flow_volume": int(assumed_daily_flow_volume),
        "assumption_source": (
            "USER-CONFIGURED in config/config.yaml. The dataset contains no timestamps, so a "
            "real daily flow volume cannot be measured from it and none is claimed."
        ),
        "implied_total_alerts_per_day": rate * assumed_daily_flow_volume,
        "implied_false_positive_alerts_per_day": fp_rate_all_rows * assumed_daily_flow_volume,
        "how_to_change": (
            "Edit sensitivity.assumed_daily_flow_volume in config/config.yaml and re-run "
            "scripts/evaluate.py."
        ),
    }
