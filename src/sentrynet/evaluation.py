"""Evaluation: PR-AUC, operating threshold selection, and false-positive windows.

select_operating_threshold refuses to run if PortScan is in the data it's given, since the
threshold has to come from validation only. A "window" here means a block of 1,000 rows --
there's no clock in this dataset, so it's never a time window.

PR-AUC is reported next to its no-skill baseline (the positive-class prevalence), since on
an attack-heavy slice a random ranker already scores high and the bare number is misleading
on its own.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

WINDOW_SIZE = 1000
NORMAL_LABEL = "Normal"
UNSEEN_ATTACKS = ("PortScan",)


def binary_labels(attack_type: pd.Series | Sequence[str], normal_label: str = NORMAL_LABEL) -> np.ndarray:
    """1 = attack, 0 = Normal."""
    series = pd.Series(list(attack_type)) if not isinstance(attack_type, pd.Series) else attack_type
    return (series.to_numpy(dtype=object) != normal_label).astype(int)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return {
        "true_negatives": int(((y_pred == 0) & (y_true == 0)).sum()),
        "false_positives": int(((y_pred == 1) & (y_true == 0)).sum()),
        "false_negatives": int(((y_pred == 0) & (y_true == 1)).sum()),
        "true_positives": int(((y_pred == 1) & (y_true == 1)).sum()),
    }


def _prf(cm: Mapping[str, int]) -> dict[str, float]:
    tp, fp, fn, tn = (
        cm["true_positives"],
        cm["false_positives"],
        cm["false_negatives"],
        cm["true_negatives"],
    )
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "false_positives": fp,
        "n_normal_rows": fp + tn,
    }


def per_class_recall(
    attack_type: pd.Series, scores: np.ndarray, threshold: float, normal_label: str = NORMAL_LABEL
) -> dict[str, float]:
    """Recall for each attack class present in the slice."""
    flagged = np.asarray(scores, dtype="float64") >= threshold
    out: dict[str, float] = {}
    for cls in sorted(set(attack_type) - {normal_label}):
        mask = (attack_type == cls).to_numpy()
        out[cls] = float(flagged[mask].mean()) if mask.any() else float("nan")
    return out


def candidate_thresholds(scores: np.ndarray, n_candidates: int) -> np.ndarray:
    """Deterministic candidate grid drawn from the observed score distribution."""
    scores = np.asarray(scores, dtype="float64")
    unique = np.unique(scores)
    if unique.size <= n_candidates:
        candidates = unique
    else:
        quantiles = np.linspace(0.0, 1.0, int(n_candidates))
        candidates = np.unique(np.quantile(scores, quantiles))
    # Add a threshold just above the maximum so "flag nothing" is representable.
    span = float(unique[-1] - unique[0]) if unique.size > 1 else 1.0
    return np.unique(np.concatenate([candidates, [unique[-1] + span * 1e-6]]))


def assert_no_unseen_attacks(
    attack_type: Iterable[str], unseen: Sequence[str] = UNSEEN_ATTACKS
) -> None:
    """Guard: the unseen-attack class must never influence tuning or threshold selection."""
    present = sorted(set(attack_type) & set(unseen))
    if present:
        raise AssertionError(
            f"Unseen attack class(es) {present} were passed to a tuning/threshold routine. "
            "The unseen-attack protocol forbids using PortScan for tuning or thresholding."
        )


def select_operating_threshold(
    scores: np.ndarray,
    attack_type: pd.Series,
    ddos_recall_target: float = 0.90,
    bruteforce_recall_target: float = 0.90,
    n_candidates: int = 400,
    normal_label: str = NORMAL_LABEL,
    unseen: Sequence[str] = UNSEEN_ATTACKS,
) -> dict[str, Any]:
    """Choose the decision threshold on validation data only.

    Prefers thresholds meeting both recall targets, ranked by F1/precision/FP count. If no
    threshold meets both targets, falls back to the best achievable tradeoff and flags it.
    """
    assert_no_unseen_attacks(attack_type, unseen)
    scores = np.asarray(scores, dtype="float64")
    y = binary_labels(attack_type, normal_label)

    rows: list[dict[str, Any]] = []
    for thr in candidate_thresholds(scores, n_candidates):
        y_pred = (scores >= thr).astype(int)
        cm = confusion_counts(y, y_pred)
        metrics = _prf(cm)
        recalls = per_class_recall(attack_type, scores, thr, normal_label)
        rows.append(
            {
                "threshold": float(thr),
                **metrics,
                "ddos_recall": float(recalls.get("DDoS", float("nan"))),
                "bruteforce_recall": float(recalls.get("BruteForce", float("nan"))),
                "alert_rate": float(y_pred.mean()),
            }
        )

    feasible = [
        r
        for r in rows
        if r["ddos_recall"] >= ddos_recall_target and r["bruteforce_recall"] >= bruteforce_recall_target
    ]

    if feasible:
        best = sorted(
            feasible,
            key=lambda r: (-r["f1"], -r["precision"], r["false_positives"], -r["threshold"]),
        )[0]
        rule_applied = (
            "Both recall targets are satisfiable. Among all thresholds meeting "
            f"DDoS recall >= {ddos_recall_target} and BruteForce recall >= "
            f"{bruteforce_recall_target}, selected the highest validation F1, then highest "
            "precision, then fewest false positives, then the highest threshold."
        )
        targets_met = True
    else:
        best = sorted(
            rows,
            key=lambda r: (
                -min(r["ddos_recall"], r["bruteforce_recall"]),
                -r["f1"],
                r["false_positives"],
            ),
        )[0]
        rule_applied = (
            "NO threshold satisfied both recall targets on validation. Fallback rule applied: "
            "maximise the weaker of the two class recalls, then F1, then fewest false "
            "positives. The unmet target is reported as UNMET and is NOT presented as success."
        )
        targets_met = False

    return {
        "threshold": float(best["threshold"]),
        "selected_on": "validation (Normal + DDoS + BruteForce). PortScan excluded.",
        "rule_applied": rule_applied,
        "targets": {
            "ddos_recall_target": ddos_recall_target,
            "bruteforce_recall_target": bruteforce_recall_target,
        },
        "targets_met": targets_met,
        "ddos_recall_target_met": bool(best["ddos_recall"] >= ddos_recall_target),
        "bruteforce_recall_target_met": bool(best["bruteforce_recall"] >= bruteforce_recall_target),
        "validation_metrics_at_threshold": {k: v for k, v in best.items()},
        "n_candidates_evaluated": len(rows),
    }


def evaluate_detector(
    name: str,
    scores: np.ndarray,
    attack_type: pd.Series,
    threshold: float,
    normal_label: str = NORMAL_LABEL,
    unseen: Sequence[str] = UNSEEN_ATTACKS,
    latency_seconds_per_10k: float | None = None,
) -> dict[str, Any]:
    """Full metric set for one detector on one slice."""
    scores = np.asarray(scores, dtype="float64")
    y = binary_labels(attack_type, normal_label)
    y_pred = (scores >= threshold).astype(int)
    cm = confusion_counts(y, y_pred)
    metrics = _prf(cm)
    recalls = per_class_recall(attack_type, scores, threshold, normal_label)
    prevalence = float(y.mean())

    record: dict[str, Any] = {
        "detector": name,
        "threshold": float(threshold),
        "n_rows": int(len(y)),
        "pr_auc": float(average_precision_score(y, scores)),
        "pr_auc_no_skill": prevalence,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "confusion_matrix": cm,
        "confusion_matrix_layout": "rows = actual [Normal, Attack], cols = predicted [Normal, Attack]",
        "false_positives": metrics["false_positives"],
        "false_positive_rate": metrics["false_positive_rate"],
        "n_normal_rows": metrics["n_normal_rows"],
        "alert_rate": float(y_pred.mean()),
        "per_class_recall": recalls,
        "ddos_recall": recalls.get("DDoS"),
        "bruteforce_recall": recalls.get("BruteForce"),
        "prevalence_attack": prevalence,
        "prevalence_normal": 1.0 - prevalence,
    }
    record["pr_auc_lift_over_no_skill"] = record["pr_auc"] - prevalence

    for cls in unseen:
        if cls in set(attack_type):
            record["unseen_attack_recall_portscan"] = recalls.get(cls)
            record["unseen_attack_class"] = cls
            record["unseen_attack_note"] = (
                f"{cls} was excluded from all training, hyperparameter selection, and "
                "threshold tuning. This value is the Unseen Attack Recall."
            )
    if latency_seconds_per_10k is not None:
        record["latency_seconds_per_10k_rows"] = float(latency_seconds_per_10k)
    return record


def prevalence_report(
    test_attack_type: pd.Series,
    dataset_class_counts: Mapping[str, int],
    normal_label: str = NORMAL_LABEL,
) -> dict[str, Any]:
    """Dataset-level vs. actual held-out-test prevalence, reported side by side."""
    total = sum(dataset_class_counts.values())
    dataset_normal = int(dataset_class_counts.get(normal_label, 0))
    y = binary_labels(test_attack_type, normal_label)
    return {
        "dataset_characteristic": {
            "n_rows": int(total),
            "normal_fraction": dataset_normal / total if total else 0.0,
            "attack_fraction": 1.0 - (dataset_normal / total if total else 0.0),
            "note": (
                "The raw dataset is approximately 25% Normal / 75% Attack in the binary view. "
                "This is a characteristic of the synthetic dataset, not of real network traffic."
            ),
        },
        "actual_final_test": {
            "n_rows": int(len(y)),
            "normal_fraction": float(1.0 - y.mean()),
            "attack_fraction": float(y.mean()),
            "class_counts": {k: int(v) for k, v in test_attack_type.value_counts().items()},
            "note": (
                "The held-out test prevalence is NOT forced to 25/75. It follows from the "
                "frozen split: Normal rows are shared across train/validation/test while all "
                "PortScan rows land in test, so the test slice is attack-heavy. PR-AUC is "
                "prevalence-sensitive, so pr_auc_no_skill is always reported next to it, and "
                "the 95/5 sensitivity slice covers the low-prevalence operating regime."
            ),
        },
    }


def false_positive_windows(
    attack_type: pd.Series,
    scores: np.ndarray,
    threshold: float,
    window_size: int = WINDOW_SIZE,
    normal_label: str = NORMAL_LABEL,
) -> pd.DataFrame:
    """Split the slice into sequential windows of window_size rows (not time windows)."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    scores = np.asarray(scores, dtype="float64")
    labels = pd.Series(list(attack_type)).reset_index(drop=True)
    flagged = scores >= threshold
    is_normal = (labels != normal_label).to_numpy() == False  # noqa: E712 - explicit intent

    rows = []
    for start in range(0, len(labels), window_size):
        end = min(start + window_size, len(labels))
        sl = slice(start, end)
        n_normal = int(is_normal[sl].sum())
        fp = int((flagged[sl] & is_normal[sl]).sum())
        rows.append(
            {
                "window_index": start // window_size,
                "row_start": start,
                "row_end_exclusive": end,
                "total_rows": end - start,
                "n_normal_rows": n_normal,
                "false_positives": fp,
                "false_positive_rate_among_normal": (fp / n_normal) if n_normal else 0.0,
                "alerts": int(flagged[sl].sum()),
                "window_definition": "block of consecutive rows (NOT a time window)",
            }
        )
    return pd.DataFrame(rows)


def target_status(
    pr_auc: float,
    ddos_recall: float | None,
    bruteforce_recall: float | None,
    latency_seconds_per_10k: float | None,
    pr_auc_target: float = 0.85,
    recall_target: float = 0.90,
    latency_target: float = 2.0,
) -> dict[str, Any]:
    """MET / UNMET / NOT MEASURED status for each target."""

    def status(value: float | None, target: float, comparison: str) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "NOT MEASURED"
        ok = value >= target if comparison == "ge" else value < target
        return "MET" if ok else "UNMET"

    return {
        "pr_auc": {
            "target": f">= {pr_auc_target}",
            "actual": pr_auc,
            "status": status(pr_auc, pr_auc_target, "ge"),
        },
        "ddos_recall": {
            "target": f">= {recall_target}",
            "actual": ddos_recall,
            "status": status(ddos_recall, recall_target, "ge"),
        },
        "bruteforce_recall": {
            "target": f">= {recall_target}",
            "actual": bruteforce_recall,
            "status": status(bruteforce_recall, recall_target, "ge"),
        },
        "latency_seconds_per_10k_rows": {
            "target": f"< {latency_target}",
            "actual": latency_seconds_per_10k,
            "status": status(latency_seconds_per_10k, latency_target, "lt"),
        },
    }
