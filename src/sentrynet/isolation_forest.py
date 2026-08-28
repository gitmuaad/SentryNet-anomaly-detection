"""Isolation Forest detector.

Fit on Normal training rows only. contamination is tuned on the validation set (PortScan
excluded) by PR-AUC.

Worth knowing: in scikit-learn, contamination doesn't change score_samples, only the
offset used to threshold it. Since PR-AUC only depends on score ranking, every
contamination value gives the same PR-AUC for a given forest. tune_isolation_forest
detects these ties and breaks them by validation F1 at the model's own predict() boundary,
then false-positive rate, then smallest contamination.

Score convention: anomaly_score = -decision_function, so higher means more anomalous.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

MODEL_NAME = "isolation_forest"
PR_AUC_TIE_TOLERANCE = 1e-12


def anomaly_score(model: IsolationForest, X: np.ndarray) -> np.ndarray:  # noqa: N803
    """Anomaly score; higher means more anomalous."""
    return -np.asarray(model.decision_function(X), dtype="float64")


def fit_isolation_forest(
    X_train: np.ndarray,  # noqa: N803
    params: Mapping[str, Any],
    seed: int,
) -> IsolationForest:
    """Fit an Isolation Forest on the Normal training matrix only."""
    model = IsolationForest(
        n_estimators=int(params["n_estimators"]),
        max_samples=params["max_samples"],
        contamination=float(params["contamination"]),
        random_state=int(seed),
        n_jobs=-1,
    )
    model.fit(X_train)
    return model


def isolation_forest_grid(grid_cfg: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Expand the configured grid into a deterministic, ordered list of parameter dicts."""
    return [
        {"n_estimators": int(n), "max_samples": m, "contamination": float(c)}
        for n, m, c in product(
            grid_cfg["n_estimators"], grid_cfg["max_samples"], grid_cfg["contamination"]
        )
    ]


def _default_boundary_metrics(
    model: IsolationForest, X_val: np.ndarray, y_val: np.ndarray  # noqa: N803
) -> dict[str, float]:
    """Precision/recall/F1/FPR at the model's own ``predict()`` boundary."""
    predicted_attack = (model.predict(X_val) == -1).astype(int)
    y = np.asarray(y_val, dtype=int)
    tp = int(((predicted_attack == 1) & (y == 1)).sum())
    fp = int(((predicted_attack == 1) & (y == 0)).sum())
    fn = int(((predicted_attack == 0) & (y == 1)).sum())
    tn = int(((predicted_attack == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "default_boundary_precision": precision,
        "default_boundary_recall": recall,
        "default_boundary_f1": f1,
        "default_boundary_fpr": fpr,
        "default_boundary_false_positives": fp,
    }


def tune_isolation_forest(
    X_train: np.ndarray,  # noqa: N803
    X_val: np.ndarray,  # noqa: N803
    y_val: np.ndarray,
    grid_cfg: Mapping[str, Sequence[Any]],
    seed: int,
) -> dict[str, Any]:
    """Grid-search on the validation set; returns all results plus the tie analysis."""
    configs = isolation_forest_grid(grid_cfg)
    results: list[dict[str, Any]] = []
    forest_cache: dict[tuple[Any, Any], IsolationForest] = {}

    for params in configs:
        key = (params["n_estimators"], str(params["max_samples"]))
        model = fit_isolation_forest(X_train, params, seed)
        forest_cache.setdefault(key, model)
        scores = anomaly_score(model, X_val)
        record: dict[str, Any] = {
            "params": dict(params),
            "val_pr_auc": float(average_precision_score(y_val, scores)),
            "forest_key": f"n_estimators={params['n_estimators']},max_samples={params['max_samples']}",
        }
        record.update(_default_boundary_metrics(model, X_val, y_val))
        results.append(record)

    best_pr_auc = max(r["val_pr_auc"] for r in results)
    tied = [r for r in results if abs(r["val_pr_auc"] - best_pr_auc) <= PR_AUC_TIE_TOLERANCE]

    def tie_break_key(record: dict[str, Any]) -> tuple:
        return (
            -record["default_boundary_f1"],
            record["default_boundary_fpr"],
            record["params"]["contamination"],
            record["params"]["n_estimators"],
            str(record["params"]["max_samples"]),
        )

    best = sorted(tied, key=tie_break_key)[0]

    contamination_values = sorted({r["params"]["contamination"] for r in tied})
    tie_analysis = {
        "n_configurations": len(results),
        "best_val_pr_auc": best_pr_auc,
        "n_tied_at_best_pr_auc": len(tied),
        "tied_contamination_values": contamination_values,
        "pr_auc_is_contamination_invariant": len(contamination_values) > 1,
        "explanation": (
            "contamination does not change IsolationForest.score_samples; it only shifts "
            "offset_, and decision_function = score_samples - offset_. A constant shift "
            "cannot change score ranking, and PR-AUC depends only on ranking. Identical "
            "PR-AUC across contamination values is therefore expected, not a bug."
        ),
        "tie_break_rule": (
            "1) highest validation PR-AUC; 2) highest validation F1 at the model's own "
            "predict() boundary (which contamination does control); 3) lowest validation "
            "false-positive rate on Normal rows; 4) smallest contamination; 5) deterministic "
            "sorted parameter order."
        ),
    }

    return {
        "model": MODEL_NAME,
        "results": results,
        "best_params": best["params"],
        "best_val_pr_auc": best["val_pr_auc"],
        "best_record": best,
        "tie_analysis": tie_analysis,
        "seed": int(seed),
    }
