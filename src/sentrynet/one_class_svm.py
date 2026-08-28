"""One-Class SVM detector (RBF kernel), fit on Normal training rows only.

nu and gamma are tuned on the validation set; PortScan never reaches this module.

Training cost is roughly quadratic in row count, so the Normal training partition is
subsampled with a fixed seed before fitting. The sample size and seed are recorded in the
result rather than applied silently. Set one_class_svm.train_subsample to null to disable it.

Score convention: anomaly_score = -decision_function, so higher means more anomalous.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.svm import OneClassSVM

MODEL_NAME = "one_class_svm"


def anomaly_score(model: OneClassSVM, X: np.ndarray) -> np.ndarray:  # noqa: N803
    """Anomaly score; higher means more anomalous."""
    return -np.asarray(model.decision_function(X), dtype="float64")


def subsample_normal_train(
    X_train: np.ndarray,  # noqa: N803
    n_samples: int | None,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Deterministically subsample Normal training rows, returning the sample and a log record."""
    population = int(X_train.shape[0])
    if n_samples is None or n_samples >= population:
        return X_train, {
            "subsampled": False,
            "population_size": population,
            "train_sample_size": population,
            "seed": int(seed),
            "reason": "No subsampling: configured size is null or >= the available Normal rows.",
        }
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(population, size=int(n_samples), replace=False))
    record = {
        "subsampled": True,
        "population_size": population,
        "train_sample_size": int(n_samples),
        "seed": int(seed),
        "reason": (
            "OneClassSVM training cost is approximately quadratic in the number of rows. "
            "A fixed-seed subsample of Normal TRAINING rows only is used for both tuning and "
            "the final fit, so the selected hyperparameters match the fitted model."
        ),
    }
    return X_train[idx], record


def one_class_svm_grid(grid_cfg: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Expand the configured grid into a deterministic, ordered list of parameter dicts."""
    return [
        {"nu": float(nu), "gamma": gamma}
        for nu, gamma in product(grid_cfg["nu"], grid_cfg["gamma"])
    ]


def fit_one_class_svm(
    X_train: np.ndarray,  # noqa: N803
    params: Mapping[str, Any],
    kernel: str = "rbf",
) -> OneClassSVM:
    """Fit a One-Class SVM on the (possibly subsampled) Normal training matrix."""
    model = OneClassSVM(kernel=kernel, nu=float(params["nu"]), gamma=params["gamma"])
    model.fit(X_train)
    return model


def _default_boundary_metrics(
    model: OneClassSVM, X_val: np.ndarray, y_val: np.ndarray  # noqa: N803
) -> dict[str, float]:
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
        "n_support_vectors": int(model.support_vectors_.shape[0]),
    }


def tune_one_class_svm(
    X_train: np.ndarray,  # noqa: N803
    X_val: np.ndarray,  # noqa: N803
    y_val: np.ndarray,
    grid_cfg: Mapping[str, Sequence[Any]],
    seed: int,
    kernel: str = "rbf",
    train_subsample: int | None = None,
) -> dict[str, Any]:
    """Grid-search nu and gamma on the validation set."""
    X_fit, subsample_record = subsample_normal_train(X_train, train_subsample, seed)
    configs = one_class_svm_grid(grid_cfg)

    results: list[dict[str, Any]] = []
    for params in configs:
        model = fit_one_class_svm(X_fit, params, kernel=kernel)
        scores = anomaly_score(model, X_val)
        record: dict[str, Any] = {
            "params": {"nu": params["nu"], "gamma": params["gamma"]},
            "val_pr_auc": float(average_precision_score(y_val, scores)),
        }
        record.update(_default_boundary_metrics(model, X_val, y_val))
        results.append(record)

    best_pr_auc = max(r["val_pr_auc"] for r in results)
    tied = [r for r in results if abs(r["val_pr_auc"] - best_pr_auc) <= 1e-12]

    def tie_break_key(record: dict[str, Any]) -> tuple:
        return (
            -record["default_boundary_f1"],
            record["default_boundary_fpr"],
            record["params"]["nu"],
            str(record["params"]["gamma"]),
        )

    best = sorted(tied, key=tie_break_key)[0]

    return {
        "model": MODEL_NAME,
        "kernel": kernel,
        "results": results,
        "best_params": best["params"],
        "best_val_pr_auc": best["val_pr_auc"],
        "best_record": best,
        "n_tied_at_best_pr_auc": len(tied),
        "tie_break_rule": (
            "1) highest validation PR-AUC; 2) highest validation F1 at the model's own "
            "predict() boundary; 3) lowest validation false-positive rate; 4) smallest nu; "
            "5) deterministic sorted gamma order."
        ),
        "subsample": subsample_record,
        "seed": int(seed),
    }
