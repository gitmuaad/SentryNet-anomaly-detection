"""Phase 3 — fit the preprocessor and the three detectors, tune on validation, freeze thresholds.

    python scripts/train.py                 # active protocol from config
    python scripts/train.py --protocol row_order

Order of operations, which is the whole point of the protocol:

1. ``train`` (Normal rows only) fits the preprocessor.
2. ``train`` fits the statistical baseline, Isolation Forest, and One-Class SVM.
3. ``validation`` (Normal + DDoS + BruteForce) selects hyperparameters **and** the operating
   threshold. PortScan is not present and a runtime guard enforces that.
4. The threshold is written to disk here and only **read** by ``evaluate.py``, so it
   is frozen before the final test is ever touched.
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sentrynet import baseline as bl  # noqa: E402
from sentrynet import isolation_forest as iforest  # noqa: E402
from sentrynet import one_class_svm as ocsvm  # noqa: E402
from sentrynet.config import load_config  # noqa: E402
from sentrynet.data import file_fingerprint  # noqa: E402
from sentrynet.evaluation import (  # noqa: E402
    assert_no_unseen_attacks,
    binary_labels,
    evaluate_detector,
    select_operating_threshold,
)
from sentrynet.features import (  # noqa: E402
    DERIVED_NUMERIC,
    SOURCE_CATEGORICAL,
    SOURCE_NUMERIC,
    add_derived_features,
    build_feature_frame,
)
from sentrynet.inference import KIND_BASELINE, KIND_IFOREST, KIND_OCSVM  # noqa: E402
from sentrynet.monitoring import build_psi_reference  # noqa: E402
from sentrynet.persistence import (  # noqa: E402
    run_record,
    save_json,
    save_scoring_bundle,
)
from sentrynet.preprocessing import fit_preprocessor_on_normal_train  # noqa: E402
from sentrynet.splits import load_splits, subset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=None, help="random | row_order")
    args = parser.parse_args()

    cfg = load_config()
    cfg.ensure_dirs()
    protocol = args.protocol or cfg["splits"]["active_protocol"]
    label = cfg["dataset"]["label_column"]
    normal_label = cfg["dataset"]["normal_label"]
    seed = cfg.seed

    clean_path = cfg.path("clean_csv")
    if not clean_path.exists():
        print("[FAIL] Run `python scripts/prepare_data.py` first.", file=sys.stderr)
        return 1
    clean = pd.read_csv(clean_path)
    splits = load_splits(cfg.path("splits_dir") / protocol)

    train = subset(clean, splits["train"])
    validation = subset(clean, splits["validation"])

    # --- Guards that make the protocol real rather than aspirational -------------------
    train_classes = set(train[label].unique())
    if train_classes != {normal_label}:
        print(f"[FAIL] Train partition is not Normal-only: {train_classes}", file=sys.stderr)
        return 1
    assert_no_unseen_attacks(validation[label], cfg["splits"]["unseen_attacks"])

    print(f"Protocol: {protocol}")
    print(f"  train      {len(train):>7,} rows  (Normal only)")
    print(f"  validation {len(validation):>7,} rows  {dict(validation[label].value_counts())}")

    # --- Preprocessing: fit on Normal TRAIN only --------------------------------------
    numeric_features = list(SOURCE_NUMERIC) + list(DERIVED_NUMERIC)
    categorical_features = list(SOURCE_CATEGORICAL)
    drop = set(cfg["correlation"]["drop_features"])
    if drop:
        numeric_features = [c for c in numeric_features if c not in drop]

    X_train_raw = build_feature_frame(train)
    X_val_raw = build_feature_frame(validation)
    preprocessor = fit_preprocessor_on_normal_train(
        X_train_raw,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        log1p_features=cfg["preprocessing"]["log1p_features"],
    )
    X_train = preprocessor.transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    y_val = binary_labels(validation[label], normal_label)
    print(f"  feature matrix: {X_train.shape[1]} columns -> {preprocessor.feature_names_out}")

    train_enriched = add_derived_features(train)
    val_enriched = add_derived_features(validation)

    selection: dict = {}
    bundles: dict = {}

    # --- 1. Statistical baseline ------------------------------------------------------
    print("\n[1/3] Statistical baseline ...")
    baseline_results = []
    for params in bl.baseline_grid(cfg["baseline"]["grid"]):
        model = bl.fit_baseline(
            train_enriched, cfg["baseline"]["core_features"], params,
            min_band_width=cfg["baseline"]["min_band_width"],
        )
        scores = model.score(val_enriched)
        from sklearn.metrics import average_precision_score

        baseline_results.append(
            {"params": params, "val_pr_auc": float(average_precision_score(y_val, scores))}
        )
    best_baseline = sorted(
        baseline_results,
        key=lambda r: (-r["val_pr_auc"], r["params"]["rule"], r["params"]["percentile"]),
    )[0]
    baseline_model = bl.fit_baseline(
        train_enriched, cfg["baseline"]["core_features"], best_baseline["params"],
        min_band_width=cfg["baseline"]["min_band_width"],
    )
    selection[KIND_BASELINE] = {
        "model": KIND_BASELINE,
        "results": baseline_results,
        "best_params": best_baseline["params"],
        "best_val_pr_auc": best_baseline["val_pr_auc"],
        "fitted_statistics": baseline_model.stats_,
        "note": (
            "Statistics are estimated from Normal TRAINING rows only. failed_logins has zero "
            "variance in Normal training data, which is why every denominator is floored at "
            "one integer unit."
        ),
    }
    print(f"      best {best_baseline['params']}  val PR-AUC = {best_baseline['val_pr_auc']:.6f}")

    # --- 2. Isolation Forest ----------------------------------------------------------
    print("[2/3] Isolation Forest ...")
    if_result = iforest.tune_isolation_forest(
        X_train, X_val, y_val, cfg["isolation_forest"]["grid"], seed
    )
    if_model = iforest.fit_isolation_forest(X_train, if_result["best_params"], seed)
    selection[KIND_IFOREST] = if_result
    tie = if_result["tie_analysis"]
    print(f"      best {if_result['best_params']}  val PR-AUC = {if_result['best_val_pr_auc']:.6f}")
    print(
        f"      PR-AUC ties: {tie['n_tied_at_best_pr_auc']}/{tie['n_configurations']} configs; "
        f"contamination-invariant = {tie['pr_auc_is_contamination_invariant']}"
    )

    # --- 3. One-Class SVM -------------------------------------------------------------
    print("[3/3] One-Class SVM (this is the slow one) ...")
    svm_result = ocsvm.tune_one_class_svm(
        X_train, X_val, y_val, cfg["one_class_svm"]["grid"], seed,
        kernel=cfg["one_class_svm"]["kernel"],
        train_subsample=cfg["one_class_svm"]["train_subsample"],
    )
    X_fit_svm, _ = ocsvm.subsample_normal_train(
        X_train, cfg["one_class_svm"]["train_subsample"], seed
    )
    svm_model = ocsvm.fit_one_class_svm(
        X_fit_svm, svm_result["best_params"], kernel=cfg["one_class_svm"]["kernel"]
    )
    selection[KIND_OCSVM] = svm_result
    sub = svm_result["subsample"]
    print(f"      best {svm_result['best_params']}  val PR-AUC = {svm_result['best_val_pr_auc']:.6f}")
    print(
        f"      subsampled: {sub['subsampled']} "
        f"({sub['train_sample_size']:,} of {sub['population_size']:,} Normal train rows, seed {sub['seed']})"
    )

    # --- Operating thresholds, chosen on validation only ------------------------------
    print("\nSelecting operating thresholds on validation (PortScan excluded) ...")
    scorers = {
        KIND_BASELINE: lambda frame: baseline_model.score(add_derived_features(frame)),
        KIND_IFOREST: lambda frame: iforest.anomaly_score(
            if_model, preprocessor.transform(build_feature_frame(frame))
        ),
        KIND_OCSVM: lambda frame: ocsvm.anomaly_score(
            svm_model, preprocessor.transform(build_feature_frame(frame))
        ),
    }

    thresholds: dict = {}
    validation_metrics: dict = {}
    for kind, score_fn in scorers.items():
        val_scores = score_fn(validation)
        chosen = select_operating_threshold(
            val_scores,
            validation[label],
            ddos_recall_target=float(cfg["threshold"]["ddos_recall_target"]),
            bruteforce_recall_target=float(cfg["threshold"]["bruteforce_recall_target"]),
            n_candidates=int(cfg["threshold"]["n_candidates"]),
            normal_label=normal_label,
            unseen=tuple(cfg["splits"]["unseen_attacks"]),
        )
        thresholds[kind] = chosen
        validation_metrics[kind] = evaluate_detector(
            kind, val_scores, validation[label], chosen["threshold"], normal_label=normal_label,
            unseen=tuple(cfg["splits"]["unseen_attacks"]),
        )
        status = "MET" if chosen["targets_met"] else "UNMET"
        print(
            f"  {kind:<22} threshold = {chosen['threshold']:>12.6f}  "
            f"val F1 = {validation_metrics[kind]['f1']:.4f}  "
            f"DDoS/BF recall targets: {status}"
        )

    # --- Primary model: chosen by validation PR-AUC only ------------------------------
    configured = cfg["evaluation"]["primary_model"]
    if configured:
        primary = configured
        primary_reason = "explicitly configured in config/config.yaml"
    else:
        primary = max(validation_metrics, key=lambda k: validation_metrics[k]["pr_auc"])
        primary_reason = "highest validation PR-AUC (test data was not consulted)"
    print(f"\nPrimary detector: {primary}  ({primary_reason})")

    # --- Persist -----------------------------------------------------------------------
    artifacts = cfg.path("artifacts_dir") / protocol
    artifacts.mkdir(parents=True, exist_ok=True)
    fingerprint = file_fingerprint(cfg.path("raw_csv"))

    save_scoring_bundle(artifacts / "preprocessor.joblib", {
        "preprocessor": preprocessor,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "log1p_features": list(cfg["preprocessing"]["log1p_features"]),
        "feature_names_out": preprocessor.feature_names_out,
        "fit_on": "train partition, Normal rows only",
        "seed": seed,
    })
    save_scoring_bundle(artifacts / "baseline.joblib", {
        "kind": KIND_BASELINE, "model": baseline_model,
        "threshold": thresholds[KIND_BASELINE]["threshold"],
        "params": best_baseline["params"], "preprocessor": None, "seed": seed,
    })
    save_scoring_bundle(artifacts / "isolation_forest.joblib", {
        "kind": KIND_IFOREST, "model": if_model,
        "threshold": thresholds[KIND_IFOREST]["threshold"],
        "params": if_result["best_params"], "preprocessor": preprocessor, "seed": seed,
    })
    save_scoring_bundle(artifacts / "one_class_svm.joblib", {
        "kind": KIND_OCSVM, "model": svm_model,
        "threshold": thresholds[KIND_OCSVM]["threshold"],
        "params": svm_result["best_params"], "preprocessor": preprocessor, "seed": seed,
    })

    psi_reference = build_psi_reference(
        train, cfg["monitoring"]["psi_features"], int(cfg["monitoring"]["psi_bins"])
    )
    save_scoring_bundle(artifacts / "psi_reference.joblib", {"psi_reference": psi_reference})

    save_json(artifacts / "thresholds.json", thresholds)
    save_json(artifacts / "model_metadata.json", {
        "run": run_record(cfg.data, fingerprint, protocol),
        "primary_model": primary,
        "primary_model_reason": primary_reason,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_names_out": preprocessor.feature_names_out,
        "best_params": {k: selection[k].get("best_params") for k in selection},
        "one_class_svm_subsample": svm_result["subsample"],
        "autoencoder": (
            "Autoencoder was optional in the approved Action Plan and was not included due to "
            "schedule/scope."
        ),
    })
    save_json(cfg.path("metrics_dir") / f"model_selection_{protocol}.json", {
        "run": run_record(cfg.data, fingerprint, protocol),
        "primary_model": primary,
        "primary_model_reason": primary_reason,
        "selection": selection,
        "thresholds": thresholds,
        "validation_metrics_at_threshold": validation_metrics,
        "protocol_note": (
            "Hyperparameters and thresholds were selected on the validation partition only "
            "(Normal + DDoS + BruteForce). PortScan and the test partition were not consulted."
        ),
    })

    print(f"\nArtifacts -> {artifacts}")
    print(f"Metrics   -> {cfg.path('metrics_dir') / f'model_selection_{protocol}.json'}")
    print("Thresholds are now FROZEN. scripts/evaluate.py only reads them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
