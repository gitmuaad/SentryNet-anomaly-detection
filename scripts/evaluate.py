"""Phase 4 + 5 — final test evaluation, sensitivity slices, FP windows, PSI, evasion.

    python scripts/evaluate.py
    python scripts/evaluate.py --protocol row_order

This script **never** fits a model and **never** re-selects a threshold. It loads the frozen
artifacts written by ``train.py`` and uses the final test partition exactly once per
detector. PortScan appears here and only here, reported as *Unseen Attack Recall*.
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sentrynet.config import load_config  # noqa: E402
from sentrynet.data import file_fingerprint  # noqa: E402
from sentrynet.evaluation import (  # noqa: E402
    evaluate_detector,
    false_positive_windows,
    prevalence_report,
    target_status,
)
from sentrynet.evasion import evasion_report  # noqa: E402
from sentrynet.inference import KIND_BASELINE, KIND_IFOREST, KIND_OCSVM, build_pipeline  # noqa: E402
from sentrynet.monitoring import psi_report_from_frame  # noqa: E402
from sentrynet.persistence import load_scoring_bundle, run_record, save_json  # noqa: E402
from sentrynet.sensitivity import (  # noqa: E402
    build_prevalence_slice,
    evaluate_slice,
    implied_daily_alerts,
)
from sentrynet.splits import load_splits, subset  # noqa: E402

DETECTORS = (
    (KIND_BASELINE, "baseline.joblib", "Statistical Baseline"),
    (KIND_IFOREST, "isolation_forest.joblib", "Isolation Forest"),
    (KIND_OCSVM, "one_class_svm.joblib", "One-Class SVM"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=None, help="random | row_order")
    args = parser.parse_args()

    cfg = load_config()
    cfg.ensure_dirs()
    protocol = args.protocol or cfg["splits"]["active_protocol"]
    label = cfg["dataset"]["label_column"]
    normal_label = cfg["dataset"]["normal_label"]
    unseen = tuple(cfg["splits"]["unseen_attacks"])
    seed = cfg.seed

    artifacts = cfg.path("artifacts_dir") / protocol
    if not (artifacts / "isolation_forest.joblib").exists():
        print(f"[FAIL] No artifacts in {artifacts}. Run scripts/train.py.", file=sys.stderr)
        return 1

    clean = pd.read_csv(cfg.path("clean_csv"))
    splits = load_splits(cfg.path("splits_dir") / protocol)
    train = subset(clean, splits["train"])
    test = subset(clean, splits["test"])
    thresholds_frozen: dict[str, float] = {}

    pipelines = {}
    for kind, filename, _ in DETECTORS:
        bundle = load_scoring_bundle(artifacts / filename)
        pipelines[kind] = build_pipeline(
            kind=bundle["kind"],
            model=bundle["model"],
            threshold=bundle["threshold"],
            preprocessor=bundle["preprocessor"],
            metadata={"params": bundle["params"], "seed": bundle["seed"]},
        )
        thresholds_frozen[kind] = float(bundle["threshold"])

    print(f"Protocol: {protocol}")
    print(f"  test partition: {len(test):,} rows  {dict(test[label].value_counts())}")
    print(f"  frozen thresholds (loaded, not recomputed): {thresholds_frozen}")

    prevalence = prevalence_report(
        test[label], cfg["dataset"]["expected_class_counts"], normal_label
    )
    print(
        f"  actual test prevalence: {prevalence['actual_final_test']['normal_fraction']:.4f} Normal "
        f"/ {prevalence['actual_final_test']['attack_fraction']:.4f} Attack"
    )

    fingerprint = file_fingerprint(cfg.path("raw_csv"))
    final: dict = {}
    window_frames: dict = {}
    sensitivity_out: dict = {}
    comparison_rows = []

    for kind, _, pretty in DETECTORS:
        pipe = pipelines[kind]
        scores = pipe.score(test)
        record = evaluate_detector(
            pretty, scores, test[label], pipe.threshold, normal_label=normal_label, unseen=unseen
        )
        record["detector_key"] = kind
        record["params"] = pipe.metadata.get("params")
        record["targets"] = target_status(
            record["pr_auc"], record.get("ddos_recall"), record.get("bruteforce_recall"), None,
            pr_auc_target=float(cfg["evaluation"]["pr_auc_target"]),
            recall_target=float(cfg["threshold"]["ddos_recall_target"]),
        )
        final[kind] = record

        windows = false_positive_windows(
            test[label], scores, pipe.threshold,
            window_size=int(cfg["evaluation"]["window_size"]), normal_label=normal_label,
        )
        windows.to_csv(cfg.path("tables_dir") / f"fp_windows_{protocol}_{kind}.csv", index=False)
        window_frames[kind] = windows
        record["false_positive_windows_summary"] = {
            "window_size_rows": int(cfg["evaluation"]["window_size"]),
            "window_definition": "block of 1,000 consecutive rows - NOT a time window",
            "n_windows": int(len(windows)),
            "total_false_positives": int(windows["false_positives"].sum()),
            "mean_false_positives_per_window": float(windows["false_positives"].mean()),
            "max_false_positives_in_a_window": int(windows["false_positives"].max()),
            "mean_fp_rate_among_normal": float(windows["false_positive_rate_among_normal"].mean()),
        }

        print(
            f"\n  {pretty}"
            f"\n    PR-AUC {record['pr_auc']:.6f} (no-skill {record['pr_auc_no_skill']:.4f})"
            f"  P {record['precision']:.4f}  R {record['recall']:.4f}  F1 {record['f1']:.4f}"
            f"\n    DDoS recall {record['ddos_recall']:.4f}   BruteForce recall {record['bruteforce_recall']:.4f}"
            f"   Unseen (PortScan) recall {record.get('unseen_attack_recall_portscan', float('nan')):.4f}"
            f"\n    false positives {record['false_positives']:,} / {record['n_normal_rows']:,} Normal"
            f"  (FPR {record['false_positive_rate']:.4f})"
        )

        comparison_rows.append(
            {
                "detector": pretty,
                "pr_auc": round(record["pr_auc"], 6),
                "pr_auc_no_skill": round(record["pr_auc_no_skill"], 6),
                "precision": round(record["precision"], 6),
                "recall": round(record["recall"], 6),
                "f1": round(record["f1"], 6),
                "ddos_recall": round(record["ddos_recall"], 6),
                "bruteforce_recall": round(record["bruteforce_recall"], 6),
                "unseen_attack_recall_portscan": round(
                    record.get("unseen_attack_recall_portscan", float("nan")), 6
                ),
                "false_positives": record["false_positives"],
                "false_positive_rate": round(record["false_positive_rate"], 6),
                "threshold": record["threshold"],
            }
        )

        # --- prevalence sensitivity slices (held-out rows only) ----------------------
        for slice_cfg in cfg["sensitivity"]["slices"]:
            name = slice_cfg["name"]
            slice_df, slice_record = build_prevalence_slice(
                test, float(slice_cfg["normal_fraction"]), seed, label, normal_label
            )
            slice_scores = pipe.score(slice_df)
            metrics = evaluate_slice(slice_scores, slice_df[label], pipe.threshold, normal_label)
            sensitivity_out.setdefault(name, {})[kind] = {
                "slice": slice_record,
                "metrics": metrics,
                "implied_daily_scenario": implied_daily_alerts(
                    metrics, int(cfg["sensitivity"]["assumed_daily_flow_volume"])
                ),
            }

    # --- Sensitivity console summary ---------------------------------------------------
    print("\nPrevalence sensitivity (held-out rows only, reproducible resampling):")
    for name, per_detector in sensitivity_out.items():
        first = next(iter(per_detector.values()))["slice"]
        print(
            f"  {name}: {first['n_rows']:,} rows "
            f"({first['achieved_normal_fraction']:.3f} Normal), training rows used: {first['training_rows_used']}"
        )
        for kind, payload in per_detector.items():
            m = payload["metrics"]
            print(
                f"    {kind:<22} alert rate {m['alert_rate']:.4f}  P {m['precision']:.4f}  "
                f"R {m['recall']:.4f}  FP {m['false_positives']:,}  "
                f"FP/1000 rows {m['false_positives_per_1000_rows']:.3f}  "
                f"alerts/10000 rows {m['expected_alerts_per_10000_rows']:.1f}"
            )

    # --- PSI drift examples ------------------------------------------------------------
    psi_reference = load_scoring_bundle(artifacts / "psi_reference.joblib")["psi_reference"]
    psi_examples = {
        "held_out_normal_rows": psi_report_from_frame(
            test.loc[test[label] == normal_label], psi_reference,
            float(cfg["monitoring"]["psi_review_threshold"]),
        ),
        "held_out_portscan_rows": psi_report_from_frame(
            test.loc[test[label] == "PortScan"], psi_reference,
            float(cfg["monitoring"]["psi_review_threshold"]),
        ),
        "full_held_out_test": psi_report_from_frame(
            test, psi_reference, float(cfg["monitoring"]["psi_review_threshold"])
        ),
    }
    print("\nPSI drift check (reference = Normal training distribution):")
    for name, report in psi_examples.items():
        print(
            f"  {name:<24} max PSI {report['max_psi']:.4f}  "
            f"review triggered: {report['retraining_review_recommended']} "
            f"{report['features_triggering_review']}"
        )

    # --- Synthetic evasion stress testing ----------------------------------------------
    print("\nSynthetic evasion stress testing (held-out attack rows, disk untouched):")
    test_attacks = test.loc[test[label] != normal_label]
    evasion_out = {}
    for kind, _, pretty in DETECTORS:
        pipe = pipelines[kind]
        report = evasion_report(
            score_fn=pipe.score,
            attack_rows=test_attacks,
            normal_train=train,
            threshold=pipe.threshold,
            features=cfg["evasion"]["target_features"],
            strengths=cfg["evasion"]["strengths"],
            seed=seed,
            percentiles=cfg["evasion"]["normal_range_percentiles"],
            max_rows_per_class=int(cfg["evasion"]["max_rows_per_class"]),
            label_column=label,
        )
        evasion_out[kind] = report
        full = [s for s in report["by_strength"] if s["evasion_strength"] == 1.0]
        if full:
            s = full[0]
            print(
                f"  {kind:<22} original recall {s['original_recall']:.4f} -> "
                f"full-strength evasion recall {s['evasion_recall']:.4f} "
                f"(drop {s['recall_drop']:.4f})"
            )

    # --- Figures and tables ------------------------------------------------------------
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(cfg.path("tables_dir") / f"final_comparison_{protocol}.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    for kind, _, pretty in DETECTORS:
        w = window_frames[kind]
        ax.plot(w["window_index"], w["false_positives"], marker="o", ms=3, label=pretty)
    ax.set_xlabel("window index (each window = 1,000 consecutive rows, NOT a time window)")
    ax.set_ylabel("false positives")
    ax.set_title(f"False positives per 1,000-row window — final test set ({protocol} protocol)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.path("figures_dir") / f"fp_windows_{protocol}.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for ax, (kind, _, pretty) in zip(axes, DETECTORS):
        pipe = pipelines[kind]
        scores = pipe.score(test)
        for cls, color in [
            ("Normal", "#2a9d8f"), ("DDoS", "#e76f51"),
            ("BruteForce", "#e9c46a"), ("PortScan", "#264653"),
        ]:
            mask = (test[label] == cls).to_numpy()
            if mask.any():
                ax.hist(scores[mask], bins=60, alpha=0.55, label=cls, color=color, log=True)
        ax.axvline(pipe.threshold, color="black", ls="--", lw=1.2, label="threshold")
        ax.set_title(pretty)
        ax.set_xlabel("anomaly score (higher = more anomalous)")
    axes[0].set_ylabel("count (log scale)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Anomaly score distributions on the final test set ({protocol} protocol)", y=1.02)
    fig.tight_layout()
    fig.savefig(cfg.path("figures_dir") / f"score_distributions_{protocol}.png", dpi=150)
    plt.close(fig)

    evasion_rows = [
        {
            "detector": kind,
            "evasion_strength": r["evasion_strength"],
            "original_recall": r["original_recall"],
            "evasion_recall": r["evasion_recall"],
            "recall_drop": r["recall_drop"],
            **{f"{c}_evasion_recall": v["evasion_recall"] for c, v in r["per_class"].items()},
        }
        for kind, report in evasion_out.items()
        for r in report["by_strength"]
    ]
    pd.DataFrame(evasion_rows).round(6).to_csv(
        cfg.path("tables_dir") / f"evasion_recall_{protocol}.csv", index=False
    )

    # --- Persist metrics ----------------------------------------------------------------
    run = run_record(cfg.data, fingerprint, protocol)
    save_json(cfg.path("metrics_dir") / f"final_evaluation_{protocol}.json", {
        "run": run,
        "prevalence": prevalence,
        "frozen_thresholds": thresholds_frozen,
        "threshold_note": (
            "Thresholds were selected on validation only and loaded read-only here. The test "
            "partition never influenced any hyperparameter, feature, or threshold choice."
        ),
        "detectors": final,
    })
    save_json(cfg.path("metrics_dir") / f"sensitivity_{protocol}.json", sensitivity_out)
    save_json(cfg.path("metrics_dir") / f"psi_{protocol}.json", {
        "run": run, "reference": {k: v for k, v in psi_reference.items() if k != "features"},
        "examples": psi_examples,
    })
    save_json(cfg.path("metrics_dir") / f"evasion_{protocol}.json", {"run": run, "detectors": evasion_out})

    print(f"\nMetrics -> {cfg.path('metrics_dir')}")
    print(f"Tables  -> {cfg.path('tables_dir')}")
    print(f"Figures -> {cfg.path('figures_dir')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
