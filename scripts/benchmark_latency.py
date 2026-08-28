"""Benchmarks batch inference latency for 10,000 rows. Target: under 2 seconds.

    python scripts/benchmark_latency.py

Measures feature generation, preprocessing, and scoring together; excludes reading the CSV
and loading the model, since that's a one-off cost. Uses time.perf_counter with a few
warm-up runs discarded, then reports the median of several measured repetitions.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import _bootstrap  # noqa: F401

import pandas as pd  # noqa: E402

from sentrynet.config import load_config  # noqa: E402
from sentrynet.inference import KIND_BASELINE, KIND_IFOREST, KIND_OCSVM, build_pipeline  # noqa: E402
from sentrynet.persistence import environment_record, load_scoring_bundle, run_record, save_json  # noqa: E402
from sentrynet.splits import load_splits, subset  # noqa: E402

DETECTORS = (
    (KIND_BASELINE, "baseline.joblib", "Statistical Baseline"),
    (KIND_IFOREST, "isolation_forest.joblib", "Isolation Forest"),
    (KIND_OCSVM, "one_class_svm.joblib", "One-Class SVM"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=None)
    args = parser.parse_args()

    cfg = load_config()
    cfg.ensure_dirs()
    protocol = args.protocol or cfg["splits"]["active_protocol"]
    artifacts = cfg.path("artifacts_dir") / protocol
    n_rows = int(cfg["latency"]["n_rows"])
    warmups = int(cfg["latency"]["warmup_repeats"])
    repeats = int(cfg["latency"]["measured_repeats"])
    target = 2.0

    if not (artifacts / "isolation_forest.joblib").exists():
        print(f"[FAIL] No artifacts in {artifacts}. Run scripts/train.py.", file=sys.stderr)
        return 1

    clean = pd.read_csv(cfg.path("clean_csv"))
    splits = load_splits(cfg.path("splits_dir") / protocol)
    test = subset(clean, splits["test"])
    if len(test) < n_rows:
        print(f"[FAIL] Test partition has only {len(test)} rows, need {n_rows}.", file=sys.stderr)
        return 1
    # Exactly n_rows, taken deterministically from the held-out test partition.
    batch = test.iloc[:n_rows].reset_index(drop=True).drop(
        columns=[cfg["dataset"]["label_column"]], errors="ignore"
    )
    print(f"Benchmarking {len(batch):,} rows (protocol: {protocol})")

    results = {}
    for kind, filename, pretty in DETECTORS:
        load_start = time.perf_counter()
        bundle = load_scoring_bundle(artifacts / filename)
        pipe = build_pipeline(
            kind=bundle["kind"], model=bundle["model"], threshold=bundle["threshold"],
            preprocessor=bundle["preprocessor"], metadata={"params": bundle["params"]},
        )
        load_seconds = time.perf_counter() - load_start

        for _ in range(warmups):
            pipe.score(batch)

        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            pipe.score(batch)
            timings.append(time.perf_counter() - start)

        median = statistics.median(timings)
        results[kind] = {
            "detector": pretty,
            "n_rows": int(len(batch)),
            "median_seconds": median,
            "mean_seconds": statistics.fmean(timings),
            "min_seconds": min(timings),
            "max_seconds": max(timings),
            "stdev_seconds": statistics.stdev(timings) if len(timings) > 1 else 0.0,
            "all_timings_seconds": timings,
            "warmup_repeats": warmups,
            "measured_repeats": repeats,
            "rows_per_second_median": len(batch) / median if median else None,
            "model_load_seconds_excluded_from_timing": load_seconds,
            "target_seconds_per_10k_rows": target,
            "target_status": "MET" if median < target else "UNMET",
        }
        print(
            f"  {pretty:<22} median {median:.4f}s  "
            f"(min {min(timings):.4f} / max {max(timings):.4f})  "
            f"-> target < {target}s: {results[kind]['target_status']}"
        )

    payload = {
        "run": run_record(cfg.data, None, protocol),
        "measurement_scope": {
            "includes": [
                "derived feature generation",
                "preprocessing transform (log1p -> StandardScaler -> OneHotEncoder)",
                "model decision score",
            ],
            "excludes": ["reading the CSV from disk", "loading model artifacts (reported separately)"],
            "timer": "time.perf_counter",
            "statistic_reported": "median of measured repetitions after warm-up",
        },
        "environment": environment_record(),
        "detectors": results,
    }
    out = cfg.path("metrics_dir") / f"latency_{protocol}.json"
    save_json(out, payload)
    print(f"\nEnvironment: Python {payload['environment']['python_version']}, "
          f"scikit-learn {payload['environment']['scikit_learn_version']}")
    print(f"Platform: {payload['environment']['platform']}")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
