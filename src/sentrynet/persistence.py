"""Saves and loads trained artifacts (preprocessor, models, thresholds, PSI reference).

save_scoring_bundle refuses to write a bundle that contains a label column, so a leak
would fail loudly instead of shipping.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import joblib
import sklearn

from sentrynet.features import LABEL_COLUMNS

FORBIDDEN_BUNDLE_KEYS = set(LABEL_COLUMNS) | {"y", "y_true", "labels", "attack_labels"}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def environment_record() -> dict[str, Any]:
    """Versions and platform, recorded with every experiment for reproducibility."""
    return {
        "python_version": sys.version.split()[0],
        "scikit_learn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or "unavailable",
        "machine": platform.machine(),
    }


def run_record(
    config: Mapping[str, Any],
    dataset_fingerprint: str | None = None,
    split_protocol: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Provenance block embedded in every metrics file."""
    record = {
        "run_timestamp_utc": utc_timestamp(),
        "seed": int(config["project"]["seed"]),
        "split_protocol": split_protocol,
        "dataset_fingerprint_sha256": dataset_fingerprint,
        "environment": environment_record(),
        "config": dict(config),
    }
    if extra:
        record.update(dict(extra))
    return record


def _assert_no_labels_in_bundle(bundle: Mapping[str, Any]) -> None:
    leaked = sorted(set(bundle) & FORBIDDEN_BUNDLE_KEYS)
    if leaked:
        raise AssertionError(
            f"Refusing to persist an inference bundle containing label keys: {leaked}."
        )
    for key, value in bundle.items():
        if hasattr(value, "columns"):
            present = sorted(set(value.columns) & FORBIDDEN_BUNDLE_KEYS)
            if present:
                raise AssertionError(
                    f"Refusing to persist: bundle entry {key!r} carries label columns {present}."
                )


def save_scoring_bundle(path: str | Path, bundle: Mapping[str, Any]) -> Path:
    """Persist an inference bundle with ``joblib`` after asserting it carries no labels."""
    _assert_no_labels_in_bundle(bundle)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(bundle), out)
    return out


def load_scoring_bundle(path: str | Path) -> dict[str, Any]:
    """Load a persisted inference bundle."""
    out = Path(path)
    if not out.exists():
        raise FileNotFoundError(f"Artifact {out} not found. Run `python scripts/train.py` first.")
    return joblib.load(out)


def save_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, default=str)
        handle.write("\n")
    return out


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
