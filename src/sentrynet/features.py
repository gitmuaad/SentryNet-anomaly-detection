"""Deterministic feature engineering.

Owner in the Action Plan: **Muaad Alkathiri** (derived-ratio features).

Four derived features are built from the six source fields. Nothing here looks at
``attack_type``: the label is dropped before a feature matrix is ever produced, so label
leakage is structurally impossible rather than merely avoided by convention.

Safe-denominator rule (decision D-03)
-------------------------------------
Every ratio uses::

    ratio = numerator / maximum(denominator, SAFE_DENOMINATOR_FLOOR)

with ``SAFE_DENOMINATOR_FLOOR = 1.0``. All three denominators (``dst_bytes``,
``packet_count``, ``duration``) are integer-valued byte/packet counts or whole seconds, so
1.0 is the smallest meaningful non-zero denominator. This guarantees a finite result for
every input, including zero, without inventing an arbitrarily tiny epsilon that would
explode the ratio to ~1e12. The result is additionally passed through a finiteness guard so
that no ``inf`` or ``NaN`` can ever leave this module.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

SOURCE_NUMERIC: tuple[str, ...] = (
    "duration",
    "src_bytes",
    "dst_bytes",
    "packet_count",
    "failed_logins",
)
SOURCE_CATEGORICAL: tuple[str, ...] = ("protocol",)
DERIVED_NUMERIC: tuple[str, ...] = (
    "total_bytes",
    "src_dst_ratio",
    "bytes_per_packet",
    "failed_logins_per_second",
)

#: Columns that must never reach a model as an input feature.
LABEL_COLUMNS: tuple[str, ...] = ("attack_type", "is_attack", "binary_label")

SAFE_DENOMINATOR_FLOOR = 1.0


def safe_divide(
    numerator: pd.Series | np.ndarray,
    denominator: pd.Series | np.ndarray,
    floor: float = SAFE_DENOMINATOR_FLOOR,
) -> np.ndarray:
    """Divide with a deterministic denominator floor.

    ``numerator / maximum(denominator, floor)``, followed by a finiteness guard that maps
    any residual ``inf``/``NaN`` to ``0.0``. The guard should never fire for this dataset; it
    exists so that a malformed uploaded CSV cannot poison the Gradio scoring path.
    """
    num = np.asarray(numerator, dtype="float64")
    den = np.asarray(denominator, dtype="float64")
    den = np.where(np.isfinite(den), den, floor)
    den = np.maximum(den, floor)
    out = num / den
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def add_derived_features(frame: pd.DataFrame, floor: float = SAFE_DENOMINATOR_FLOOR) -> pd.DataFrame:
    """Return a copy of ``frame`` with the four required derived features appended.

    ==============================  ==================================================
    Feature                         Formula
    ==============================  ==================================================
    ``total_bytes``                 ``src_bytes + dst_bytes``
    ``src_dst_ratio``               ``src_bytes / max(dst_bytes, 1.0)``
    ``bytes_per_packet``            ``total_bytes / max(packet_count, 1.0)``
    ``failed_logins_per_second``    ``failed_logins / max(duration, 1.0)``
    ==============================  ==================================================
    """
    missing = [c for c in SOURCE_NUMERIC if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required source columns: {missing}")

    out = frame.copy()
    src = out["src_bytes"].astype("float64")
    dst = out["dst_bytes"].astype("float64")
    packets = out["packet_count"].astype("float64")
    duration = out["duration"].astype("float64")
    failed = out["failed_logins"].astype("float64")

    total = src + dst
    out["total_bytes"] = np.nan_to_num(total.to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    out["src_dst_ratio"] = safe_divide(src, dst, floor)
    out["bytes_per_packet"] = safe_divide(out["total_bytes"], packets, floor)
    out["failed_logins_per_second"] = safe_divide(failed, duration, floor)
    return out


def feature_columns(
    numeric: Sequence[str] = SOURCE_NUMERIC,
    derived: Sequence[str] = DERIVED_NUMERIC,
    categorical: Sequence[str] = SOURCE_CATEGORICAL,
) -> tuple[list[str], list[str]]:
    """Return ``(numeric_columns, categorical_columns)`` in a fixed, documented order."""
    return list(numeric) + list(derived), list(categorical)


def build_feature_frame(
    frame: pd.DataFrame,
    numeric: Sequence[str] = SOURCE_NUMERIC,
    derived: Sequence[str] = DERIVED_NUMERIC,
    categorical: Sequence[str] = SOURCE_CATEGORICAL,
    floor: float = SAFE_DENOMINATOR_FLOOR,
) -> pd.DataFrame:
    """Build the model-ready feature frame ``X``.

    The label column and every bookkeeping column are excluded by construction: only the
    explicitly listed feature columns are selected, so ``attack_type`` cannot leak in even if
    it is present in the input.
    """
    enriched = add_derived_features(frame, floor=floor)
    num_cols, cat_cols = feature_columns(numeric, derived, categorical)
    selected = num_cols + cat_cols
    absent = [c for c in selected if c not in enriched.columns]
    if absent:
        raise ValueError(f"Missing required feature columns: {absent}")
    features = enriched.loc[:, selected].copy()
    assert_no_labels(features)
    assert_finite(features[num_cols])
    return features


def assert_no_labels(frame: pd.DataFrame, label_columns: Iterable[str] = LABEL_COLUMNS) -> None:
    """Raise if any label-bearing column is present in a feature matrix."""
    present = [c for c in label_columns if c in frame.columns]
    if present:
        raise AssertionError(
            f"Label leakage: {present} must never appear in the feature matrix."
        )


def assert_finite(frame: pd.DataFrame) -> None:
    """Raise if any numeric feature contains ``NaN`` or ``inf``."""
    values = frame.to_numpy(dtype="float64", copy=False)
    if not np.isfinite(values).all():
        bad = [c for c in frame.columns if not np.isfinite(frame[c].to_numpy(dtype="float64")).all()]
        raise AssertionError(f"Non-finite values produced in features: {bad}")
