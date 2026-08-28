"""Synthetic evasion stress testing.

Owner in the Action Plan: **Feras Mithwah**.

What this is
------------
Held-out **attack-labelled** rows are copied and their numeric fields are pulled toward the
empirical Normal *training* range. Recall on the originals is compared with recall on the
variants, at the frozen operating threshold, to measure how much detection is lost when an
attacker makes traffic look ordinary.

What this is **not**
--------------------
This is **not** a proof of adversarial robustness. It is a reproducible, deliberately simple
stress test against one specific evasion idea: "stay inside the Normal envelope". A real
adversary is not constrained to this transformation. It is described throughout the report
only as *synthetic evasion stress testing*.

Safety
------
The original dataset on disk is never modified. Every generated row carries
``is_synthetic_evasion_variant = True`` and is written only under ``outputs/``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

VARIANT_FLAG = "is_synthetic_evasion_variant"


def normal_envelope(
    normal_train: pd.DataFrame,
    features: Sequence[str],
    percentiles: Sequence[float] = (1.0, 99.0),
) -> dict[str, dict[str, float]]:
    """Empirical ``[lo, hi]`` range per feature, estimated from Normal training rows only."""
    lo_p, hi_p = float(percentiles[0]), float(percentiles[1])
    return {
        col: {
            "lo": float(np.percentile(normal_train[col].to_numpy(dtype="float64"), lo_p)),
            "hi": float(np.percentile(normal_train[col].to_numpy(dtype="float64"), hi_p)),
        }
        for col in features
    }


def synthesise_evasion_variants(
    attack_rows: pd.DataFrame,
    envelope: Mapping[str, Mapping[str, float]],
    strength: float,
    features: Sequence[str],
    seed: int,
    max_rows_per_class: int | None = None,
    label_column: str = "attack_type",
) -> pd.DataFrame:
    """Blend attack rows toward the Normal envelope.

    For each targeted feature::

        clipped = clip(x, lo, hi)
        x_evaded = (1 - strength) * x + strength * clipped

    ``strength = 0`` leaves the row unchanged; ``strength = 1`` forces it fully inside the
    Normal training range. Integer-valued source columns are rounded back to integers so the
    variants stay physically plausible (you cannot send half a packet).
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")

    rng = np.random.default_rng(seed)
    frames = []
    for cls, group in attack_rows.groupby(label_column, sort=True):
        if max_rows_per_class is not None and len(group) > max_rows_per_class:
            idx = np.sort(rng.choice(len(group), size=int(max_rows_per_class), replace=False))
            group = group.iloc[idx]
        frames.append(group)
    # Concatenating per-class groups would reorder rows; sort back to the input order so
    # the output is a row-for-row counterpart of the source frame.
    sampled = pd.concat(frames).sort_index() if frames else attack_rows.copy()
    sampled = sampled.reset_index(drop=True)

    out = sampled.copy()
    for col in features:
        if col not in out.columns:
            continue
        original = sampled[col].to_numpy(dtype="float64")
        lo, hi = envelope[col]["lo"], envelope[col]["hi"]
        clipped = np.clip(original, lo, hi)
        blended = (1.0 - strength) * original + strength * clipped
        if pd.api.types.is_integer_dtype(sampled[col].dtype):
            blended = np.rint(blended)
        out[col] = blended
    out[VARIANT_FLAG] = True
    out["evasion_strength"] = float(strength)
    return out


def evasion_report(
    score_fn,
    attack_rows: pd.DataFrame,
    normal_train: pd.DataFrame,
    threshold: float,
    features: Sequence[str],
    strengths: Sequence[float],
    seed: int,
    percentiles: Sequence[float] = (1.0, 99.0),
    max_rows_per_class: int | None = None,
    label_column: str = "attack_type",
) -> dict[str, Any]:
    """Recall on originals vs. recall on evasion variants, overall and per attack class.

    ``score_fn`` maps a raw flow frame to anomaly scores (higher = more anomalous).
    """
    envelope = normal_envelope(normal_train, features, percentiles)

    baseline_rows = synthesise_evasion_variants(
        attack_rows, envelope, 0.0, features, seed, max_rows_per_class, label_column
    )
    original_scores = np.asarray(score_fn(baseline_rows), dtype="float64")
    original_flagged = original_scores >= threshold
    original_recall = float(original_flagged.mean())
    original_by_class = {
        str(cls): float(original_flagged[(baseline_rows[label_column] == cls).to_numpy()].mean())
        for cls in sorted(baseline_rows[label_column].unique())
    }

    per_strength = []
    for strength in strengths:
        variants = synthesise_evasion_variants(
            attack_rows, envelope, float(strength), features, seed, max_rows_per_class, label_column
        )
        flagged = np.asarray(score_fn(variants), dtype="float64") >= threshold
        recall = float(flagged.mean())
        by_class = {
            str(cls): float(flagged[(variants[label_column] == cls).to_numpy()].mean())
            for cls in sorted(variants[label_column].unique())
        }
        per_strength.append(
            {
                "evasion_strength": float(strength),
                "n_rows": int(len(variants)),
                "original_recall": original_recall,
                "evasion_recall": recall,
                "recall_drop": original_recall - recall,
                "per_class": {
                    cls: {
                        "original_recall": original_by_class.get(cls),
                        "evasion_recall": by_class.get(cls),
                        "recall_drop": (original_by_class.get(cls, 0.0) - by_class.get(cls, 0.0)),
                    }
                    for cls in sorted(by_class)
                },
            }
        )

    return {
        "test_name": "Synthetic evasion stress testing",
        "not_a_claim": (
            "This does NOT prove adversarial robustness. It measures recall loss against one "
            "specific, simple evasion strategy: moving numeric fields into the Normal "
            "training range."
        ),
        "threshold": float(threshold),
        "seed": int(seed),
        "target_features": list(features),
        "normal_envelope_percentiles": list(percentiles),
        "normal_envelope": envelope,
        "source_rows": "held-out FINAL TEST attack rows only; the dataset on disk is unmodified",
        "original_recall": original_recall,
        "original_recall_by_class": original_by_class,
        "by_strength": per_strength,
    }
