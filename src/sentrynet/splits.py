"""Frozen split protocols.

Owner in the Action Plan: Bader Alkhalifa (validation-strategy design), published by
Mousa Alharthi.

**The dataset has no timestamp, no date, and no session identifier.** There is therefore no
chronological ordering and no calendar split is possible. Two reproducible protocols are
implemented instead:

``row_order``
    *Sequential row-order stability split.* Partitions are cut from the source-file row
    order. This is a **stability check only**. It is explicitly **not** a temporal split and
    must never be described as one, because source-file row order carries no time semantics.

``random``
    Fixed random split with a documented ``random_state``.

Partition rules, identical under both protocols:

* ``train`` — Normal rows only. Fits the models and the preprocessor.
* ``validation`` — held-out Normal + DDoS + BruteForce. Hyperparameter and operating
  threshold selection. **PortScan is excluded.**
* ``test`` — held-out Normal + held-out DDoS/BruteForce + **all** PortScan rows. Used once.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from sentrynet.data import ROW_ID

PROTOCOLS = ("random", "row_order")

PROTOCOL_DESCRIPTIONS = {
    "random": "Fixed random split with a documented random_state (reproducible).",
    "row_order": (
        "Sequential row-order stability split: partitions cut from source-file row order. "
        "This is a stability check, NOT a temporal split - the dataset has no timestamp."
    ),
}


def _ordered_indices(values: pd.Index, protocol: str, rng: np.random.Generator) -> np.ndarray:
    """Return the row ids of one class stratum in the order the protocol prescribes."""
    ids = np.asarray(values, dtype="int64")
    if protocol == "row_order":
        return np.sort(ids)
    if protocol == "random":
        permuted = ids.copy()
        rng.shuffle(permuted)
        return permuted
    raise ValueError(f"Unknown split protocol: {protocol!r}. Expected one of {PROTOCOLS}.")


def _cut(ids: np.ndarray, fractions: Sequence[float]) -> list[np.ndarray]:
    """Split ``ids`` into contiguous chunks by ``fractions`` (which must sum to ~1)."""
    total = len(ids)
    bounds = [0]
    running = 0.0
    for frac in fractions[:-1]:
        running += frac
        bounds.append(int(round(running * total)))
    bounds.append(total)
    return [ids[bounds[i] : bounds[i + 1]] for i in range(len(fractions))]


def build_splits(
    frame: pd.DataFrame,
    split_cfg: Mapping[str, Any],
    seed: int,
    protocol: str,
    label_column: str = "attack_type",
    normal_label: str = "Normal",
) -> dict[str, np.ndarray]:
    """Return ``{"train": ids, "validation": ids, "test": ids}`` as ``source_row_id`` arrays."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown split protocol: {protocol!r}. Expected one of {PROTOCOLS}.")
    if ROW_ID not in frame.columns:
        raise ValueError(f"Frame must carry the {ROW_ID!r} traceability column.")

    rng = np.random.default_rng(seed)
    normal_cfg = split_cfg["normal"]
    tuning_attacks = list(split_cfg["tuning_attacks"])
    unseen_attacks = list(split_cfg["unseen_attacks"])
    attack_cfg = split_cfg["tuning_attack_split"]

    train_ids: list[np.ndarray] = []
    val_ids: list[np.ndarray] = []
    test_ids: list[np.ndarray] = []

    # --- Normal: three-way. Only the train chunk may fit models or preprocessing. --------
    normal_rows = frame.loc[frame[label_column] == normal_label, ROW_ID]
    ordered = _ordered_indices(normal_rows, protocol, rng)
    n_train, n_val, n_test = _cut(
        ordered,
        [float(normal_cfg["train"]), float(normal_cfg["validation"]), float(normal_cfg["test"])],
    )
    train_ids.append(n_train)
    val_ids.append(n_val)
    test_ids.append(n_test)

    # --- Tuning attacks (DDoS, BruteForce): validation + test only. Never train. ---------
    for attack in tuning_attacks:
        rows = frame.loc[frame[label_column] == attack, ROW_ID]
        ordered = _ordered_indices(rows, protocol, rng)
        a_val, a_test = _cut(
            ordered, [float(attack_cfg["validation"]), float(attack_cfg["test"])]
        )
        val_ids.append(a_val)
        test_ids.append(a_test)

    # --- Unseen attacks (PortScan): final test only. Never train, never validation. ------
    for attack in unseen_attacks:
        rows = frame.loc[frame[label_column] == attack, ROW_ID]
        test_ids.append(_ordered_indices(rows, protocol, rng))

    splits = {
        "train": np.sort(np.concatenate(train_ids)),
        "validation": np.sort(np.concatenate(val_ids)),
        "test": np.sort(np.concatenate(test_ids)),
    }
    _assert_split_rules(frame, splits, split_cfg, label_column, normal_label)
    return splits


def _assert_split_rules(
    frame: pd.DataFrame,
    splits: Mapping[str, np.ndarray],
    split_cfg: Mapping[str, Any],
    label_column: str,
    normal_label: str,
) -> None:
    """Hard runtime guarantees. These mirror the tests in ``tests/test_splits.py``."""
    labels = frame.set_index(ROW_ID)[label_column]

    train_labels = set(labels.loc[splits["train"]].unique())
    if train_labels != {normal_label}:
        raise AssertionError(
            f"Training partition must contain only {normal_label!r} rows, found {sorted(train_labels)}."
        )

    unseen = set(split_cfg["unseen_attacks"])
    val_labels = set(labels.loc[splits["validation"]].unique())
    leaked = val_labels & unseen
    if leaked:
        raise AssertionError(
            f"Unseen attack classes {sorted(leaked)} leaked into validation/tuning. "
            "The unseen-attack protocol forbids this."
        )

    _assert_disjoint(splits)


def _assert_disjoint(splits: Mapping[str, np.ndarray]) -> None:
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = np.intersect1d(splits[a], splits[b])
            if overlap.size:
                raise AssertionError(
                    f"Partitions {a!r} and {b!r} overlap on {overlap.size} rows (e.g. {overlap[:5]})."
                )


def split_metadata(
    frame: pd.DataFrame,
    splits: Mapping[str, np.ndarray],
    split_cfg: Mapping[str, Any],
    seed: int,
    protocol: str,
    label_column: str = "attack_type",
    normal_label: str = "Normal",
    dataset_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Build the metadata record persisted next to the frozen row-id files."""
    labels = frame.set_index(ROW_ID)[label_column]
    unseen = list(split_cfg["unseen_attacks"])

    per_split: dict[str, Any] = {}
    for name, ids in splits.items():
        counts = labels.loc[ids].value_counts().to_dict()
        per_split[name] = {
            "n_rows": int(len(ids)),
            "class_counts": {str(k): int(v) for k, v in sorted(counts.items())},
            "n_normal": int(counts.get(normal_label, 0)),
            "n_attack": int(len(ids) - counts.get(normal_label, 0)),
        }

    val_counts = per_split["validation"]["class_counts"]
    return {
        "protocol": protocol,
        "protocol_description": PROTOCOL_DESCRIPTIONS[protocol],
        "is_temporal_split": False,
        "temporal_split_possible": False,
        "why_not_temporal": (
            "The dataset contains no timestamp, date, or session identifier, so it has no "
            "chronological ordering at all. Source-file row order carries no time meaning and "
            "is used only as a stability check."
        ),
        "random_seed": int(seed),
        "fractions": {
            "normal": dict(split_cfg["normal"]),
            "tuning_attack_split": dict(split_cfg["tuning_attack_split"]),
            "unseen_attack_placement": "test only (100%)",
        },
        "tuning_attacks": list(split_cfg["tuning_attacks"]),
        "unseen_attacks": unseen,
        "splits": per_split,
        "portscan_exclusion_confirmed": all(a not in val_counts for a in unseen),
        "portscan_exclusion_statement": (
            "Zero PortScan rows appear in the training or validation partitions. PortScan is "
            "evaluated only on the final test set, as the unseen-attack score."
        ),
        "train_is_normal_only": per_split["train"]["n_attack"] == 0,
        "dataset_fingerprint_sha256": dataset_fingerprint,
    }


def write_splits(
    directory: str | Path,
    splits: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> None:
    """Freeze the split to disk so every team member uses identical partitions."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        pd.DataFrame({ROW_ID: np.asarray(ids, dtype="int64")}).to_csv(
            out / f"{name}_ids.csv", index=False
        )
    with open(out / "split_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(dict(metadata), handle, indent=2, default=str)
        handle.write("\n")


def load_splits(directory: str | Path) -> dict[str, np.ndarray]:
    """Load frozen split row ids from disk."""
    out = Path(directory)
    splits = {}
    for name in ("train", "validation", "test"):
        path = out / f"{name}_ids.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Frozen split file {path} is missing. Run `python scripts/prepare_data.py` first."
            )
        splits[name] = pd.read_csv(path)[ROW_ID].to_numpy(dtype="int64")
    return splits


def subset(frame: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
    """Return the rows of ``frame`` whose ``source_row_id`` is in ``ids``, order preserved."""
    return frame.loc[frame[ROW_ID].isin(set(ids.tolist()))].reset_index(drop=True)
