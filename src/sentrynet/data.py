"""Dataset loading, schema verification, deduplication, and the privacy review.

The dataset is the Kaggle "Cyber Security Attack Using Network Traffic" set (juanschafle,
v1). It's synthetic and has no timestamp, IP address, user/host identifier, or free text,
so there's nothing here to hash or mask.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

# If any of these show up, the schema has changed and this pipeline needs a second look.
FORBIDDEN_TIME_COLUMNS = ("timestamp", "time", "date", "datetime", "start_time", "end_time")
FORBIDDEN_IDENTIFIER_COLUMNS = (
    "src_ip",
    "dst_ip",
    "ip",
    "ip_address",
    "user",
    "user_id",
    "username",
    "host",
    "hostname",
    "mac",
    "mac_address",
    "session_id",
)
FORBIDDEN_TEXT_COLUMNS = ("message", "log", "payload", "text", "description", "raw_log")

# Row counter for traceability, not a real identifier.
ROW_ID = "source_row_id"

PRIVACY_STATEMENT = (
    "No PII/identifier masking is required for the published feature set."
)


@dataclass
class VerificationResult:
    """Outcome of :func:`verify_dataset`."""

    status: str
    checks: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "PASS"

    def add(self, name: str, passed: bool, expected: Any, actual: Any, material: bool = True) -> None:
        self.checks.append(
            {
                "check": name,
                "expected": expected,
                "actual": actual,
                "passed": bool(passed),
                "material": bool(material),
            }
        )
        if not passed and material:
            self.status = "FAIL"

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "checks": self.checks, "summary": self.summary}


def file_fingerprint(path: str | Path) -> str:
    """SHA-256 of the raw file, so every run records exactly which bytes it consumed."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw(path: str | Path, add_row_id: bool = True) -> pd.DataFrame:
    """Load the raw CSV, preserving row order (the row-order split protocol depends on it)."""
    frame = pd.read_csv(path)
    if add_row_id:
        frame.insert(0, ROW_ID, range(len(frame)))
    return frame


def verify_dataset(frame: pd.DataFrame, dataset_cfg: dict[str, Any]) -> VerificationResult:
    """Validate the CSV against the expected schema.

    A material failure means the data doesn't match what the pipeline was built for, and
    the caller should stop rather than silently adapt to it.
    """
    result = VerificationResult(status="PASS")
    expected_columns = list(dataset_cfg["expected_columns"])
    columns = [c for c in frame.columns if c != ROW_ID]

    result.add("columns_exact_match", columns == expected_columns, expected_columns, columns)
    result.add(
        "row_count",
        len(frame) == int(dataset_cfg["expected_rows"]),
        int(dataset_cfg["expected_rows"]),
        len(frame),
    )

    label = dataset_cfg["label_column"]
    if label in frame.columns:
        actual_counts = frame[label].value_counts().to_dict()
        expected_counts = dict(dataset_cfg["expected_class_counts"])
        tolerance = float(dataset_cfg.get("class_count_tolerance", 0.0))
        for cls, expected_n in expected_counts.items():
            actual_n = int(actual_counts.get(cls, 0))
            allowed = max(0.0, tolerance * expected_n)
            result.add(
                f"class_count::{cls}",
                abs(actual_n - expected_n) <= allowed,
                expected_n,
                actual_n,
            )
        result.add(
            "no_unexpected_classes",
            set(actual_counts) <= set(expected_counts),
            sorted(expected_counts),
            sorted(actual_counts),
        )
    else:
        result.add("label_column_present", False, label, None)
        actual_counts = {}

    lowered = {c.lower() for c in frame.columns}
    result.add(
        "no_timestamp_column",
        not (lowered & set(FORBIDDEN_TIME_COLUMNS)),
        "none of %s" % (FORBIDDEN_TIME_COLUMNS,),
        sorted(lowered & set(FORBIDDEN_TIME_COLUMNS)),
    )
    result.add(
        "no_identifier_column",
        not (lowered & set(FORBIDDEN_IDENTIFIER_COLUMNS)),
        "none of %s" % (FORBIDDEN_IDENTIFIER_COLUMNS,),
        sorted(lowered & set(FORBIDDEN_IDENTIFIER_COLUMNS)),
    )
    result.add(
        "no_free_text_column",
        not (lowered & set(FORBIDDEN_TEXT_COLUMNS)),
        "none of %s" % (FORBIDDEN_TEXT_COLUMNS,),
        sorted(lowered & set(FORBIDDEN_TEXT_COLUMNS)),
    )

    missing = frame.isna().sum()
    total_missing = int(missing.sum())
    result.add("no_missing_values", total_missing == 0, 0, total_missing, material=False)

    result.summary = {
        "n_rows": int(len(frame)),
        "n_columns": len(columns),
        "columns": columns,
        "dtypes": {c: str(frame[c].dtype) for c in frame.columns},
        "class_counts": {k: int(v) for k, v in actual_counts.items()},
        "missing_values": {c: int(missing[c]) for c in frame.columns},
        "privacy_statement": PRIVACY_STATEMENT,
        "near_duplicate_detection": (
            "Not required: the dataset has no timestamp, IP, user, or session identifier, "
            "so only exact-row matching is meaningful."
        ),
    }
    return result


def count_exact_duplicates(frame: pd.DataFrame, ignore: tuple[str, ...] = (ROW_ID,)) -> int:
    """Number of rows that are exact duplicates of an earlier row."""
    subset = [c for c in frame.columns if c not in ignore]
    return int(frame.duplicated(subset=subset, keep="first").sum())


def remove_exact_duplicates(
    frame: pd.DataFrame, ignore: tuple[str, ...] = (ROW_ID,)
) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicate rows, keeping the first occurrence.

    Returns the cleaned frame and the number of rows removed. Only *exact* duplicates are
    removed; near-duplicate detection is out of scope for this schema.
    """
    subset = [c for c in frame.columns if c not in ignore]
    mask = frame.duplicated(subset=subset, keep="first")
    removed = int(mask.sum())
    return frame.loc[~mask].reset_index(drop=True), removed


def privacy_review(frame: pd.DataFrame) -> dict[str, Any]:
    """Produce the privacy review record.

    The feature set has no IP addresses, user/host identifiers, MAC addresses, or free
    text, so there's nothing to mask, and nothing is hashed.
    """
    lowered = {c.lower() for c in frame.columns}
    return {
        "columns_reviewed": sorted(frame.columns),
        "identifier_columns_found": sorted(lowered & set(FORBIDDEN_IDENTIFIER_COLUMNS)),
        "free_text_columns_found": sorted(lowered & set(FORBIDDEN_TEXT_COLUMNS)),
        "timestamp_columns_found": sorted(lowered & set(FORBIDDEN_TIME_COLUMNS)),
        "masking_performed": False,
        "hashing_performed": False,
        "conclusion": PRIVACY_STATEMENT,
        "note": (
            "This conclusion is revisited only if identifier fields are later added to the "
            "dataset. The added source_row_id is a positional counter, not an identifier of "
            "a person, host, or session."
        ),
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False, default=str)
        handle.write("\n")
