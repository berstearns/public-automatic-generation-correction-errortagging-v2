"""Shared I/O for standalone eval scripts.

Each eval script in this package consumes a `raw_results.json` produced
by the pipeline (`gen_gec_errant.pipeline`) and emits a single CSV
table corresponding to one paper-ready figure.

Schema of `raw_results.json` (top-level keys are model names):

    {
      "<model_name>": {
        "perplexities": [float, ...],
        "error_summary": {
          "errors_per_sentence": [int, ...],
          "total_errors": int,
          "avg_errors_per_sentence": float,
          "error_rate": float,
          "sentences_with_errors": int,
          "total_sentences": int,
          "error_type_counts": {"<ERRANT_CODE>": int, ...},
          ...
        },
        "annotations": [...],          # optional, used by some tables
        "full_text_error_summary": {...},  # optional
        "region_error_summary": {...}      # optional
      },
      ...
    }
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def load_raw_results(path: Path) -> dict[str, dict]:
    """Load and validate `raw_results.json`. Returns dict keyed by model name."""
    if not path.exists():
        raise SystemExit(f"raw_results.json not found at {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}: invalid JSON ({exc})") from exc
    if not isinstance(data, dict) or not data:
        raise SystemExit(f"{path}: expected non-empty dict-of-models")
    return data


def write_csv(out: Path, fieldnames: list[str], rows: Iterable[Mapping[str, Any]]) -> int:
    """Write rows to CSV, creating parent dirs. Returns row count."""
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return len(rows)
