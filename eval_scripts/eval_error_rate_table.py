"""Standalone eval: per-model error-rate table.

Columns:
    model | dataset | total_sentences | sentences_with_errors | error_rate
          | total_errors | avg_errors_per_sentence

`error_rate` here is the fraction of sentences carrying at least one ERRANT
edit, as already computed upstream in `error_summary["error_rate"]`.

Usage:
    python -m eval_scripts.eval_error_rate_table \\
        --input <run_dir>/raw_results.json \\
        --dataset CELVA-SP \\
        --out tables/error_rate.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eval_scripts._io import load_input, write_csv


_FIELDS = [
    "model", "dataset",
    "total_sentences", "sentences_with_errors", "error_rate",
    "total_errors", "avg_errors_per_sentence",
]


def compute_row(model: str, dataset: str, error_summary: dict) -> dict:
    return {
        "model": model,
        "dataset": dataset,
        "total_sentences": int(error_summary.get("total_sentences", 0)),
        "sentences_with_errors": int(error_summary.get("sentences_with_errors", 0)),
        "error_rate": float(error_summary.get("error_rate", 0.0)),
        "total_errors": int(error_summary.get("total_errors", 0)),
        "avg_errors_per_sentence": float(error_summary.get("avg_errors_per_sentence", 0.0)),
    }


def build_rows(data: dict[str, dict], dataset: str) -> list[dict]:
    return [compute_row(m, dataset, r.get("error_summary") or {}) for m, r in data.items()]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="raw_results.json from a pipeline run")
    ap.add_argument("--dataset", type=str, default="", help="dataset label attached to every row")
    ap.add_argument("--out", type=Path, required=True, help="output CSV path")
    args = ap.parse_args(argv)

    data = load_input(args.input)
    rows = build_rows(data, args.dataset)
    n = write_csv(args.out, _FIELDS, rows)
    print(f"error-rate table: wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
