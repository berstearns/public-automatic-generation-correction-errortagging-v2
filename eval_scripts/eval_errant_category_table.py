"""Standalone eval: per-model ERRANT error-category breakdown.

Columns: model | dataset | error_type | count | proportion

`proportion` is `count / sum(count for the same model)` (rows under the
same model sum to ~1.0). With `--top_n N` (default 0 = all), only the
top-N error types per model are emitted.

Usage:
    python -m eval_scripts.eval_errant_category_table \\
        --input <run_dir>/raw_results.json \\
        --dataset CELVA-SP \\
        --top_n 10 \\
        --out tables/errant_categories.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eval_scripts._io import load_input, write_csv


_FIELDS = ["model", "dataset", "error_type", "count", "proportion"]


def build_rows(data: dict[str, dict], dataset: str, top_n: int = 0) -> list[dict]:
    rows: list[dict] = []
    for model, model_data in data.items():
        counts = (model_data.get("error_summary") or {}).get("error_type_counts") or {}
        items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        if top_n > 0:
            items = items[:top_n]
        total = sum(counts.values()) or 1
        for err_type, count in items:
            rows.append({
                "model": model,
                "dataset": dataset,
                "error_type": err_type,
                "count": int(count),
                "proportion": count / total,
            })
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="raw_results.json from a pipeline run")
    ap.add_argument("--dataset", type=str, default="", help="dataset label attached to every row")
    ap.add_argument("--top_n", type=int, default=0, help="Keep only top-N error types per model (0 = all)")
    ap.add_argument("--out", type=Path, required=True, help="output CSV path")
    args = ap.parse_args(argv)

    data = load_input(args.input)
    rows = build_rows(data, args.dataset, top_n=args.top_n)
    n = write_csv(args.out, _FIELDS, rows)
    print(f"ERRANT category table: wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
