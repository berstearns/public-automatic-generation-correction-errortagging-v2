"""Standalone eval: per-model perplexity table.

Columns: model | dataset | n | ppl_mean | ppl_median | ppl_std | ppl_25th | ppl_75th

Usage:
    python -m eval_scripts.eval_perplexity_table \\
        --input <run_dir>/raw_results.json \\
        --dataset CELVA-SP \\
        --out tables/perplexity.csv
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from eval_scripts._io import load_input, write_csv


_FIELDS = ["model", "dataset", "n", "ppl_mean", "ppl_median", "ppl_std", "ppl_25th", "ppl_75th"]


def compute_row(model: str, dataset: str, ppls: list[float]) -> dict:
    if not ppls:
        return {f: 0.0 for f in _FIELDS} | {"model": model, "dataset": dataset, "n": 0}
    if len(ppls) >= 2:
        q25, _q50, q75 = statistics.quantiles(ppls, n=4, method="inclusive")
        std = statistics.stdev(ppls)
    else:
        q25 = q75 = ppls[0]
        std = 0.0
    return {
        "model": model,
        "dataset": dataset,
        "n": len(ppls),
        "ppl_mean": statistics.fmean(ppls),
        "ppl_median": statistics.median(ppls),
        "ppl_std": std,
        "ppl_25th": q25,
        "ppl_75th": q75,
    }


def build_rows(data: dict[str, dict], dataset: str) -> list[dict]:
    return [compute_row(m, dataset, r.get("perplexities") or []) for m, r in data.items()]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="raw_results.json from a pipeline run")
    ap.add_argument("--dataset", type=str, default="", help="dataset label attached to every row")
    ap.add_argument("--out", type=Path, required=True, help="output CSV path")
    args = ap.parse_args(argv)

    data = load_input(args.input)
    rows = build_rows(data, args.dataset)
    n = write_csv(args.out, _FIELDS, rows)
    print(f"perplexity table: wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
