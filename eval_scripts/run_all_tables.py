"""Convenience wrapper: produce every standalone eval table for one run.

Calls each `eval_scripts.eval_*_table` module in turn against a single
`raw_results.json` and writes all outputs into a chosen directory.

Usage:
    python -m eval_scripts.run_all_tables \\
        --input <run_dir>/raw_results.json \\
        --dataset CELVA-SP \\
        --out_dir tables/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eval_scripts import (
    eval_errant_category_table,
    eval_error_rate_table,
    eval_perplexity_table,
)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="raw_results.json from a pipeline run")
    ap.add_argument("--dataset", type=str, default="", help="dataset label attached to every row")
    ap.add_argument("--out_dir", type=Path, required=True, help="directory to write the CSVs into")
    ap.add_argument("--top_n", type=int, default=10, help="ERRANT categories: keep top-N per model")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    common = ["--input", str(args.input), "--dataset", args.dataset]

    eval_perplexity_table.main(common + ["--out", str(args.out_dir / "perplexity.csv")])
    eval_error_rate_table.main(common + ["--out", str(args.out_dir / "error_rate.csv")])
    eval_errant_category_table.main(
        common + ["--top_n", str(args.top_n), "--out", str(args.out_dir / "errant_categories.csv")]
    )


if __name__ == "__main__":
    main()
