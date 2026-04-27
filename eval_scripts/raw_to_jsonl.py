"""Convert `raw_results.json` to `predictions.jsonl`.

JSONL schema (one record per (model, item)):

    {"model": "<name>", "item_id": <int>,
     "ppl": <float|null>, "errors": <int|null>,
     "error_types": [<str>, ...]}

`error_types` is left empty here because `raw_results.json` does not
preserve per-sentence ERRANT codes (only model-aggregated counts in
`error_summary.error_type_counts`). A richer prediction script that
emits this JSONL directly should fill `error_types` per item.

Usage:
    python -m eval_scripts.raw_to_jsonl \\
        --input runs/2026-04-27_celva/raw_results.json \\
        --out runs/2026-04-27_celva/predictions.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eval_scripts._io import load_raw_results, model_dict_to_jsonl_records, write_jsonl


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="raw_results.json")
    ap.add_argument("--out", type=Path, required=True, help="output predictions.jsonl")
    args = ap.parse_args(argv)

    data = load_raw_results(args.input)
    n = write_jsonl(args.out, model_dict_to_jsonl_records(data))
    print(f"raw_to_jsonl: wrote {n} records to {args.out}")


if __name__ == "__main__":
    main()
