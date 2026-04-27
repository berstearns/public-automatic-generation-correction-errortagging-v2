from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from eval_scripts import eval_errant_category_table


def test_errant_category_table_smoke(raw_results_path: Path, tmp_path: Path):
    out = tmp_path / "errant_categories.csv"
    eval_errant_category_table.main(
        ["--input", str(raw_results_path), "--dataset", "smoke", "--out", str(out)]
    )
    assert out.exists()
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert {r["model"] for r in rows} == {"ft-gpt2-small", "ft-pythia-160m"}

    # Rows for one model sum to ~1.0 in proportion
    by_model: dict[str, list] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    for m, mrows in by_model.items():
        assert abs(sum(float(r["proportion"]) for r in mrows) - 1.0) < 1e-9


def test_errant_category_top_n(raw_results_path: Path, tmp_path: Path):
    out = tmp_path / "top2.csv"
    eval_errant_category_table.main(
        ["--input", str(raw_results_path), "--dataset", "smoke", "--top_n", "2", "--out", str(out)]
    )
    with out.open() as f:
        rows = list(csv.DictReader(f))
    # 2 models × 2 top categories each
    assert len(rows) == 4
