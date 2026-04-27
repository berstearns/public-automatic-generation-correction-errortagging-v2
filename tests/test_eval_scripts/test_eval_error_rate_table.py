from __future__ import annotations

import csv
from pathlib import Path

from eval_scripts import eval_error_rate_table


def test_error_rate_table_smoke(raw_results_path: Path, tmp_path: Path):
    out = tmp_path / "error_rate.csv"
    eval_error_rate_table.main(
        ["--input", str(raw_results_path), "--dataset", "smoke", "--out", str(out)]
    )
    assert out.exists()
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    by_model = {r["model"]: r for r in rows}
    small = by_model["ft-gpt2-small"]
    assert int(small["total_sentences"]) == 5
    assert int(small["sentences_with_errors"]) == 4
    assert 0.0 < float(small["error_rate"]) <= 1.0
    assert int(small["total_errors"]) == 4 + 2 + 1
