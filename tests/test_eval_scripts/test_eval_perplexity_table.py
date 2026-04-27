from __future__ import annotations

import csv
from pathlib import Path

from eval_scripts import eval_perplexity_table


def test_perplexity_table_smoke(raw_results_path: Path, tmp_path: Path):
    out = tmp_path / "perplexity.csv"
    eval_perplexity_table.main(
        ["--input", str(raw_results_path), "--dataset", "smoke", "--out", str(out)]
    )
    assert out.exists()
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert {r["model"] for r in rows} == {"ft-gpt2-small", "ft-pythia-160m"}
    assert all(r["dataset"] == "smoke" for r in rows)
    for r in rows:
        assert int(r["n"]) > 0
        assert float(r["ppl_std"]) >= 0.0
        # mean lies between 25th and 75th percentile bounds
        assert float(r["ppl_25th"]) <= float(r["ppl_mean"]) <= float(r["ppl_75th"]) + 1e-6
