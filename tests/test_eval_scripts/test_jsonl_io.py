"""Tests for JSONL I/O: round-trip + cross-format eval-script consumption."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from eval_scripts import (
    eval_errant_category_table,
    eval_error_rate_table,
    eval_perplexity_table,
    raw_to_jsonl,
)
from eval_scripts._io import (
    aggregate_records_to_models,
    iter_jsonl,
    load_input,
    load_jsonl,
    model_dict_to_jsonl_records,
)


def test_iter_jsonl_skips_blank_lines(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}\n\n{"b": 2}\n')
    rows = list(iter_jsonl(p))
    assert rows == [{"a": 1}, {"b": 2}]


def test_aggregate_round_trip_preserves_perplexities(predictions_jsonl_path: Path):
    data = load_jsonl(predictions_jsonl_path)
    assert set(data.keys()) == {"ft-gpt2-small", "ft-pythia-160m"}
    assert data["ft-gpt2-small"]["perplexities"] == [12.0, 14.0, 13.5, 11.8, 15.2]
    es = data["ft-gpt2-small"]["error_summary"]
    assert es["total_sentences"] == 5
    assert es["sentences_with_errors"] == 4
    assert es["total_errors"] == 7
    assert es["error_type_counts"]["R:VERB:TENSE"] == 4
    assert es["error_type_counts"]["M:DET"] == 2
    assert es["error_type_counts"]["U:PUNCT"] == 1


def test_load_input_autodetects_format(raw_results_path: Path, predictions_jsonl_path: Path):
    a = load_input(raw_results_path)
    b = load_input(predictions_jsonl_path)
    assert isinstance(a, dict) and isinstance(b, dict)
    assert "ft-gpt2-small" in a and "ft-gpt2-small" in b


def test_eval_scripts_consume_jsonl_directly(predictions_jsonl_path: Path, tmp_path: Path):
    """All three eval scripts work transparently against JSONL input."""
    out_ppl = tmp_path / "p.csv"
    eval_perplexity_table.main(
        ["--input", str(predictions_jsonl_path), "--dataset", "smoke", "--out", str(out_ppl)]
    )
    rows = list(csv.DictReader(out_ppl.open()))
    assert len(rows) == 2
    assert {r["model"] for r in rows} == {"ft-gpt2-small", "ft-pythia-160m"}

    out_err = tmp_path / "e.csv"
    eval_error_rate_table.main(
        ["--input", str(predictions_jsonl_path), "--dataset", "smoke", "--out", str(out_err)]
    )
    rows = list(csv.DictReader(out_err.open()))
    assert {r["model"] for r in rows} == {"ft-gpt2-small", "ft-pythia-160m"}

    out_cat = tmp_path / "c.csv"
    eval_errant_category_table.main(
        ["--input", str(predictions_jsonl_path), "--dataset", "smoke", "--out", str(out_cat)]
    )
    rows = list(csv.DictReader(out_cat.open()))
    # ft-gpt2-small contributes R:VERB:TENSE, M:DET, U:PUNCT (3 categories)
    # ft-pythia-160m contributes M:DET, R:NOUN:NUM, R:VERB:TENSE (3 categories)
    assert len(rows) == 6


def test_raw_to_jsonl_then_consume(raw_results_path: Path, tmp_path: Path):
    """raw_results.json → JSONL → eval scripts: parities preserved on numeric fields."""
    out_jsonl = tmp_path / "predictions.jsonl"
    raw_to_jsonl.main(["--input", str(raw_results_path), "--out", str(out_jsonl)])
    assert out_jsonl.exists()

    # Re-aggregate JSONL and compare per-model perplexities & error counts to source
    src = json.loads(raw_results_path.read_text())
    rebuilt = load_jsonl(out_jsonl)
    for model in src:
        assert rebuilt[model]["perplexities"] == src[model]["perplexities"]
        # error_types is empty in this round-trip (lossy by design); error_type_counts will be {}
        assert rebuilt[model]["error_summary"]["error_type_counts"] == {}


def test_records_with_missing_fields_are_tolerated(tmp_path: Path):
    """Records may carry only a subset of (ppl, errors, error_types)."""
    p = tmp_path / "partial.jsonl"
    p.write_text(
        '{"model": "m", "item_id": 0, "ppl": 10.0}\n'
        '{"model": "m", "item_id": 1, "errors": 2}\n'
        '{"model": "m", "item_id": 2}\n'
    )
    data = load_jsonl(p)
    es = data["m"]["error_summary"]
    assert data["m"]["perplexities"] == [10.0]
    assert es["errors_per_sentence"] == [2]
    assert es["total_sentences"] == 1  # falls back to len(errors_per_sentence)


def test_model_dict_to_jsonl_records_yields_per_item(raw_results_path: Path):
    src = json.loads(raw_results_path.read_text())
    records = list(model_dict_to_jsonl_records(src))
    by_model: Counter = Counter(r["model"] for r in records)
    assert by_model["ft-gpt2-small"] == 5
    assert by_model["ft-pythia-160m"] == 6
