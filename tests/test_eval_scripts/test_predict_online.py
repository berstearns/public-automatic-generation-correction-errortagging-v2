"""Tests for predict_online: pure-Python parts only.

The actual HF model loading is not exercised here (would require
`transformers` + `torch` and a downloaded model); the model-aware
flow is tested via dependency injection of a fake `score_loss`.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from eval_scripts import predict_online
from eval_scripts._io import write_jsonl


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def test_read_sentences_basic(tmp_path: Path):
    p = tmp_path / "sentences.csv"
    _write_csv(
        p,
        [{"sentence": "Hello world.", "id": "1"}, {"sentence": "I has go.", "id": "2"}],
        ["sentence", "id"],
    )
    assert predict_online.read_sentences(p, "sentence") == ["Hello world.", "I has go."]


def test_read_sentences_missing_column(tmp_path: Path):
    p = tmp_path / "sentences.csv"
    _write_csv(p, [{"text": "Hi"}], ["text"])
    with pytest.raises(SystemExit) as exc:
        predict_online.read_sentences(p, "sentence")
    assert "column 'sentence' not found" in str(exc.value)


def test_perplexities_via_returns_exp_of_loss():
    losses = {"a": 0.0, "b": 1.0, "c": math.log(4.0), "": None}
    score = lambda s: losses[s]  # noqa: E731
    out = list(predict_online.perplexities_via(["a", "b", "c", ""], score))
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(math.e)
    assert out[2] == pytest.approx(4.0)
    assert out[3] is None


def test_predict_online_end_to_end_with_fake_model(tmp_path: Path, monkeypatch):
    """Patch HF loading + scoring so the CLI runs without transformers/torch."""
    csv_path = tmp_path / "sentences.csv"
    _write_csv(
        csv_path,
        [{"sentence": "alpha"}, {"sentence": "bravo charlie"}, {"sentence": ""}],
        ["sentence"],
    )
    out = tmp_path / "predictions.jsonl"

    fake_losses = {"alpha": 0.5, "bravo charlie": 1.0}
    def fake_load_hf(model_id: str, device: str):
        return ("fake-model", "fake-tokenizer")
    def fake_loss_fn(model, tokenizer, device: str):
        def score(s: str) -> float | None:
            s = s.strip()
            return fake_losses.get(s)
        return score

    monkeypatch.setattr(predict_online, "_load_hf", fake_load_hf)
    monkeypatch.setattr(predict_online, "_hf_loss_fn", fake_loss_fn)

    predict_online.main([
        "--model", "test/dummy",
        "--data", str(csv_path),
        "--column", "sentence",
        "--model_name_label", "fake",
        "--out", str(out),
        "--device", "cpu",
    ])

    records = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(records) == 3
    assert records[0] == {"model": "fake", "item_id": 0, "ppl": pytest.approx(math.exp(0.5)), "errors": None, "error_types": []}
    assert records[1]["ppl"] == pytest.approx(math.exp(1.0))
    assert records[2]["ppl"] is None  # empty sentence


def test_predict_online_then_eval_perplexity(tmp_path: Path, monkeypatch):
    """End-to-end: predict_online → predictions.jsonl → eval_perplexity_table."""
    from eval_scripts import eval_perplexity_table

    csv_path = tmp_path / "sentences.csv"
    _write_csv(
        csv_path,
        [{"sentence": f"sent-{i}"} for i in range(4)],
        ["sentence"],
    )
    jsonl = tmp_path / "predictions.jsonl"

    losses = {f"sent-{i}": float(i + 1) * 0.5 for i in range(4)}
    monkeypatch.setattr(predict_online, "_load_hf", lambda m, d: (None, None))
    monkeypatch.setattr(
        predict_online,
        "_hf_loss_fn",
        lambda m, t, d: (lambda s: losses.get(s.strip())),
    )
    predict_online.main([
        "--model", "x", "--data", str(csv_path), "--column", "sentence",
        "--out", str(jsonl), "--model_name_label", "fake",
    ])

    out_csv = tmp_path / "p.csv"
    eval_perplexity_table.main(
        ["--input", str(jsonl), "--dataset", "smoke", "--out", str(out_csv)]
    )
    rows = list(csv.DictReader(out_csv.open()))
    assert len(rows) == 1
    assert rows[0]["model"] == "fake"
    assert int(rows[0]["n"]) == 4
    assert float(rows[0]["ppl_mean"]) > 1.0
