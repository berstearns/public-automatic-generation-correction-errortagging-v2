"""Online prediction: run a HuggingFace causal LM over sentences and emit JSONL.

This is the "online" half of the two-input-mode eval contract:

  online   = `predict_online` → `predictions.jsonl` → any eval_scripts.eval_*_table
  offline  = use an existing `predictions.jsonl` or `raw_results.json` directly

`predict_online` only computes per-sentence perplexity. ERRANT-derived
fields (`errors`, `error_types`) are left null/empty — produce those
with the full pipeline (`python -m gen_gec_errant.pipeline ...`) and
emit JSONL via `eval_scripts.raw_to_jsonl`.

Usage:
    python -m eval_scripts.predict_online \\
        --model gpt2 \\
        --data data/sentences.csv \\
        --column sentence \\
        --model_name_label gpt2-native-zero-shot \\
        --out predictions.jsonl

Requires `transformers` and `torch` at runtime; not installed by default
in the eval_scripts dependency set.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Callable, Iterable, Iterator

from eval_scripts._io import write_jsonl


def read_sentences(path: Path, column: str) -> list[str]:
    """Read one column from a CSV file. Returns the values in order."""
    if not path.exists():
        raise SystemExit(f"data file not found: {path}")
    with path.open() as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise SystemExit(
                f"column {column!r} not found in {path}; have: {reader.fieldnames}"
            )
        return [row[column] for row in reader]


def perplexities_via(
    sentences: Iterable[str],
    score_loss: Callable[[str], float | None],
) -> Iterator[float | None]:
    """Map each sentence to a perplexity given a loss-per-sentence function.

    `score_loss(sentence)` returns the mean cross-entropy loss in nats, or
    `None` if the sentence cannot be scored (empty / unsupported). The
    returned iterator yields `exp(loss)` or `None`.
    """
    for s in sentences:
        loss = score_loss(s)
        yield None if loss is None else math.exp(loss)


def _load_hf(model_id: str, device: str):
    """Lazy import + load a HuggingFace causal LM and its tokenizer."""
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover — covered by import-error test
        raise SystemExit(
            "predict_online requires `transformers` and `torch`. "
            "Install with: pip install transformers torch"
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.eval()
    return model, tokenizer


def _hf_loss_fn(model, tokenizer, device: str) -> Callable[[str], float | None]:
    import torch

    def score(sentence: str) -> float | None:
        s = sentence.strip()
        if not s:
            return None
        enc = tokenizer(s, return_tensors="pt", truncation=True).to(device)
        if enc["input_ids"].shape[1] < 2:
            return None
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
        return float(out.loss.item())

    return score


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--data", type=Path, required=True, help="CSV with a sentence column")
    ap.add_argument("--column", default="sentence", help="CSV column name (default: 'sentence')")
    ap.add_argument(
        "--model_name_label",
        default=None,
        help="label written into JSONL records (defaults to --model)",
    )
    ap.add_argument("--out", type=Path, required=True, help="output JSONL")
    ap.add_argument("--device", default="cpu", help="torch device (cpu | cuda | mps)")
    args = ap.parse_args(argv)

    sentences = read_sentences(args.data, args.column)
    model, tokenizer = _load_hf(args.model, args.device)
    score = _hf_loss_fn(model, tokenizer, args.device)
    label = args.model_name_label or args.model

    records = (
        {
            "model": label,
            "item_id": i,
            "ppl": ppl,
            "errors": None,
            "error_types": [],
        }
        for i, ppl in enumerate(perplexities_via(sentences, score))
    )
    n = write_jsonl(args.out, records)
    print(f"predict_online: wrote {n} predictions for {label} to {args.out}")


if __name__ == "__main__":
    main()
