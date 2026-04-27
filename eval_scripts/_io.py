"""Shared I/O for standalone eval scripts.

Eval scripts in this package consume **either** of two formats and emit
one CSV table corresponding to one paper-ready figure:

1. `raw_results.json` — produced by `gen_gec_errant.pipeline`. Top-level
   dict keyed by model name. See `load_raw_results` for the full shape.

2. `predictions.jsonl` — line-delimited JSON, one record per (model,
   item). Schema:

       {"model": "<name>", "item_id": <int>,
        "ppl": <float|null>, "errors": <int|null>,
        "error_types": [<str>, ...]}

   `ppl`, `errors`, and `error_types` are optional; missing values are
   tolerated (records that lack a field simply do not contribute to the
   corresponding aggregate). This makes the schema additive: a future
   prediction script can emit only the fields it has, and existing eval
   scripts will keep working.

`load_input(path)` auto-detects the format by file extension (`.jsonl`
→ JSONL aggregated to model-keyed dict, anything else → JSON dict-of-models)
and is the entry point every eval script should use.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


def load_raw_results(path: Path) -> dict[str, dict]:
    """Load `raw_results.json`. Returns dict keyed by model name."""
    if not path.exists():
        raise SystemExit(f"raw_results.json not found at {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}: invalid JSON ({exc})") from exc
    if not isinstance(data, dict) or not data:
        raise SystemExit(f"{path}: expected non-empty dict-of-models")
    return data


def iter_jsonl(path: Path) -> Iterator[dict]:
    """Yield each line of a JSONL file as a parsed dict. Skips blank lines."""
    if not path.exists():
        raise SystemExit(f"jsonl not found at {path}")
    with path.open() as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: invalid JSON ({exc})") from exc


def aggregate_records_to_models(records: Iterable[dict]) -> dict[str, dict]:
    """Group per-item records by model and rebuild the dict-of-models shape.

    Output matches `raw_results.json` for the fields eval scripts need:
    `perplexities`, `error_summary` (with all derived counts/rates),
    `annotations` (left empty — JSONL doesn't carry annotation objects).
    """
    by_model: dict[str, list[dict]] = {}
    for r in records:
        m = r.get("model")
        if not isinstance(m, str) or not m:
            continue
        by_model.setdefault(m, []).append(r)

    out: dict[str, dict] = {}
    for model, items in by_model.items():
        items_sorted = sorted(items, key=lambda r: r.get("item_id", 0))
        ppls = [r["ppl"] for r in items_sorted if isinstance(r.get("ppl"), (int, float))]
        errors_per_sentence = [
            int(r["errors"]) for r in items_sorted if isinstance(r.get("errors"), (int, float))
        ]
        type_counts: Counter[str] = Counter()
        for r in items_sorted:
            type_counts.update(r.get("error_types") or [])

        total_errors = sum(errors_per_sentence)
        n = len(errors_per_sentence) or len(items_sorted)
        sentences_with_errors = sum(1 for x in errors_per_sentence if x > 0)
        out[model] = {
            "perplexities": ppls,
            "error_summary": {
                "errors_per_sentence": errors_per_sentence,
                "total_errors": total_errors,
                "avg_errors_per_sentence": (total_errors / n) if n else 0.0,
                "error_rate": (sentences_with_errors / n) if n else 0.0,
                "sentences_with_errors": sentences_with_errors,
                "total_sentences": n,
                "error_type_counts": dict(type_counts),
            },
            "annotations": [],
        }
    if not out:
        raise SystemExit("aggregated jsonl produced no models — every record missing 'model'?")
    return out


def load_jsonl(path: Path) -> dict[str, dict]:
    """Load JSONL and aggregate to dict-of-models shape (same as raw_results)."""
    return aggregate_records_to_models(iter_jsonl(path))


def load_input(path: Path) -> dict[str, dict]:
    """Auto-detect input format. `.jsonl` → JSONL; anything else → JSON."""
    if path.suffix.lower() == ".jsonl":
        return load_jsonl(path)
    return load_raw_results(path)


def model_dict_to_jsonl_records(data: dict[str, dict]) -> Iterator[dict]:
    """Reverse of `aggregate_records_to_models`: yield per-(model, item) records.

    Per-item `error_types` is left empty since `raw_results.json` does not
    preserve per-sentence ERRANT codes (only the aggregated counts). This
    is documented and accepted as lossy-down-then-up; downstream tables
    that need per-item categories should consume JSONL produced by a
    richer prediction script directly.
    """
    for model, block in data.items():
        ppls = block.get("perplexities") or []
        es = block.get("error_summary") or {}
        errs = es.get("errors_per_sentence") or []
        n = max(len(ppls), len(errs))
        for i in range(n):
            yield {
                "model": model,
                "item_id": i,
                "ppl": ppls[i] if i < len(ppls) else None,
                "errors": errs[i] if i < len(errs) else None,
                "error_types": [],
            }


def write_csv(out: Path, fieldnames: list[str], rows: Iterable[Mapping[str, Any]]) -> int:
    """Write rows to CSV, creating parent dirs. Returns row count."""
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def write_jsonl(out: Path, records: Iterable[dict]) -> int:
    """Write JSONL, creating parent dirs. Returns line count."""
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            n += 1
    return n
