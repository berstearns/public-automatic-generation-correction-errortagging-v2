"""Synthetic fixtures for eval_scripts smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _model_block(ppls, errors_per_sentence, error_type_counts):
    total = sum(error_type_counts.values())
    sentences_with_errors = sum(1 for x in errors_per_sentence if x > 0)
    return {
        "perplexities": ppls,
        "error_summary": {
            "errors_per_sentence": errors_per_sentence,
            "total_errors": total,
            "avg_errors_per_sentence": total / max(1, len(errors_per_sentence)),
            "error_rate": sentences_with_errors / max(1, len(errors_per_sentence)),
            "sentences_with_errors": sentences_with_errors,
            "total_sentences": len(errors_per_sentence),
            "error_type_counts": error_type_counts,
        },
        "annotations": [],
    }


@pytest.fixture
def raw_results_path(tmp_path: Path) -> Path:
    """Two-model synthetic `raw_results.json`."""
    data = {
        "ft-gpt2-small": _model_block(
            ppls=[12.0, 14.0, 13.5, 11.8, 15.2],
            errors_per_sentence=[2, 1, 0, 3, 1],
            error_type_counts={"R:VERB:TENSE": 4, "M:DET": 2, "U:PUNCT": 1},
        ),
        "ft-pythia-160m": _model_block(
            ppls=[18.5, 22.1, 19.7, 20.0, 17.3, 21.4],
            errors_per_sentence=[3, 2, 2, 4, 1, 2],
            error_type_counts={"R:VERB:TENSE": 6, "M:DET": 4, "R:NOUN:NUM": 3, "U:PUNCT": 1},
        ),
    }
    p = tmp_path / "raw_results.json"
    p.write_text(json.dumps(data))
    return p
