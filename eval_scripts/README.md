# eval_scripts/

Standalone evaluation scripts. Each script consumes one input file
(either `raw_results.json` from the pipeline or a `predictions.jsonl`
matching the schema below) and emits one CSV table corresponding to
one paper-ready figure.

These scripts depend only on the Python standard library, so they can
be run on a machine that does not have the full `gen_gec_errant`
package installed (e.g. a Colab cell with only the JSON output).

## Tables

| Script                              | Output CSV               | Columns |
|-------------------------------------|--------------------------|---------|
| `eval_perplexity_table.py`          | `perplexity.csv`         | model, dataset, n, ppl_mean, ppl_median, ppl_std, ppl_25th, ppl_75th |
| `eval_error_rate_table.py`          | `error_rate.csv`         | model, dataset, total_sentences, sentences_with_errors, error_rate, total_errors, avg_errors_per_sentence |
| `eval_errant_category_table.py`     | `errant_categories.csv`  | model, dataset, error_type, count, proportion |

## Running one table

```bash
python -m eval_scripts.eval_perplexity_table \
    --input runs/2026-04-27_celva/raw_results.json \
    --dataset CELVA-SP \
    --out tables/perplexity.csv
```

## Running every table at once

```bash
python -m eval_scripts.run_all_tables \
    --input runs/2026-04-27_celva/raw_results.json \
    --dataset CELVA-SP \
    --out_dir tables/
```

## Adding a new table

1. Create `eval_scripts/eval_<name>_table.py`.
2. Implement `build_rows(data, dataset, ...)` and a `main(argv=None)` entry point.
3. Use `from eval_scripts._io import load_raw_results, write_csv`.
4. Add a smoke test under `tests/eval_scripts/`.
5. Add the script to the orchestrator in `run_all_tables.py`.
6. Update the table in this README.

## Input formats

Eval scripts auto-detect the input format from the file extension:

- `raw_results.json` — pipeline output. Dict keyed by model name; each
  value contains `perplexities`, `error_summary`, `annotations`, and
  optionally `full_text_error_summary` / `region_error_summary`.
- `predictions.jsonl` — line-delimited JSON, one record per
  (model, item):

  ```jsonl
  {"model": "<name>", "item_id": <int>, "ppl": <float|null>, "errors": <int|null>, "error_types": [<str>, ...]}
  ```

  All fields except `model` are optional; missing values do not
  contribute to the corresponding aggregate. New prediction scripts
  should emit JSONL directly so per-item ERRANT codes (`error_types`)
  are preserved.

Convert one to the other with:

```bash
python -m eval_scripts.raw_to_jsonl --input raw_results.json --out predictions.jsonl
```

(The reverse — JSONL → raw_results.json — is just `load_input(...)`
in Python; eval scripts call it on every run, so explicit conversion
is rarely needed.)

See `eval_scripts/_io.py` for the canonical schema definitions.
