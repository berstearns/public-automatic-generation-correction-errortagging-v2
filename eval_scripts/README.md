# eval_scripts/

Standalone evaluation scripts. Each script consumes one
`raw_results.json` produced by `gen_gec_errant.pipeline` and emits one
CSV table corresponding to one paper-ready figure.

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

## Schema reference

`raw_results.json` is a dict keyed by model name. Each value contains
`perplexities`, `error_summary`, `annotations`, and optionally
`full_text_error_summary` and `region_error_summary`. See
`eval_scripts/_io.py` for the full shape.
