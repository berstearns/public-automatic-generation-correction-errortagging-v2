# runs/

Per-run output artifacts: `predictions.jsonl` plus the CSV tables
emitted by `eval_scripts/`. One subdirectory per (model, dataset, regime)
tuple.

Run-config naming convention:

    runs/<model_label>-<dataset_label>[-<note>]/

Each subdirectory contains:

    predictions.jsonl       # one record per (model, item)
    tables/
      perplexity.csv
      error_rate.csv
      errant_categories.csv

These are committed as evidence of what the harness has been actually
run against. The input `data/*.csv` files referenced during the run
are NOT committed — they live in the private rclone remote.

## Existing runs

| Subdirectory                                  | Model              | Data              | n  | What it shows |
|-----------------------------------------------|--------------------|-------------------|----|---------------|
| `gpt2-native-zero-shot-celva-smoke/`          | `gpt2` (124M)      | CELVA-SP 2-sample | 2  | end-to-end harness smoke against an off-the-shelf no-training model; PPL only (`predict_online` does not run GEC/ERRANT) |
