# deploy/

This directory has two roles:

1. **Runtime deploy scripts** (in this directory) — `setup_and_run.sh`,
   `run-celva-all-models.sh`, `vastai-pipeline-deploy-tmux.md`, etc.
   These are scrubbed of hardcoded paths; you'll need to fill in the
   placeholders below before running.
2. **Drop-in patch target** for sensitive overrides held in a private
   backup (rclone remote, see private deploy/README inside the backup).

## Restoring the unscrubbed working state

```bash
# 1. Get the code
git clone https://github.com/berstearns/public-automatic-generation-correction-errortagging-v2
cd public-automatic-generation-correction-errortagging-v2

# 2. Layer real configs/scripts over placeholders (drop-in patch from private backup)
rclone copy <your-private-remote>/deploy/ .

# 3. (optional) Pull trained model checkpoints + sample run outputs
rclone copy <your-private-remote>/resources/ .
```

## Placeholders to fill in

The public mirror replaces real paths with placeholders. Set these
to your local values (either by editing in place after cloning, or by
overlaying via rclone from the private backup):

| Placeholder                       | What it means                                       |
|-----------------------------------|-----------------------------------------------------|
| `./data/splits`                   | local cefr-classification SLA dataset splits dir    |
| `./models`                        | local fine-tuned model checkpoints root             |
| `i:/<your-rclone-models>/`        | rclone path to fine-tuned model checkpoints         |
| `i:/<your-rclone-root>/`          | rclone path to artificial-learners root             |
