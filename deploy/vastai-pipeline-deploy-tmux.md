# vastai pipeline deploy via tmux

End-to-end instructions for deploying and running the gen-gec-errant
pipeline on a vastai GPU server using persistent tmux windows.

## Prerequisites

- vastai instance running (check with `vastai show instances`)
- SSH access working (test with `vastai-ssh get-url`)
- rclone configured on the remote with `i:` remote (GDrive)
  - If not: `./orchestrator.sh --mode ssh --ssh-url $SSH_URL copy-rclone`

## Step 0: Get SSH URL

```bash
cd ~/p/all-my-tiny-projects/vastai
SSH_URL=$(./vastai-ssh get-url)
echo $SSH_URL
```

## Step 1: Setup (first time only)

Installs pyenv + Python 3.10.18, clones repo, installs deps, verifies imports.

```bash
./orchestrator.sh --mode ssh --ssh-url "$SSH_URL" project-setup-gen-gec
```

## Step 2: Smoke test

Quick validation: 3 sentences, gpt2-base, GPU. Output: `outputs/smoke-test-dummy/`

```bash
ssh $SSH_URL "tmux new-window -t ssh_tmux -n smoke \
  'export PYENV_ROOT=/root/.pyenv && \
   export PATH=\$PYENV_ROOT/bin:\$PYENV_ROOT/shims:\$PATH && \
   eval \"\$(pyenv init -)\" && \
   bash /workspace/gen-gec-errant/deploy/run-smoke-test.sh; bash'"
```

Check result:
```bash
ssh $SSH_URL "tmux capture-pane -t ssh_tmux:smoke -p -S -20"
```

## Step 3: Run full CELVA pipeline (24 models)

Fetches data + fine-tuned model weights from GDrive, runs all 24 models
(GPT-2, Pythia, SmolLM2 — native + fine-tuned with best checkpoints).

**Important**: The script does `git pull` internally, but bash reads the
script into memory before executing. Always ensure the remote repo is
up-to-date BEFORE launching:

```bash
# 1. Commit and push any local config changes
git add configs/pipeline/celva-all-models.yaml deploy/run-celva-all-models.sh
git commit -m "update configs" && git push origin main

# 2. Pull on remote so the script on disk is current
ssh $SSH_URL "cd /workspace/gen-gec-errant && \
  export PYENV_ROOT=/root/.pyenv && \
  export PATH=\$PYENV_ROOT/bin:\$PYENV_ROOT/shims:\$PATH && \
  eval \"\$(pyenv init -)\" && \
  git pull --ff-only"

# 3. Launch in tmux
ssh $SSH_URL "tmux new-window -t ssh_tmux -n celva \
  'export PYENV_ROOT=/root/.pyenv && \
   export PATH=\$PYENV_ROOT/bin:\$PYENV_ROOT/shims:\$PATH && \
   eval \"\$(pyenv init -)\" && \
   bash /workspace/gen-gec-errant/deploy/run-celva-all-models.sh --sync-results; bash'"
```

## Monitoring

```bash
# List tmux windows
ssh $SSH_URL "tmux list-windows -t ssh_tmux"

# Quick status (last 30 lines)
ssh $SSH_URL "tmux capture-pane -t ssh_tmux:celva -p -S -30"

# Full log
ssh $SSH_URL "tail -50 /workspace/pipeline-celva-sp.log"

# Attach interactively (detach: Ctrl+b, d)
ssh $SSH_URL -t "tmux attach -t ssh_tmux:celva"
```

## Restarting a run

```bash
# Kill existing window
ssh $SSH_URL "tmux kill-window -t ssh_tmux:celva"

# Relaunch (same command as step 3.3 above)
ssh $SSH_URL "tmux new-window -t ssh_tmux -n celva \
  'export PYENV_ROOT=/root/.pyenv && \
   export PATH=\$PYENV_ROOT/bin:\$PYENV_ROOT/shims:\$PATH && \
   eval \"\$(pyenv init -)\" && \
   bash /workspace/gen-gec-errant/deploy/run-celva-all-models.sh --sync-results; bash'"
```

Use `--skip-fetch` if data/models are already downloaded:
```bash
bash /workspace/gen-gec-errant/deploy/run-celva-all-models.sh --skip-fetch --sync-results
```

## Disk space

Check before launching (models + data need ~30GB):
```bash
ssh $SSH_URL "df -h /workspace"
```

Clean up if needed:
```bash
# Remove HuggingFace cache (re-downloads on next run)
ssh $SSH_URL "rm -rf /root/.cache/huggingface/hub"

# Remove old results
ssh $SSH_URL "rm -rf /workspace/gec-results/*"

# Remove old fine-tuned model checkpoints (if switching from final/ to best/)
ssh $SSH_URL "find /workspace/ft-models -name 'final' -type d -exec rm -rf {} + 2>/dev/null"
```

## Files

| File | Purpose |
|------|---------|
| `deploy/project-setup-remote.sh` | Pyenv + deps setup (run once) |
| `deploy/run-smoke-test.sh` | Quick GPU validation |
| `deploy/run-celva-all-models.sh` | Full 24-model CELVA pipeline |
| `configs/pipeline/smoke-test.yaml` | Smoke test config (3 sents, CPU-safe) |
| `configs/pipeline/celva-all-models.yaml` | Full config (32GB batch sizes, best/ checkpoints) |

## Orchestrator steps (in ~/p/all-my-tiny-projects/vastai/)

| Step | Description |
|------|-------------|
| `project-setup-gen-gec` | Setup: pyenv, clone, deps, verify imports |
| `run-gen-gec-smoke` | Smoke test (piped via SSH, no tmux) |
