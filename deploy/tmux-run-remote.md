# Running scripts on a remote server via tmux

Pattern for launching any bash script in a persistent tmux window on a
vastai (or any SSH-accessible) server. The script survives disconnects.

## Prerequisites

- SSH access: `ssh://root@HOST:PORT`
- tmux session already running on the remote (vastai auto-creates `ssh_tmux`)
- Script committed and pushed to GitHub
- `project-setup-gen-gec` already run (repo cloned at `/workspace/gen-gec-errant`)

## Quick reference

```bash
SSH_URL="ssh://root@HOST:PORT"

# 1. Pull latest code on remote
ssh $SSH_URL "cd /workspace/gen-gec-errant && git pull --ff-only"

# 2. Launch script in a new tmux window
ssh $SSH_URL "tmux new-window -t ssh_tmux -n WINDOW_NAME \
  'export PYENV_ROOT=/root/.pyenv && \
   export PATH=\$PYENV_ROOT/bin:\$PYENV_ROOT/shims:\$PATH && \
   eval \"\$(pyenv init -)\" && \
   bash /workspace/gen-gec-errant/deploy/YOUR_SCRIPT.sh; bash'"

# 3. Check status
ssh $SSH_URL "tmux list-windows -t ssh_tmux"

# 4. Read output (last 50 lines)
ssh $SSH_URL "tmux capture-pane -t ssh_tmux:WINDOW_NAME -p -S -50"

# 5. Attach interactively
ssh $SSH_URL -t "tmux attach -t ssh_tmux:WINDOW_NAME"
```

The trailing `; bash` keeps the window open after the script finishes so
you can inspect output or re-run.

## Via orchestrator

To integrate a new script into the orchestrator pattern, register it in
`~/p/all-my-tiny-projects/vastai/orchestrator.sh`:

```bash
# In STEP_SCRIPT:
STEP_SCRIPT[my-step]='$HOME/p/research-sketches/.../deploy/my-script.sh'

# In STEP_DESC:
STEP_DESC[my-step]="Description of what this step does"

# In STEP_ORDER (append before builtins):
... my-step copy copy-rclone logs stop-remote)
```

Then run:
```bash
./orchestrator.sh --mode ssh --ssh-url $SSH_URL my-step
```

This pipes the script via SSH (no tmux). For long-running jobs use the
tmux pattern above instead, or use `--detach` which backgrounds with nohup.

## Examples

### Smoke test
```bash
ssh $SSH_URL "tmux new-window -t ssh_tmux -n smoke \
  'export PYENV_ROOT=/root/.pyenv && \
   export PATH=\$PYENV_ROOT/bin:\$PYENV_ROOT/shims:\$PATH && \
   eval \"\$(pyenv init -)\" && \
   bash /workspace/gen-gec-errant/deploy/run-smoke-test.sh; bash'"
```

### CELVA all models (long-running)
```bash
ssh $SSH_URL "tmux new-window -t ssh_tmux -n celva \
  'export PYENV_ROOT=/root/.pyenv && \
   export PATH=\$PYENV_ROOT/bin:\$PYENV_ROOT/shims:\$PATH && \
   eval \"\$(pyenv init -)\" && \
   bash /workspace/gen-gec-errant/deploy/run-celva-all-models.sh; bash'"
```

### Monitor a running window
```bash
# List all windows
ssh $SSH_URL "tmux list-windows -t ssh_tmux"

# Tail live output
ssh $SSH_URL "tmux capture-pane -t ssh_tmux:celva -p -S -30"

# Attach for interactive control
ssh $SSH_URL -t "tmux attach -t ssh_tmux:celva"
# Detach with: Ctrl+b, d
```
