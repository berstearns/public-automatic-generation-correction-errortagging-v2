#!/bin/bash
#===============================================================================
# run-celva-all-models.sh — Run full CELVA-SP pipeline (24 models) on remote
#===============================================================================
# Runs ON the remote server (via tmux or SSH pipe).
# Fetches data + fine-tuned model weights from GDrive, then runs the pipeline.
#
# Usage (tmux on remote — recommended for long runs):
#   ssh $SSH_URL "tmux new-window -t ssh_tmux -n celva \
#     'export PYENV_ROOT=/root/.pyenv && \
#      export PATH=\$PYENV_ROOT/bin:\$PYENV_ROOT/shims:\$PATH && \
#      eval \"\$(pyenv init -)\" && \
#      bash /workspace/gen-gec-errant/deploy/run-celva-all-models.sh; bash'"
#
# Usage (direct on server):
#   bash run-celva-all-models.sh [--skip-fetch] [--sync-results]
#
# Prerequisites:
#   - project-setup-gen-gec already run (repo + deps installed)
#   - rclone configured with remote 'i:' (for GDrive access)
#
# Time estimate: ~2-4 hours on 24GB+ GPU (24 models, full CELVA-SP)
#===============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'

REPO_DIR="/workspace/gen-gec-errant"
DATA_DIR="/workspace/data"
FT_MODELS_DIR="/workspace/ft-models"
RESULTS_DIR="/workspace/gec-results/celva-sp"
PYENV_ROOT="/root/.pyenv"

GDRIVE_DATA="i:/<your-rclone-root>/data/splits"
GDRIVE_MODELS="i:/<your-rclone-root>/models"
GDRIVE_RESULTS="i:/<your-rclone-root>/generation-gec-errant/outputs"

SKIP_FETCH=""
SYNC_RESULTS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-fetch)    SKIP_FETCH="1"; shift ;;
        --sync-results)  SYNC_RESULTS="1"; shift ;;
        -h|--help)
            echo "Usage: $0 [--skip-fetch] [--sync-results]"
            echo "  --skip-fetch     Skip data/model download (already cached)"
            echo "  --sync-results   Sync results to GDrive after completion"
            exit 0
            ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Activate pyenv
export PYENV_ROOT
export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
eval "$(pyenv init -)" 2>/dev/null || true

echo -e "${CYAN}=== gen-gec-errant: CELVA-SP All Models ===${NC}"
echo "  Repo:    $REPO_DIR"
echo "  Data:    $DATA_DIR"
echo "  Models:  $FT_MODELS_DIR"
echo "  Output:  $RESULTS_DIR"
echo "  Python:  $(python3 --version)"
echo "  GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Start:   $(date)"
echo ""

# Validate setup
if [[ ! -d "$REPO_DIR" ]]; then
    echo -e "${RED}FATAL: Repo not found at $REPO_DIR. Run project-setup-gen-gec first.${NC}"
    exit 1
fi

# Pull latest code
echo -e "${CYAN}[0/4] Pulling latest code...${NC}"
git -C "$REPO_DIR" pull --ff-only 2>&1 | tail -3
echo "  HEAD: $(git -C "$REPO_DIR" log --oneline -1)"
echo ""

#===============================================================================
# Phase 1: Fetch data
#===============================================================================
if [[ -z "$SKIP_FETCH" ]]; then
    echo -e "${CYAN}[1/4] Fetching data from GDrive...${NC}"

    if ! command -v rclone &>/dev/null; then
        echo -e "${RED}FATAL: rclone not installed. Run 'copy-rclone' via orchestrator first.${NC}"
        exit 1
    fi

    mkdir -p "$DATA_DIR"
    if [[ -f "$DATA_DIR/norm-CELVA-SP.csv" ]]; then
        echo -e "  ${GREEN}✓${NC} norm-CELVA-SP.csv (cached)"
    else
        rclone copy "$GDRIVE_DATA/norm-CELVA-SP.csv" "$DATA_DIR/" 2>&1
        echo -e "  ${GREEN}✓${NC} norm-CELVA-SP.csv (downloaded)"
    fi
    echo ""

    #===========================================================================
    # Phase 2: Fetch fine-tuned model weights
    #===========================================================================
    echo -e "${CYAN}[2/4] Fetching fine-tuned model weights...${NC}"
    mkdir -p "$FT_MODELS_DIR"

    fetch_model() {
        local src="$1"
        local dest="$FT_MODELS_DIR/$src"
        if [[ -f "$dest/model.safetensors" || -f "$dest/pytorch_model.bin" ]]; then
            echo -e "  ${GREEN}✓${NC} $src (cached)"
        else
            mkdir -p "$dest"
            rclone copy "$GDRIVE_MODELS/$src" "$dest/" 2>&1
            echo -e "  ${GREEN}✓${NC} $src"
        fi
    }

    # GPT-2 fine-tuned
    fetch_model "gpt2/gpt2-small-all-data/best/checkpoint-7596"
    fetch_model "gpt2/gpt2-medium-all-data/best/checkpoint-5625"
    fetch_model "gpt2/gpt2-large-all-data/best/checkpoint-6750"

    # Pythia fine-tuned
    fetch_model "pythia/pythia-70m-all-data/best/checkpoint-13497"
    fetch_model "pythia/pythia-160m-all-data/best/checkpoint-15951"
    fetch_model "pythia/pythia-410m-all-data/best/checkpoint-21476"
    fetch_model "pythia/pythia-1b-all-data/best/checkpoint-21036"
    fetch_model "pythia/pythia-1.4b-all-data/best/checkpoint-12278"

    # SmolLM2 fine-tuned
    fetch_model "smollm2/smollm2-135m-all-data/best/checkpoint-16289"
    fetch_model "smollm2/smollm2-360m-all-data/best/checkpoint-16712"
    fetch_model "smollm2/smollm2-1.7b-all-data/best/checkpoint-6688"
    echo ""
else
    echo -e "${YELLOW}[1-2/4] Skipping data/model fetch (--skip-fetch)${NC}"
    echo ""
fi

#===============================================================================
# Phase 3: Run pipeline
#===============================================================================
CONFIG="$REPO_DIR/configs/pipeline/celva-all-models.yaml"

echo -e "${CYAN}[3/4] Running pipeline (24 models)...${NC}"
echo "  Config: $CONFIG"
echo "  Start:  $(date)"
echo ""

cd "$REPO_DIR"

if python3 -m gen_gec_errant.pipeline --config "$CONFIG" 2>&1 | tee "/workspace/pipeline-celva-sp.log"; then
    echo ""
    echo -e "${GREEN}Pipeline completed successfully${NC}"
else
    echo ""
    echo -e "${RED}Pipeline FAILED — check /workspace/pipeline-celva-sp.log${NC}"
    exit 1
fi

#===============================================================================
# Phase 4: Verify & optionally sync results
#===============================================================================
echo ""
echo -e "${CYAN}[4/4] Verifying results...${NC}"

if [[ -d "$RESULTS_DIR" ]]; then
    echo "  Output files:"
    ls -lh "$RESULTS_DIR/" 2>/dev/null | head -20
    echo ""

    FILE_COUNT=$(find "$RESULTS_DIR" -type f | wc -l)
    echo -e "  ${GREEN}✓${NC} $FILE_COUNT files in $RESULTS_DIR"
else
    echo -e "${RED}  ✗ Output dir not found: $RESULTS_DIR${NC}"
fi

if [[ -n "$SYNC_RESULTS" ]]; then
    echo ""
    echo "  Syncing results to GDrive..."
    rclone copy "$RESULTS_DIR/" "${GDRIVE_RESULTS}/celva-sp/" 2>&1
    echo -e "  ${GREEN}✓${NC} Synced to ${GDRIVE_RESULTS}/celva-sp/"
fi

echo ""
echo -e "${GREEN}=== All done. $(date) ===${NC}"
echo "  Results: $RESULTS_DIR"
echo "  Log:     /workspace/pipeline-celva-sp.log"
