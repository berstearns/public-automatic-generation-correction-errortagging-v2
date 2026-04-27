#!/bin/bash
#===============================================================================
# run-all-split-sentences.sh — Run all datasets with sentence-level splitting
#===============================================================================
# Runs the pipeline for CELVA-SP, EFCAMDAT-test, and KUPA-KEYS with
# split_sentences: true. Outputs go to *_split_sentences dirs to avoid
# overwriting old (text-level) results.
#
# Usage (on remote):
#   bash deploy/run-all-split-sentences.sh [--skip-fetch] [--sync-results] [--dataset DATASET]
#
# Datasets: celva, efcamdat-test, kupa-keys, all (default: all)
#===============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'

REPO_DIR="/workspace/gen-gec-errant"
DATA_DIR="/workspace/data"
FT_MODELS_DIR="/workspace/ft-models"
PYENV_ROOT="/root/.pyenv"

GDRIVE_DATA="i:/<your-rclone-root>/data/splits"
GDRIVE_MODELS="i:/<your-rclone-root>/models"
GDRIVE_RESULTS="i:/<your-rclone-root>/generation-gec-errant/outputs"

SKIP_FETCH=""
SYNC_RESULTS=""
DATASET="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-fetch)    SKIP_FETCH="1"; shift ;;
        --sync-results)  SYNC_RESULTS="1"; shift ;;
        --dataset)       DATASET="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--skip-fetch] [--sync-results] [--dataset DATASET]"
            echo "  --skip-fetch     Skip data/model download"
            echo "  --sync-results   Sync results to GDrive after completion"
            echo "  --dataset NAME   Run only: celva, efcamdat-test, kupa-keys, all (default: all)"
            exit 0
            ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Activate pyenv
export PYENV_ROOT
export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
eval "$(pyenv init -)" 2>/dev/null || true

echo -e "${CYAN}=== gen-gec-errant: Sentence-Level Splitting — All Datasets ===${NC}"
echo "  Repo:    $REPO_DIR"
echo "  Dataset: $DATASET"
echo "  Python:  $(python3 --version)"
echo "  GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Start:   $(date)"
echo ""

# Validate setup
if [[ ! -d "$REPO_DIR" ]]; then
    echo -e "${RED}FATAL: Repo not found at $REPO_DIR${NC}"
    exit 1
fi

# Pull latest code
echo -e "${CYAN}[0] Pulling latest code...${NC}"
git -C "$REPO_DIR" pull --ff-only 2>&1 | tail -3
echo "  HEAD: $(git -C "$REPO_DIR" log --oneline -1)"
echo ""

#===============================================================================
# Fetch data + models
#===============================================================================
if [[ -z "$SKIP_FETCH" ]]; then
    echo -e "${CYAN}[1] Fetching data from GDrive...${NC}"

    if ! command -v rclone &>/dev/null; then
        echo -e "${RED}FATAL: rclone not installed${NC}"
        exit 1
    fi

    mkdir -p "$DATA_DIR"

    for csv in norm-CELVA-SP.csv norm-EFCAMDAT-test.csv norm-KUPA-KEYS.csv; do
        if [[ -f "$DATA_DIR/$csv" ]]; then
            echo -e "  ${GREEN}✓${NC} $csv (cached)"
        else
            rclone copy "$GDRIVE_DATA/$csv" "$DATA_DIR/" 2>&1
            echo -e "  ${GREEN}✓${NC} $csv (downloaded)"
        fi
    done
    echo ""

    echo -e "${CYAN}[2] Fetching fine-tuned model weights...${NC}"
    mkdir -p "$FT_MODELS_DIR"

    fetch_model() {
        local src="$1"
        local dest="$FT_MODELS_DIR/$src"
        if ls "$dest"/*.safetensors "$dest"/model*.bin &>/dev/null; then
            echo -e "  ${GREEN}✓${NC} $src (cached)"
        else
            mkdir -p "$dest"
            rclone copy "$GDRIVE_MODELS/$src" "$dest/" 2>&1
            echo -e "  ${GREEN}✓${NC} $src"
        fi
    }

    # GPT-2 fine-tuned (best checkpoints)
    fetch_model "gpt2/gpt2-small-all-data/best/checkpoint-7596"
    fetch_model "gpt2/gpt2-medium-all-data/best/checkpoint-5625"
    fetch_model "gpt2/gpt2-large-all-data/best/checkpoint-6750"

    # Pythia fine-tuned (best checkpoints)
    fetch_model "pythia/pythia-70m-all-data/best/checkpoint-13497"
    fetch_model "pythia/pythia-160m-all-data/best/checkpoint-15951"
    fetch_model "pythia/pythia-410m-all-data/best/checkpoint-21476"
    fetch_model "pythia/pythia-1b-all-data/best/checkpoint-21036"
    fetch_model "pythia/pythia-1.4b-all-data/best/checkpoint-12278"

    # SmolLM2 fine-tuned (best checkpoints)
    fetch_model "smollm2/smollm2-135m-all-data/best/checkpoint-16289"
    fetch_model "smollm2/smollm2-360m-all-data/best/checkpoint-16712"
    fetch_model "smollm2/smollm2-1.7b-all-data/best/checkpoint-6688"

    # Also fetch final checkpoints (used by efcamdat-test and kupa-keys configs)
    for family in pythia/pythia-70m pythia/pythia-160m pythia/pythia-410m pythia/pythia-1b pythia/pythia-1.4b \
                  smollm2/smollm2-135m smollm2/smollm2-360m smollm2/smollm2-1.7b; do
        fetch_model "${family}-all-data/final"
    done
    echo ""
else
    echo -e "${YELLOW}[1-2] Skipping data/model fetch (--skip-fetch)${NC}"
    echo ""
fi

#===============================================================================
# Run pipelines
#===============================================================================
cd "$REPO_DIR"

PASSED=0; FAILED=0

run_pipeline() {
    local name="$1"
    local config="$2"
    local results_dir="$3"
    local log_file="/workspace/pipeline-${name}_split_sentences.log"

    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}  Running: $name (sentence-level)${NC}"
    echo -e "${CYAN}  Config:  $config${NC}"
    echo -e "${CYAN}  Output:  $results_dir${NC}"
    echo -e "${CYAN}  Start:   $(date)${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    if python3 -m gen_gec_errant.pipeline --config "$config" 2>&1 | tee "$log_file"; then
        echo -e "${GREEN}  ✓ $name completed successfully${NC}"
        PASSED=$((PASSED + 1))

        if [[ -n "$SYNC_RESULTS" ]]; then
            echo "  Syncing results to GDrive..."
            rclone copy "$results_dir/" "${GDRIVE_RESULTS}/${name}_split_sentences/" 2>&1
            echo -e "  ${GREEN}✓${NC} Synced to GDrive"
        fi
    else
        echo -e "${RED}  ✗ $name FAILED — see $log_file${NC}"
        FAILED=$((FAILED + 1))
    fi
    echo ""
}

if [[ "$DATASET" == "all" || "$DATASET" == "celva" ]]; then
    run_pipeline "celva-sp" \
        "configs/pipeline/celva-all-models-split-sentences.yaml" \
        "/workspace/gec-results/celva-sp_split_sentences"
fi

if [[ "$DATASET" == "all" || "$DATASET" == "efcamdat-test" ]]; then
    run_pipeline "efcamdat-test" \
        "configs/pipeline/efcamdat-test-all-models-split-sentences.yaml" \
        "/workspace/gec-results/efcamdat-test_split_sentences"
fi

if [[ "$DATASET" == "all" || "$DATASET" == "kupa-keys" ]]; then
    run_pipeline "kupa-keys" \
        "configs/pipeline/kupa-keys-all-models-split-sentences.yaml" \
        "/workspace/gec-results/kupa-keys_split_sentences"
fi

#===============================================================================
# Summary
#===============================================================================
echo -e "${CYAN}=============================================${NC}"
echo -e "${CYAN}  SUMMARY${NC}"
echo -e "${CYAN}=============================================${NC}"
echo "  Passed:  $PASSED"
echo "  Failed:  $FAILED"
echo "  End:     $(date)"
echo -e "${CYAN}=============================================${NC}"
