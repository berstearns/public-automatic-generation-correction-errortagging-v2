#!/bin/bash
#===============================================================================
# run-smoke-test.sh — Run smoke test on remote after setup
#===============================================================================
# Runs ON the remote via SSH pipe (orchestrator handles the connection).
# Expects project-setup-remote.sh to have been run first.
#
# Usage (via orchestrator):
#   ./orchestrator.sh --mode ssh --ssh-url URL run-gen-gec-smoke
#
# Usage (direct SSH pipe):
#   ssh -p PORT root@HOST "bash -s" < deploy/run-smoke-test.sh
#
# Usage (direct on server):
#   bash run-smoke-test.sh [--device auto|cpu|cuda] [--repo-dir PATH]
#===============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'

REPO_DIR="/workspace/gen-gec-errant"
DEVICE="auto"
PYENV_ROOT="/root/.pyenv"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)   DEVICE="$2"; shift 2 ;;
        --repo-dir) REPO_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--device auto|cpu|cuda] [--repo-dir PATH]"
            exit 0
            ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Activate pyenv
export PYENV_ROOT
export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
eval "$(pyenv init -)" 2>/dev/null || true

CONFIG="$REPO_DIR/configs/pipeline/smoke-test.yaml"
OUTPUT_DIR="$REPO_DIR/outputs/smoke-test-dummy"

echo -e "${CYAN}=== gen-gec-errant: Smoke Test ===${NC}"
echo "  Repo:    $REPO_DIR"
echo "  Config:  $CONFIG"
echo "  Output:  $OUTPUT_DIR"
echo "  Device:  $DEVICE"
echo "  Python:  $(python3 --version)"
echo "  GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# Validate setup
if [[ ! -d "$REPO_DIR" ]]; then
    echo -e "${RED}FATAL: Repo not found at $REPO_DIR. Run project-setup-gen-gec first.${NC}"
    exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
    echo -e "${RED}FATAL: Smoke config not found at $CONFIG${NC}"
    exit 1
fi

# Clean previous smoke output
rm -rf "$OUTPUT_DIR"

# Run pipeline
echo -e "${CYAN}[1/2] Running smoke test pipeline...${NC}"
cd "$REPO_DIR"

if python3 -m gen_gec_errant.pipeline --config "$CONFIG" --device "$DEVICE" 2>&1; then
    echo ""
    echo -e "${GREEN}Pipeline completed${NC}"
else
    echo ""
    echo -e "${RED}Pipeline FAILED${NC}"
    exit 1
fi

# Verify outputs
echo ""
echo -e "${CYAN}[2/2] Verifying outputs...${NC}"

PASS=true
CHECKS_PASS=0
CHECKS_FAIL=0

check() {
    local label="$1" ok="$2" detail="${3:-}"
    if [[ "$ok" == "true" ]]; then
        echo -e "  ${GREEN}[PASS]${NC} $label${detail:+ — $detail}"
        CHECKS_PASS=$((CHECKS_PASS + 1))
    else
        echo -e "  ${RED}[FAIL]${NC} $label${detail:+ — $detail}"
        CHECKS_FAIL=$((CHECKS_FAIL + 1))
        PASS=false
    fi
}

check "Output dir exists: $OUTPUT_DIR" \
    "$([[ -d "$OUTPUT_DIR" ]] && echo true || echo false)"

# Check for expected output files
for f in raw_results.json prompts.json full_results.tsv errors_long_format.tsv; do
    check "$f exists" \
        "$([[ -f "$OUTPUT_DIR/$f" ]] && echo true || echo false)"
done

# Check raw_results.json has gpt2-base data
if [[ -f "$OUTPUT_DIR/raw_results.json" ]]; then
    has_data=$(python3 -c "
import json
with open('$OUTPUT_DIR/raw_results.json') as f:
    data = json.load(f)
has = 'gpt2-base' in data.get('model_results', data) if isinstance(data, dict) else False
print('true' if has else 'false')
" 2>/dev/null || echo "false")
    check "raw_results.json contains gpt2-base data" "$has_data"
fi

echo ""
echo "========================================="
echo "  Checks passed: $CHECKS_PASS"
echo "  Checks failed: $CHECKS_FAIL"
echo "========================================="

if [[ "$PASS" == "true" ]]; then
    echo -e "${GREEN}ALL CHECKS PASSED — smoke test OK${NC}"
    echo ""
    echo "  Output: $OUTPUT_DIR"
    ls -la "$OUTPUT_DIR/" 2>/dev/null
else
    echo -e "${RED}SOME CHECKS FAILED${NC}"
    exit 1
fi
