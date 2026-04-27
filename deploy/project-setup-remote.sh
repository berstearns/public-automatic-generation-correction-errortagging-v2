#!/bin/bash
#===============================================================================
# project-setup-remote.sh — Setup gen-gec-errant on a remote GPU server
#===============================================================================
# Runs ON the remote via SSH pipe (orchestrator handles the connection).
# Installs pyenv + Python 3.10.18, clones repo from GitHub, installs deps,
# verifies everything works.
#
# Usage (via orchestrator):
#   ./orchestrator.sh --mode ssh --ssh-url URL project-setup-gen-gec
#
# Usage (direct SSH pipe):
#   ssh -p PORT root@HOST "bash -s" < deploy/project-setup-remote.sh
#
# Usage (direct on server):
#   bash project-setup-remote.sh [--force] [--repo-dir PATH]
#
# Code is deployed via git clone from GitHub (public repo, no token needed).
#===============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'

FORCE=""
GITHUB_REPO="https://github.com/berstearns/public-automatic-generation-correction-errortagging-v2.git"
REPO_DIR="/workspace/gen-gec-errant"
PYENV_ROOT="/root/.pyenv"
PYTHON_VERSION="3.10.18"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)    FORCE="1"; shift ;;
        --repo-dir) REPO_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--force] [--repo-dir PATH]"
            exit 0
            ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

phase() {
    echo ""
    echo -e "${CYAN}[$1/6] $2${NC}"
    echo "---"
}

echo -e "${CYAN}=== gen-gec-errant: Remote Project Setup ===${NC}"
echo "  Repo:    $GITHUB_REPO"
echo "  Dir:     $REPO_DIR"
echo "  Python:  $PYTHON_VERSION (via pyenv)"
echo "  GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  RAM:     $(free -h | awk '/Mem:/{print $2}')"
echo ""

# ============================================================
phase 1 "System packages"
# ============================================================
apt-get update -qq > /dev/null 2>&1
apt-get install -y -qq \
    git build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev \
    wget curl > /dev/null 2>&1
echo -e "${GREEN}System packages installed${NC}"

# ============================================================
phase 2 "Install Python $PYTHON_VERSION via pyenv"
# ============================================================
export PYENV_ROOT
export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
eval "$(pyenv init -)" 2>/dev/null || true

if ! command -v pyenv &>/dev/null; then
    echo "pyenv not found, installing..."
    git clone --depth 1 https://github.com/pyenv/pyenv.git "$PYENV_ROOT"
    export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
    eval "$(pyenv init -)"
fi

if pyenv versions --bare | grep -q "^${PYTHON_VERSION}$"; then
    echo "Python $PYTHON_VERSION already installed"
else
    echo "Installing Python $PYTHON_VERSION (this takes a few minutes)..."
    pyenv install "$PYTHON_VERSION"
fi

pyenv global "$PYTHON_VERSION"
echo -e "${GREEN}Python $(python3 --version) set as default via pyenv${NC}"

# ============================================================
phase 3 "Clone / update repo from GitHub"
# ============================================================
if [[ -d "$REPO_DIR/.git" ]]; then
    echo "Updating existing repo..."
    git -C "$REPO_DIR" fetch --all 2>&1 | tail -2
    git -C "$REPO_DIR" reset --hard origin/main 2>&1 | tail -2
    echo -e "${GREEN}Repo updated to latest main${NC}"
else
    echo "Cloning from $GITHUB_REPO..."
    rm -rf "$REPO_DIR"
    git clone "$GITHUB_REPO" "$REPO_DIR" 2>&1 | tail -3
    echo -e "${GREEN}Repo cloned${NC}"
fi

echo "  HEAD: $(git -C "$REPO_DIR" log --oneline -1)"

# ============================================================
phase 4 "Python dependencies"
# ============================================================
echo "Upgrading pip..."
pip install --upgrade pip 2>&1 | tail -1

echo "Installing package in editable mode (with all deps)..."
pip install -e "$REPO_DIR" 2>&1 | tail -5

echo "Downloading spaCy model..."
python3 -m spacy download en_core_web_sm 2>&1 | tail -2

echo -e "${GREEN}Python deps installed${NC}"

# ============================================================
phase 5 "Verify imports"
# ============================================================
python3 -c "
import torch
print(f'  torch {torch.__version__}  (CUDA: {torch.cuda.is_available()})')
import transformers
print(f'  transformers {transformers.__version__}')
import errant
print(f'  errant OK')
import spacy
nlp = spacy.load('en_core_web_sm')
print(f'  spacy {spacy.__version__} + en_core_web_sm')
import numpy, scipy, matplotlib, pandas, yaml
print(f'  numpy {numpy.__version__}, scipy {scipy.__version__}, matplotlib {matplotlib.__version__}, pandas {pandas.__version__}')
"

# Verify pipeline package is importable
python3 -c "
from gen_gec_errant.pipeline import run_pipeline
from gen_gec_errant.pipeline.config import PipelineConfig, load_config_from_yaml
print('  gen_gec_errant.pipeline OK')
from gen_gec_errant.data_loader import run_data_loader
print('  gen_gec_errant.data_loader OK')
from gen_gec_errant.generation import run_generation
print('  gen_gec_errant.generation OK')
from gen_gec_errant.gec import run_gec
print('  gen_gec_errant.gec OK')
from gen_gec_errant.annotation import run_annotation
print('  gen_gec_errant.annotation OK')
from gen_gec_errant.analysis import run_analysis
print('  gen_gec_errant.analysis OK')
"
echo -e "${GREEN}All imports verified${NC}"

# ============================================================
phase 6 "Summary"
# ============================================================
echo ""
echo -e "${GREEN}=== Setup Complete ===${NC}"
echo "  Repo:     $REPO_DIR"
echo "  Python:   $(python3 --version)"
echo "  pyenv:    $PYENV_ROOT"
echo "  GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""
echo "Next steps:"
echo "  Run smoke test:  python3 -m gen_gec_errant.pipeline --config configs/pipeline/smoke-test.yaml"
echo "  Or via orchestrator: ./orchestrator.sh ... run-gen-gec-smoke"
