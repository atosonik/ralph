#!/usr/bin/env bash
# ============================================================================
# AutoRalph Phase 0.5 — H100 bootstrap script
#
# Run this on a fresh H100 VM from Shadeform (Lambda, Latitude, etc.).
# It does everything: clone → install → data prep → noise floor → AutoRalph-1.
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/AutoRalphBase/autoralph/main/scripts/run_h100.sh | bash
#
#   Or, if you've already cloned:
#   cd autoralph && bash scripts/run_h100.sh
#
# Expected wall-clock:
#   Data prep (~1B tokens):  ~10-20 min
#   Noise floor (10 seeds):  ~30 min  (proxy config, 500 steps each)
#   AutoRalph-1 training:   ~35 min  (default config, 2000 steps)
#   Total:                   ~60-90 min
#
# Output:
#   runs/h100_noise_floor/noise_floor_summary.json  — empirical noise floor
#   runs/h100_autoralph1/                          — AutoRalph-1 checkpoint + logs
#   runs/h100_calibration/calibration.json          — H100 reference timings
# ============================================================================

set -euo pipefail

REPO_URL="git@github-bitzic:AutoRalphBase/autoralph.git"
WORKDIR="${AUTORALPH_DIR:-$HOME/autoralph}"
DATA_TOKENS="${DATA_TOKENS:-1000000000}"        # 1B training tokens
EVAL_TOKENS="${EVAL_TOKENS:-5000000}"           # 5M eval tokens
NOISE_RUNS="${NOISE_RUNS:-10}"
AUTORALPH1_SEED="${AUTORALPH1_SEED:-1337}"
# Two-tier model (whitepaper v1.1 §5.4):
#   "verified"   = full TDX+nvtrust attestation chain (needs CC-capable H100)
#   "unverified" = no attestation, scored at α=0.5 (any H100 works)
# Default to unverified since most Shadeform rentals aren't CC-capable.
TIER="${TIER:-unverified}"

echo "=============================================="
echo "  AutoRalph Phase 0.5 — H100 Bootstrap"
echo "=============================================="

# --- Step 0: Clone if needed ---
if [ ! -d "$WORKDIR/.git" ]; then
    echo "[0/5] Cloning repo..."
    git clone "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"

# --- Step 1: Python environment ---
echo "[1/5] Setting up Python environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet torch --index-url https://download.pytorch.org/whl/cu124
pip install --quiet -e '.[data]'

# Verify GPU.
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# --- Step 2: Prepare FineWeb-Edu data ---
echo "[2/5] Preparing FineWeb-Edu data (~1B tokens)..."
if [ -f "data/data_manifest.json" ]; then
    echo "  data_manifest.json exists, skipping data prep."
    echo "  (delete data/data_manifest.json to re-prepare)"
else
    python -m data.prepare \
        --source fineweb-edu \
        --out data/shards \
        --shard-tokens 10000000 \
        --total-tokens "$DATA_TOKENS" \
        --eval-tokens "$EVAL_TOKENS"
fi

# --- Step 3: H100 calibration benchmark ---
echo "[3/5] Running calibration benchmark..."
mkdir -p runs/h100_calibration
python -m calibration.benchmark | tee runs/h100_calibration/calibration.json

# --- Step 4: Noise floor calibration ---
echo "[4/5] Running noise floor calibration ($NOISE_RUNS runs)..."
python scripts/noise_floor.py \
    --runs "$NOISE_RUNS" \
    --base-seed 5000 \
    --config configs/h100_proxy.json \
    --out-dir runs/h100_noise_floor

# --- Step 5: AutoRalph-1 training ---
echo "[5/5] Training AutoRalph-1 (300M params, ~262M tokens)..."
python -m recipe.train \
    --config configs/h100_default.json \
    --out-dir runs/h100_autoralph1 \
    --seed "$AUTORALPH1_SEED"

# --- Summary ---
echo ""
echo "=============================================="
echo "  DONE — Phase 0.5 complete"
echo "=============================================="
echo ""
echo "Calibration:   runs/h100_calibration/calibration.json"
echo "Noise floor:   runs/h100_noise_floor/noise_floor_summary.json"
echo "AutoRalph-1:  runs/h100_autoralph1/"
echo ""

if [ -f "runs/h100_noise_floor/noise_floor_summary.json" ]; then
    python -c "
import json
nf = json.load(open('runs/h100_noise_floor/noise_floor_summary.json'))
print(f\"Noise floor: mean={nf['val_bpb']['mean']:.4f}  std={nf['val_bpb']['std']:.4f}  margin(2σ)={nf['suggested_noise_floor_margin']:.4f}\")
"
fi

if [ -f "runs/h100_autoralph1/final_state.json" ]; then
    python -c "
import json
fs = json.load(open('runs/h100_autoralph1/final_state.json'))
print(f\"AutoRalph-1: final_loss={fs['final_loss']:.4f}  tokens={fs['tokens_seen']:,}  wall={fs['wall_clock_s']:.0f}s\")
"
fi

echo ""
echo "Next steps:"
echo "  1. Run the hidden eval on AutoRalph-1:"
echo "     python -c \"..."
echo "  2. Post results to GitHub Discussions"
echo "  3. Build + test the Docker container:"
echo "     docker build -t autoralph-proof:latest ."
echo "  4. (Phase 0.5c) Rent a CC-capable H100 for real TDX+nvtrust attestation"
