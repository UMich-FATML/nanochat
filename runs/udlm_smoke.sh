#!/bin/bash
# Smoke test for UDLM (uniform diffusion) pretraining on the Slurm cluster.
# Runs the UDLM test suite on an H200 (sm90: exercises both FA3 and SDPA paths),
# then a tiny end-to-end training run to verify loss goes down and val nelbo logs.
#
# Submit FROM THE REPO ROOT (the job relies on SLURM_SUBMIT_DIR):
#   sbatch runs/udlm_smoke.sh
# or override the account/partition/resources:
#   sbatch --partition=main --gres=gpu:2 runs/udlm_smoke.sh
#
#SBATCH --job-name=udlm-smoke
#SBATCH --partition=lowprio
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=runs/udlm_smoke_%j.log

set -uo pipefail
# NOTE: sbatch runs a *copy* of this script from the spool, so $0 cannot locate
# the repo - use the directory the job was submitted from instead.
cd "${SLURM_SUBMIT_DIR}" || exit 1
export OMP_NUM_THREADS=1

echo "=== node: $(hostname) | date: $(date) ==="
nvidia-smi -L

# --- env ---------------------------------------------------------------------
# .venv should already contain CUDA torch (run `uv sync --extra gpu` on the login
# node first so this is a no-op); tolerate an offline node by falling back to
# whatever .venv currently has.
uv sync --extra gpu --group dev || echo "WARNING: uv sync failed (offline node?), using existing .venv"
source .venv/bin/activate

# --- tests -------------------------------------------------------------------
echo "=== pytest ==="
python -m pytest tests/test_udlm.py tests/test_attention_fallback.py -v -s
pytest_status=$?
echo "=== pytest exit status: $pytest_status ==="

# --- tiny end-to-end training run --------------------------------------------
# requires: uv run python -m nanochat.dataset -n 2  &&  python -m scripts.tok_train
echo "=== udlm_train smoke (depth=4, 20 iterations) ==="
python -m scripts.udlm_train --depth=4 --max-seq-len=512 --device-batch-size=1 \
    --total-batch-size=512 --num-iterations=20 --eval-tokens=8192
train_status=$?

echo "=== done: pytest=$pytest_status train=$train_status ==="
[ $pytest_status -eq 0 ] && [ $train_status -eq 0 ]
