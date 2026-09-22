#!/bin/bash
# Pre-sweep gate on a GPU node: corners + centre of the grid with one model.
#SBATCH --job-name=pencil-simple-smoke
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --account=25m0834
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
export PROJ="${PROJ:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJ"
export HF_HOME="$PROJ/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false MPLCONFIGDIR="$PROJ/.mplcache" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
mkdir -p "$MPLCONFIGDIR"
# envs/eval is a symlink to ../pencil_taxonomy/envs/eval. Resolve the DIRECTORY
# (not the python binary -- that symlink chain ends at the base interpreter) and
# call its python by absolute path, so nothing depends on activate's hardcoded
# VIRTUAL_ENV being right. Then prove vLLM is importable BEFORE any work: a
# stale venv path otherwise fails only after prompt building, five minutes in.
VENV="$(readlink -f "$PROJ/envs/eval")"
export PATH="$VENV/bin:$PATH" VIRTUAL_ENV="$VENV"
echo "[sbatch] python=$(command -v python)"
python -c "import vllm, sys; print('[sbatch] vllm', vllm.__version__)" \
  || { echo "[sbatch] FATAL: vllm not importable from $VENV"; exit 1; }
python -u -m src.smoke --model "${MODEL:-qwen}"
