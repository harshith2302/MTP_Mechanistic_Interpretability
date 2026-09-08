#!/bin/bash
# One model, one job.  sbatch scripts/run_model.sbatch qwen
#
# Cluster rules that reject a job outright if broken: --qos must equal the
# partition name, --gres must be untyped, and the `debug` partition is unusable.
# l40 has 32 CPUs across 8 GPUs, so 4 CPUs is the per-GPU share.
#
# Log paths are relative, so they resolve against the directory you submit from.
# Submit from the repo root. Slurm opens these files BEFORE this script runs, so
# `logs/` must already exist -- it is gitignored and absent from a fresh clone:
#   mkdir -p logs && sbatch scripts/run_model.sbatch qwen
#SBATCH --job-name=pencil-taxonomy
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --account=25m0834
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
export PROJ="${PROJ:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJ"

# Compute nodes are air-gapped. Models load by local path, so nothing here needs
# the network -- these flags make a stray lookup fail fast instead of hanging.
export HF_HOME="$PROJ/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false MPLCONFIGDIR="$PROJ/.mplcache"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
mkdir -p logs "$MPLCONFIGDIR"

source "$PROJ/envs/eval/bin/activate"

MODEL="${1:?usage: sbatch scripts/run_model.sbatch <qwen|llama|mistral|olmo>}"
RUN_ID="${RUN_ID:-$(date -u +%Y-%m-%dT%H%MZ)}"
echo "[sbatch] model=$MODEL run_id=$RUN_ID node=$(hostname -s)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 'FATAL: no CUDA -- wrong venv pin')"

python -u -m src.run_model "$MODEL" --run-id "$RUN_ID"
echo "[sbatch] exit=$? $(date)"
