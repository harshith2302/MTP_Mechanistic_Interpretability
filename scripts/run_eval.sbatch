#!/bin/bash
# Behavioral sweep: one array task per model, one vLLM engine per task.
#
# Slurm values below were read off Prajna (sinfo / scontrol / sacctmgr), not
# guessed -- but confirm the account and walltime before a long run:
#   partition l40  : 6 nodes, gpu:8/node, 2-day limit, DEFAULT partition
#   partition a40  : 19 nodes, gpu:4/node, 4-day limit
#   partition dgx  : 9 nodes, gpu:8/node, 6-day limit
#   QOS names mirror partition names; accounts available: 25m0834, cccp
#   there is NO module system on this cluster -- do not add `module load`.
#
#SBATCH --job-name=pencil-eval
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --account=25m0834
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --array=0-3

set -euo pipefail

# Submit from the repo root, or export PROJ yourself.
export PROJ="${PROJ:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJ"

# --- offline: the GPU nodes have no internet -------------------------------
export HF_HOME="$PROJ/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="$PROJ/.mplcache"
mkdir -p "$MPLCONFIGDIR" logs

source "$PROJ/envs/eval/bin/activate"

bash scripts/preflight.sh || exit 1

# One run_id for the whole array so all four models land in one directory.
# Pass RUN_ID=... to resume an interrupted sweep into the same place.
RUN_ID="${RUN_ID:-$(date -u +%Y-%m-%dT%H%MZ)_${SLURM_ARRAY_JOB_ID}}"
echo "[sbatch] task=$SLURM_ARRAY_TASK_ID run_id=$RUN_ID"

python -m src.run_eval \
  --model-index "$SLURM_ARRAY_TASK_ID" \
  --config configs/experiment.yaml \
  --run-id "$RUN_ID"
