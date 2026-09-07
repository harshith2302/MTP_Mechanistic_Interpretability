#!/bin/bash
# number_word_probability = 0.0 ablation (digits only, no "seventeen") on
# N in {6,10,14,18,22}. Small: run AFTER the main sweep, never instead of it.
#
#SBATCH --job-name=pencil-ablation
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --account=25m0834
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/cccp/25m0834/MTP_Mechanistic_Interpretability/logs/%x-%A_%a.out
#SBATCH --error=/home/cccp/25m0834/MTP_Mechanistic_Interpretability/logs/%x-%A_%a.err
#SBATCH --array=0-3

set -euo pipefail
export PROJ="${PROJ:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJ"

export HF_HOME="$PROJ/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false MPLCONFIGDIR="$PROJ/.mplcache"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
mkdir -p "$MPLCONFIGDIR" logs
source "$PROJ/envs/eval/bin/activate"

RUN_ID="${RUN_ID:-$(date -u +%Y-%m-%dT%H%MZ)_${SLURM_ARRAY_JOB_ID}}"
python -m src.run_eval \
  --model-index "$SLURM_ARRAY_TASK_ID" \
  --config configs/experiment.yaml \
  --n-values 6 10 14 18 22 \
  --number-word-probability 0.0 \
  --run-id "$RUN_ID" --tag ablation_digits
