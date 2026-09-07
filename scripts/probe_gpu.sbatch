#!/bin/bash
# Throwaway diagnostic: run the full preflight on a GPU node against a chosen venv.
#   sbatch -p l40 -q l40 --export=ALL,VENV=envs/eval-cu12 scripts/probe_gpu.sbatch
#SBATCH --job-name=pencil-probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=/home/cccp/25m0834/MTP_Mechanistic_Interpretability/logs/probe_%x_%j.out
#SBATCH --error=/home/cccp/25m0834/MTP_Mechanistic_Interpretability/logs/probe_%x_%j.out

set -uo pipefail
export PROJ=/home/cccp/25m0834/MTP_Mechanistic_Interpretability
cd "$PROJ"
VENV="${VENV:-envs/eval-cu12}"

echo "=== node $(hostname -s) | partition ${SLURM_JOB_PARTITION:-?} | venv $VENV | $(date) ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

source "$PROJ/$VENV/bin/activate"
export HF_HOME="$PROJ/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="$PROJ/.mplcache"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

bash scripts/preflight.sh
echo "=== preflight exit=$? | $(date) ==="
