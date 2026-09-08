#!/bin/bash
# Mechanistic phase, step 1: verify the minimal pairs on a GPU.
#   sbatch scripts/verify_pairs.sbatch
#SBATCH --job-name=pencil-verify-pairs
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --account=25m0834
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cccp/25m0834/MTP_Mechanistic_Interpretability/logs/%x-%j.out
#SBATCH --error=/home/cccp/25m0834/MTP_Mechanistic_Interpretability/logs/%x-%j.err

set -euo pipefail
export PROJ="${PROJ:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJ"

export HF_HOME="$PROJ/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="$PROJ/.mplcache" OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
mkdir -p logs results/patching

# The MECH venv, not eval -- nnsight/transformer_lens live here.
source "$PROJ/envs/mech/bin/activate"

echo "[verify] node=$(hostname -s) job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 'FATAL: no CUDA -- wrong venv pin')"

python -u scripts/verify_pairs.py "$@"
echo "[verify] exit=$? $(date)"
