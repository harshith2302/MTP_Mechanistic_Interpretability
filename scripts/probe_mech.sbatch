#!/bin/bash
# Prove envs/mech actually works on a GPU node BEFORE any real experiment.
#   sbatch scripts/probe_mech.sbatch
#
# The failure this exists to catch: a torch built for CUDA 13 installs cleanly,
# imports cleanly, reports device_count()==1, and still returns False from
# torch.cuda.is_available() here. Nothing before check 2 would notice.
#
#SBATCH --job-name=mech-probe
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --account=25m0834
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cccp/25m0834/MTP_Mechanistic_Interpretability/logs/%x-%j.out
#SBATCH --error=/home/cccp/25m0834/MTP_Mechanistic_Interpretability/logs/%x-%j.out

set -uo pipefail
export PROJ=/home/cccp/25m0834/MTP_Mechanistic_Interpretability
cd "$PROJ"

echo "=== node $(hostname -s) | partition ${SLURM_JOB_PARTITION:-?} | $(date -u) ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

source "$PROJ/envs/mech/bin/activate"
# The GPU nodes have no internet. Load by local path only.
export HF_HOME="$PROJ/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1 DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="$PROJ/.mplcache" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python "$PROJ/scripts/probe_mech.py"
echo "=== exit=$? | $(date -u) ==="
