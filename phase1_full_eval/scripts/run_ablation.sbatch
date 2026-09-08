#!/bin/bash
# number_word_probability = 0.0 ablation: all numerals, no "seventeen".
# N in {6,10,14,18,22} only -- run AFTER the main sweep, never instead of it.
# Derived from run_eval.sbatch so it inherits every cluster fix below;
# only the job name, walltime, N values and the two ablation flags differ.
#
# Behavioral sweep: one array task per model, one vLLM engine per task.
#
# Slurm values below were read off Prajna (sinfo / scontrol / sacctmgr), not
# guessed. Cross-checked 2026-09-07 against ~/cs725_project/cs725_Project/CLUSTER.md,
# a survey of this same cluster:
#   partition l40  : 6 nodes, gpu:8/node, 32 CPU/node (= 4 CPU per GPU), 2-day, DEFAULT
#   partition a40  : 19 nodes, gpu:4/node, 64 CPU/node (= 16 CPU per GPU), 4-day
#   partition dgx  : 9 nodes, gpu:8/node, A100-80GB, 6-day
#   `debug` is UNUSABLE -- every submit fails AssocGrpSubmitJobsLimit.
#   QOS must equal the partition name or the job is rejected outright.
#   gres must be UNTYPED (`gpu:1`); `gpu:l40s:1` is not configured and fails.
#   srun is allowed ONLY on the `interactive` partition; use sbatch everywhere else.
#   There is NO module system on this cluster -- do not add `module load`.
#
# Live QOS limits (sacctmgr, 2026-09-07): l40 MaxJobsPU=4, MaxSubmitPU=5.
# MaxSubmitPU counts PENDING ARRAY ELEMENTS, so --array must stay <= 5 on l40
# (<= 6 on a40). Our 4 tasks fit; do not widen the array without re-checking.
# The account also shares a lab-wide 20-job submit cap, so a full queue elsewhere
# in `cccp` can still block submission.
#
# cpus-per-task=4 matches l40's per-GPU share exactly (32 CPU / 8 GPU). Asking
# for 8 would let only 4 of the node's 8 GPUs be used and delays backfill for no
# gain -- vLLM's CPU work here is tokenisation, not a dataloader.
#
#SBATCH --job-name=pencil-ablation
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --account=25m0834
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/cccp/25m0834/MTP_Mechanistic_Interpretability/logs/%x-%A_%a.out
#SBATCH --error=/home/cccp/25m0834/MTP_Mechanistic_Interpretability/logs/%x-%A_%a.err
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
# Several array tasks share one repo on Lustre; concurrent .pyc writes
# from different nodes raced and broke imports in job 305111.
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PROJ/.mplcache"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
mkdir -p "$MPLCONFIGDIR" logs

echo "[sbatch] node=$(hostname) partition=${SLURM_JOB_PARTITION:-?}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

source "$PROJ/envs/eval/bin/activate"

bash scripts/preflight.sh || exit 1

# One run_id for the whole array so all four models land in one directory.
# Pass RUN_ID=... to resume an interrupted sweep into the same place.
RUN_ID="${RUN_ID:-$(date -u +%Y-%m-%dT%H%MZ)_${SLURM_ARRAY_JOB_ID}}"
echo "[sbatch] task=$SLURM_ARRAY_TASK_ID run_id=$RUN_ID"

python -m src.run_eval \
  --model-index "$SLURM_ARRAY_TASK_ID" \
  --config configs/experiment.yaml \
  --n-values 6 10 14 18 22 \
  --number-word-probability 0.0 \
  --run-id "$RUN_ID" --tag ablation_digits
