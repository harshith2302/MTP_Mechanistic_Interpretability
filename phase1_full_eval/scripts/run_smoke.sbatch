#!/bin/bash
# Phase 6 smoke run -- the gate before the full sweep.
#
#   sbatch scripts/run_smoke.sbatch                          # defaults below
#   sbatch --export=ALL,MODEL_INDEX=1 scripts/run_smoke.sbatch
#
# Does three things in one allocation:
#   1. preflight (8 checks, incl. a real vLLM load + generation)
#   2. OLMo-2 context-budget dry run -- where does the 4096 window actually
#      overflow, measured with the real tokenizer rather than a chars/token guess
#   3. the smoke generation itself: 3 N values x 3 stories = 198 prompts
#
# Slurm rules on Prajna (see CLAUDE.md): --qos must equal the partition, --gres
# must be untyped, log paths must be absolute, and `debug` is unusable. This is a
# batch job on l40 rather than an interactive srun: `interactive` caps at 4 h with
# MaxJobsPU=2, and an srun --pty dies with the ssh session and leaves no log.
#
#SBATCH --job-name=pencil-smoke
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --account=25m0834
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/home/cccp/25m0834/MTP_Mechanistic_Interpretability/logs/%x-%j.out
#SBATCH --error=/home/cccp/25m0834/MTP_Mechanistic_Interpretability/logs/%x-%j.err

set -euo pipefail

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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
mkdir -p "$MPLCONFIGDIR" logs

source "$PROJ/envs/eval/bin/activate"

MODEL_INDEX="${MODEL_INDEX:-0}"        # 0=Qwen 1=Llama 2=Mistral 3=OLMo-2
OLMO_INDEX="${OLMO_INDEX:-3}"
SMOKE_N="${SMOKE_N:-4 10 16}"
SMOKE_STORIES="${SMOKE_STORIES:-3}"

echo "[smoke] node=$(hostname -s) partition=${SLURM_JOB_PARTITION:-?} job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo
echo "===== 1. preflight ====="
bash scripts/preflight.sh || exit 1

echo
echo "===== 2. OLMo-2 context budget (dry run, no generation) ====="
python -m src.run_eval --model-index "$OLMO_INDEX" \
  --n-values 12 16 18 20 22 --stories-per-n 2 --dry-run

echo
echo "===== 3. smoke generation (model-index=$MODEL_INDEX) ====="
python -m src.run_eval --model-index "$MODEL_INDEX" \
  --n-values $SMOKE_N --stories-per-n "$SMOKE_STORIES" --tag smoke

echo
echo "===== 4. what landed ====="
find results/raw -name '*smoke*' -type d | while read -r d; do
  echo "  $d"
  for f in "$d"/*/*.jsonl; do
    [ -e "$f" ] || continue
    echo "    $(basename "$(dirname "$f")")/$(basename "$f"): $(wc -l < "$f") records"
  done
done

echo
echo "===== 5. the numbers to actually read ====="
python - <<'PY'
import glob, json, collections
# N*.jsonl only -- prompts.jsonl is the prompt store, not results. Globbing it in
# doubles every denominator with phantom records that have no parse_status.
files = sorted(glob.glob("results/raw/*smoke*/**/N*.jsonl", recursive=True))
recs = [json.loads(l) for f in files for l in open(f, encoding="utf-8") if l.strip()]
if not recs:
    print("  no smoke records found"); raise SystemExit
byN = collections.defaultdict(lambda: {"tot": 0, "corr": 0, "ov": 0,
                                       "ps": collections.Counter()})
for r in recs:
    d = byN[r["n_people"]]
    if r.get("context_overflow"):
        d["ov"] += 1; continue
    d["tot"] += 1
    d["corr"] += bool(r.get("correct"))
    d["ps"][r["parse_status"]] += 1
print(f"  {'N':>3} {'graded':>7} {'correct':>8} {'acc':>7} {'overflow':>9}   parse_status")
for n in sorted(byN):
    d = byN[n]
    ps = " ".join(f"{k}={v}" for k, v in d["ps"].most_common())
    print(f"  {n:>3} {d['tot']:>7} {d['corr']:>8} {d['corr']/max(d['tot'],1):>6.1%} "
          f"{d['ov']:>9}   {ps}")
t = sum(d["tot"] for d in byN.values()); c = sum(d["corr"] for d in byN.values())
print(f"\n  OVERALL {c}/{t} = {c/max(t,1):.1%}  "
      f"(context_overflow excluded, never counted wrong)")

print("\n  coarse_category on wrong answers:")
for k, v in collections.Counter(r["coarse_category"] for r in recs
                                if not r.get("correct")).most_common():
    print(f"     {k:18s} {v:4d}")

print("\n  accuracy by question_type:")
bt = collections.defaultdict(lambda: [0, 0])
for r in recs:
    if r.get("context_overflow"): continue
    bt[r["question_type"]][0] += bool(r.get("correct")); bt[r["question_type"]][1] += 1
for k, (c2, t2) in sorted(bt.items(), key=lambda x: x[1][0] / max(x[1][1], 1)):
    print(f"     {k:28s} {c2:3d}/{t2:3d} = {c2/max(t2,1):5.1%}")
PY

echo "[smoke] done $(date). Report these records to Harshith before the full sweep."
