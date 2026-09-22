#!/bin/bash
# The whole experiment, in order. extract.py needs a GPU; everything else is CPU.
#   MODEL=llama bash run.sh              full run -> results/llama/
#   MODEL=qwen N=500 bash run.sh         pilot for another model
# On the cluster:  sbatch --gres=gpu:1 --cpus-per-task=16 --mem=96G --time=04:00:00 --wrap "bash run.sh"
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-$(readlink -f ../phase1_local/envs/mech)/bin/python}"
MODEL="${MODEL:-$(grep -m1 "^model:" config.yaml | awk '{print $2}')}"
OUT="${OUT:-results/$MODEL}"
export MODEL
export PYTHONDONTWRITEBYTECODE=1 TRANSFORMERS_VERBOSITY=error MPLCONFIGDIR="$OUT/.mpl"

$PY generate.py --model "$MODEL" --out "$OUT" ${N:+--n "$N"}
DATA_DIR="$OUT" $PY -m pytest -q -p no:cacheprovider test_data.py
$PY extract.py --model "$MODEL" --out "$OUT"
$PY probe.py --out "$OUT"
$PY plots.py --out "$OUT"
