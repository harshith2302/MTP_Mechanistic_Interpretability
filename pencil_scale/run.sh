#!/bin/bash
# The whole experiment, in order. extract.py needs a GPU; the rest is CPU.
#   MODEL=llama bash run.sh                 full run -> results/llama/
#   MODEL=llama N=500 OUT=results/pilot bash run.sh   pilot
# On the cluster:
#   sbatch --partition=a40 --qos=a40 --gres=gpu:1 --cpus-per-task=16 --mem=96G \
#          --time=06:00:00 --wrap "MODEL=llama bash run.sh"
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
