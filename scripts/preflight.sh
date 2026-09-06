#!/bin/bash
# COMPUTE NODE. Fails loudly now rather than 40 minutes into an allocation.
set -uo pipefail

PROJ="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJ
FAIL=0
ok()   { echo "  [ ok ] $1"; }
bad()  { echo "  [FAIL] $1"; FAIL=1; }

echo "== preflight =="

echo "-- 1. model paths --"
python3 - <<'PY' || FAIL=1
import os, sys, yaml
root = os.environ["PROJ"]
cfg = yaml.safe_load(open(os.path.join(root, "configs/models.yaml")))
bad = 0
for m in cfg["models"]:
    p = os.path.join(root, m["local_path"])
    cj = os.path.join(p, "config.json")
    weights = []
    if os.path.isdir(p):
        weights = [f for f in os.listdir(p) if f.endswith((".safetensors", ".bin"))]
    if os.path.isfile(cj) and weights:
        print(f"  [ ok ] {m['short_name']}: {len(weights)} weight file(s)")
    else:
        print(f"  [FAIL] {m['short_name']}: missing config.json or weights at {p}")
        bad = 1
    if not m.get("revision"):
        print(f"  [warn] {m['short_name']}: revision is null "
              f"(run scripts/record_revisions.py on the login node)")
sys.exit(bad)
PY

echo "-- 2. imports --"
if python3 -c "import vllm, transformers, num2words, pandas, matplotlib, yaml" 2>/dev/null; then
  ok "vllm transformers num2words pandas matplotlib yaml"
else
  bad "import check -- is envs/eval activated?"
  python3 -c "import vllm, transformers, num2words, pandas, matplotlib, yaml" 2>&1 | tail -3
fi

echo "-- 3. gpu --"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader | sed 's/^/  /'
  N=$(nvidia-smi --list-gpus | wc -l)
  [ "$N" -ge 1 ] && ok "$N GPU(s) visible" || bad "no GPU visible"
else
  bad "nvidia-smi not found -- are you on a compute node?"
fi

echo "-- 4. offline flags --"
for v in HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; do
  [ "${!v:-0}" = "1" ] && ok "$v=1" || bad "$v is not 1 -- a runtime download could be attempted"
done

echo "-- 5. results writable --"
mkdir -p "$PROJ/results/raw" 2>/dev/null
if touch "$PROJ/results/.wtest" 2>/dev/null; then rm -f "$PROJ/results/.wtest"; ok "results/ writable"
else bad "results/ not writable"; fi

echo "-- 6. regression tests (the generator gate) --"
if python3 -m pytest "$PROJ/tests" -q >/dev/null 2>&1; then ok "test suite green"
else bad "test suite FAILING -- do not run a sweep on a broken generator"; fi

echo "-- 7. one real 5-token generation --"
if [ "$FAIL" = "0" ]; then
  python3 - <<'PY' || FAIL=1
import os, yaml
from vllm import LLM, SamplingParams
root = os.environ["PROJ"]
cfg = yaml.safe_load(open(os.path.join(root, "configs/models.yaml")))
m = min(cfg["models"], key=lambda x: x["max_context"])
p = os.path.join(root, m["local_path"])
llm = LLM(model=p, tokenizer=p, dtype="bfloat16", max_model_len=512,
          gpu_memory_utilization=0.60, enforce_eager=True, seed=0)
out = llm.generate(["Hello"], SamplingParams(temperature=0.0, max_tokens=5))
print(f"  [ ok ] {m['short_name']} generated: {out[0].outputs[0].text!r}")
PY
else
  echo "  [skip] earlier checks failed"
fi

echo
if [ "$FAIL" = "0" ]; then echo "== preflight PASSED =="; exit 0
else echo "== preflight FAILED -- do not submit =="; exit 1; fi
