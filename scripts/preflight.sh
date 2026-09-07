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
# Prajna login nodes have no GPU at all; only compute nodes do, and srun is
# restricted to the `interactive` partition. Running preflight on the login node
# is legitimate (checks 1,2,4,5,6 are the CPU-side gate) -- it must not report a
# false failure, but it must also refuse to claim the suite passed.
ON_LOGIN=0
case "$(hostname -s)" in login*) ON_LOGIN=1 ;; esac

if [ "$ON_LOGIN" = "1" ]; then
  echo "  [skip] login node -- no GPU here. CPU-side checks only."
  echo "         For the full gate, submit it as a batch job:"
  echo "         sbatch -p l40 -q l40 --export=ALL,VENV=envs/eval \\"
  echo "                scripts/probe_gpu.sbatch"
elif command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,driver_version,memory.free --format=csv,noheader | sed 's/^/  /'
  N=$(nvidia-smi --list-gpus | wc -l)
  [ "$N" -ge 1 ] && ok "$N GPU(s) visible" || bad "no GPU visible"
else
  bad "nvidia-smi not found -- are you on a compute node?"
fi

echo "-- 3b. torch can actually USE the gpu --"
# THE check. Prajna drivers are 570.86.15 = CUDA 12.8. A torch built for CUDA 13
# (any cu130 wheel, which is what plain `pip install vllm` now resolves to) gets
# a GPU allocated, sees device_count()==1, and still returns
# is_available()==False. nvidia-smi looks perfectly healthy throughout, so check
# 3 alone cannot catch it. Verified 2026-09-07 on cn23-a40 and cn43-l40.
if [ "$ON_LOGIN" = "1" ]; then
  python3 - <<'PY'
import torch
cu = torch.version.cuda or "?"
major = cu.split(".")[0]
print(f"  [info] torch {torch.__version__} built for CUDA {cu}")
if major >= "13":
    print(f"  [FAIL] cu{major}x wheel: the GPU nodes run driver 12.8 and this "
          f"CANNOT see them.")
    raise SystemExit(1)
print("  [ ok ] CUDA major 12 -- compatible with the 12.8 driver "
      "(verify on a GPU node)")
PY
  [ $? -eq 0 ] || bad "torch CUDA major version is incompatible with this cluster"
else
  python3 - <<'PY'
import sys, torch
print(f"  [info] torch {torch.__version__} built for CUDA {torch.version.cuda}")
if not torch.cuda.is_available():
    sys.exit(f"  [FAIL] torch.cuda.is_available() is False with "
             f"{torch.cuda.device_count()} device(s) allocated -- wrong CUDA build.")
d = torch.cuda.get_device_name(0)
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
_ = x @ x
torch.cuda.synchronize()
print(f"  [ ok ] {d} {torch.cuda.get_device_capability(0)} -- bf16 matmul ran")
PY
  [ $? -eq 0 ] || bad "torch cannot use the allocated GPU"
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
if [ "$ON_LOGIN" = "1" ]; then
  echo "  [skip] login node -- needs a GPU"
elif [ "$FAIL" = "0" ]; then
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
if [ "$FAIL" != "0" ]; then
  echo "== preflight FAILED -- do not submit =="; exit 1
elif [ "$ON_LOGIN" = "1" ]; then
  echo "== preflight PASSED (CPU-side only) =="
  echo "   NOT cleared to submit: the GPU checks were skipped. Re-run inside an"
  echo "   allocation before the smoke run or the sweep."
  exit 0
else
  echo "== preflight PASSED -- cleared to submit =="; exit 0
fi
