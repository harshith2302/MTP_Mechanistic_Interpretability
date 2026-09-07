#!/bin/bash
# LOGIN NODE ONLY -- this is the only script allowed to touch the network,
# besides download_models.sh. Run once.
set -euo pipefail

PROJ="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJ
export HF_HOME="$PROJ/hf_cache"
echo "[setup] PROJ=$PROJ"

# ---------------------------------------------------------------------------
# THE PIN IS LOAD-BEARING. Do not relax it to a bare `pip install vllm`.
#
# Prajna's GPU nodes run driver 570.86.15 = CUDA 12.8 (a40 and l40; dgx is older
# still at 550.144.03 = CUDA 12.4). A torch built for CUDA 13 cannot use them:
# the job gets a GPU, nvidia-smi looks healthy, device_count() reports 1, and
# torch.cuda.is_available() is still False. Measured 2026-09-07 on cn23-a40 and
# cn43-l40 with torch 2.13.0+cu130, which is what a bare `pip install vllm`
# resolves to today.
#
# The constraint is a genuine squeeze, so the pin has exactly one solution:
#   * torch >= 2.11.0  -> CUDA 13 wheels  -> unusable here
#   * vllm  >= 0.21.0  -> requires torch >= 2.11 -> unusable here
#   * vllm  0.12.0-0.20.2 -> manylinux_2_31/2_35 wheels; this is RHEL 8 with
#                            glibc 2.28, so pip falls back to an sdist build and
#                            dies on `CUDA_HOME is not set`
#   * vllm  0.11.2 -> torch 2.9.0+cu128, manylinux1 wheel -> INSTALLS AND RUNS
#
# vllm 0.11.2 is therefore the newest release that works on this cluster.
# Verified end to end on an L40S: full preflight green, OLMo-2 loaded and
# generated. If you bump this, re-run scripts/probe_gpu.sbatch first.
# ---------------------------------------------------------------------------
VLLM_PIN="${VLLM_PIN:-0.11.2}"

python3 -m venv "$PROJ/envs/eval"
source "$PROJ/envs/eval/bin/activate"
pip install --upgrade pip
pip install "vllm==${VLLM_PIN}" transformers accelerate num2words pyyaml pandas matplotlib pytest
pip freeze > "$PROJ/envs/eval-requirements.lock"

echo "[setup] eval venv ready. Locked to envs/eval-requirements.lock"
python -c "import vllm, transformers, torch; print('vllm', vllm.__version__, '| transformers', transformers.__version__, '| torch', torch.__version__)"

# Fail here rather than 40 minutes into an allocation. The CUDA major version is
# checkable on the login node even though there is no GPU on it.
python - <<'PY'
import sys, torch
major = (torch.version.cuda or "0").split(".")[0]
if major != "12":
    sys.exit(f"[setup] FATAL: torch is built for CUDA {torch.version.cuda}. "
             f"Prajna's driver is 12.8; only CUDA 12 wheels can see the GPUs.")
print(f"[setup] ok: torch CUDA {torch.version.cuda} matches the 12.8 driver.")
PY

echo
echo "[setup] Note: a renamed venv is a BROKEN venv -- absolute paths are baked"
echo "[setup] into bin/pip, bin/activate and every console script. Rebuild in"
echo "[setup] place rather than mv-ing envs/eval anywhere."
echo
echo "[setup] The mech venv (nnsight / transformer_lens) is NOT built here --"
echo "[setup] its torch pin fights vLLM's. Build it separately when phase 2 starts."
