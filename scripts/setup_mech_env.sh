#!/bin/bash
# LOGIN NODE ONLY -- the GPU nodes have no internet. Run once.
#
# Builds envs/mech for the mechanistic phase (nnsight / transformer_lens).
# Kept separate from envs/eval on purpose: vLLM pins torch hard, and mixing the
# two produces a venv where one of them silently stops working.
set -euo pipefail

PROJ="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJ
export HF_HOME="$PROJ/hf_cache"
echo "[mech] PROJ=$PROJ"

# ---------------------------------------------------------------------------
# THE CUDA 12 PIN IS LOAD-BEARING -- the same trap as envs/eval, different
# package. Prajna's driver is 570.86.15 = CUDA 12.8 on a40/l40 (dgx is older at
# 550.144.03 = CUDA 12.4). A torch built for CUDA 13 installs cleanly, imports
# cleanly, reports device_count() == 1, and still returns False from
# torch.cuda.is_available() on this cluster. It fails silently, inside the
# allocation, after the model has loaded.
#
# `pip install nnsight transformer_lens` on its own resolves to the newest torch,
# which is a cu130 wheel. So torch is installed FIRST from the cu128 index and
# then held there by a constraints file while everything else resolves around
# it. torch 2.9.0+cu128 is the version already proven end-to-end on an L40S by
# envs/eval, which is why it is the one pinned here too -- one torch to reason
# about across both venvs.
#
# If you bump this, re-run scripts/probe_gpu.sbatch BEFORE trusting it.
# ---------------------------------------------------------------------------
TORCH_PIN="${TORCH_PIN:-2.9.0}"
CU_INDEX="https://download.pytorch.org/whl/cu128"

python3 -m venv "$PROJ/envs/mech"
source "$PROJ/envs/mech/bin/activate"
pip install --upgrade pip

echo "[mech] installing torch==${TORCH_PIN} from the CUDA 12.8 index first"
pip install "torch==${TORCH_PIN}" --index-url "$CU_INDEX"

CONSTRAINTS="$PROJ/envs/mech-constraints.txt"
echo "torch==${TORCH_PIN}" > "$CONSTRAINTS"

echo "[mech] installing nnsight / transformer_lens against the pinned torch"
# num2words is not optional: src/simulate.py needs it, and the mech phase
# regenerates the same stories to build its patching pairs.
pip install -c "$CONSTRAINTS" nnsight transformer_lens \
    num2words pyyaml pandas matplotlib pytest

pip freeze > "$PROJ/envs/mech-requirements.lock"

# Fail on the login node rather than 40 minutes into an allocation. The CUDA
# major version is checkable without a GPU present.
python - <<'PY'
import sys, torch
major = (torch.version.cuda or "0").split(".")[0]
if major != "12":
    sys.exit(f"[mech] FATAL: torch is built for CUDA {torch.version.cuda}. "
             f"Prajna's driver is 12.8; only CUDA 12 wheels can see the GPUs. "
             f"The constraints file did not hold -- do not use this venv.")
print(f"[mech] ok: torch {torch.__version__}, CUDA {torch.version.cuda}")
PY

# transformer_lens exposes no __version__; ask the metadata, not the module.
python - <<'PY'
from importlib.metadata import version
for pkg in ("torch", "nnsight", "transformer-lens", "transformers", "num2words"):
    print(f"[mech] {pkg:18s} {version(pkg)}")
PY

echo
echo "[mech] envs/mech ready, locked to envs/mech-requirements.lock"
echo "[mech] A renamed venv is a BROKEN venv -- absolute paths are baked into"
echo "[mech] bin/pip and bin/activate. Rebuild in place, never mv it."
echo "[mech] Next: sbatch scripts/probe_mech.sbatch to confirm it sees a GPU"
echo "[mech] and can load a model by local path with no network."
echo
echo "[mech] NOTE: this venv resolves a newer transformers than envs/eval"
echo "[mech] (5.x vs 4.57). Chat-template rendering was verified byte-identical"
echo "[mech] across that gap for all four models, so the mech phase can reuse the"
echo "[mech] sweep's prompts. probe_mech.sbatch re-checks it; if it ever fails,"
echo "[mech] activations would be computed on prompts the sweep never saw."
