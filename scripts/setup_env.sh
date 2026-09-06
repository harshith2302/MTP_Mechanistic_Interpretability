#!/bin/bash
# LOGIN NODE ONLY -- this is the only script allowed to touch the network,
# besides download_models.sh. Run once.
set -euo pipefail

PROJ="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJ
export HF_HOME="$PROJ/hf_cache"
echo "[setup] PROJ=$PROJ"

python3 -m venv "$PROJ/envs/eval"
source "$PROJ/envs/eval/bin/activate"
pip install --upgrade pip
pip install vllm transformers accelerate num2words pyyaml pandas matplotlib pytest
pip freeze > "$PROJ/envs/eval-requirements.lock"

echo "[setup] eval venv ready. Locked to envs/eval-requirements.lock"
python -c "import vllm, transformers; print('vllm', vllm.__version__, '| transformers', transformers.__version__)"
echo
echo "[setup] The mech venv (nnsight / transformer_lens) is NOT built here --"
echo "[setup] its torch pin fights vLLM's. Build it separately when phase 2 starts."
