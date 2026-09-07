#!/bin/bash
# LOGIN NODE ONLY. Pulls ~60 GB. Reads the HF token from outside the repo --
# NEVER write a token into any file in this repository.
set -euo pipefail

PROJ="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJ HF_HOME="$PROJ/hf_cache"

if [ -f "$HOME/.hf_token" ]; then
  HF_TOKEN=$(cat "$HOME/.hf_token")
elif [ -f "$HOME/.cache/huggingface/token" ]; then
  HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
else
  echo "ERROR: no HF token found (~/.hf_token or ~/.cache/huggingface/token)." >&2
  echo "       Two of the four models are gated; accept their licences first." >&2
  exit 1
fi
export HF_TOKEN

source "$PROJ/envs/eval/bin/activate"
mkdir -p "$PROJ/models"

# `hf download` takes ONE pattern per --exclude flag. Passing several patterns
# after a single --exclude makes the extras positional FILENAMES, which act as
# an ALLOWLIST -- that inverts the filter and downloads exactly the junk we are
# trying to skip. Keep one flag per pattern.
#
#   original/*      Llama's 15 GB consolidated .pth checkpoint (vLLM can't use it)
#   consolidated*   Mistral ships a 14 GB single-file copy of the sharded weights
EXCLUDES=(
  --exclude "original/*"
  --exclude "consolidated*"
  --exclude "*.pth"
  --exclude "*.msgpack"
  --exclude "*.h5"
  --exclude "*.gguf"
)

REPOS=(
  Qwen/Qwen2.5-7B-Instruct
  meta-llama/Llama-3.1-8B-Instruct
  mistralai/Mistral-7B-Instruct-v0.3
  allenai/OLMo-2-1124-7B-Instruct
)

# A plain for-loop, not `... | while read` -- hf download inherits stdin from a
# pipe and can eat the remaining repo lines.
for repo in "${REPOS[@]}"; do
  dest="$PROJ/models/$(basename "$repo")"
  echo "[download] $repo -> $dest"
  hf download "$repo" --local-dir "$dest" "${EXCLUDES[@]}"
done

echo
echo "[verify] checking every model actually has weights ..."
fail=0
for repo in "${REPOS[@]}"; do
  dest="$PROJ/models/$(basename "$repo")"
  n=$(ls "$dest"/*.safetensors 2>/dev/null | wc -l)
  if [ -f "$dest/config.json" ] && [ "$n" -gt 0 ]; then
    echo "  [ ok ] $(basename "$repo"): config.json + $n safetensors shard(s)"
  else
    echo "  [FAIL] $(basename "$repo"): no config.json and/or no safetensors in $dest"
    fail=1
  fi
done
[ "$fail" = "0" ] || { echo "[download] INCOMPLETE -- do not proceed." >&2; exit 1; }

echo
echo "[download] recording resolved commit shas into configs/models.yaml ..."
python3 "$PROJ/scripts/record_revisions.py"
echo "[download] done. Weights are on disk; the compute node needs no token."
