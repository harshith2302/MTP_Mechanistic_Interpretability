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

REPOS="Qwen/Qwen2.5-7B-Instruct
meta-llama/Llama-3.1-8B-Instruct
mistralai/Mistral-7B-Instruct-v0.3
allenai/OLMo-2-1124-7B-Instruct"

echo "$REPOS" | while read -r repo; do
  dest="$PROJ/models/$(basename "$repo")"
  echo "[download] $repo -> $dest"
  hf download "$repo" --local-dir "$dest" \
    --exclude "*.pth" "original/*" "*.msgpack" "*.h5"
done

echo
echo "[download] recording resolved commit shas into configs/models.yaml ..."
python3 "$PROJ/scripts/record_revisions.py"
echo "[download] done. Weights are on disk; the compute node needs no token."
