#!/bin/bash
# Prove envs/mech actually works on a GPU node BEFORE any real experiment.
#   sbatch scripts/probe_mech.sbatch
#
# The failure this exists to catch: a torch built for CUDA 13 installs cleanly,
# imports cleanly, reports device_count()==1, and still returns False from
# torch.cuda.is_available() here. Nothing before check 2 would notice.
#
#SBATCH --job-name=mech-probe
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --account=25m0834
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cccp/25m0834/MTP_Mechanistic_Interpretability/logs/%x-%j.out
#SBATCH --error=/home/cccp/25m0834/MTP_Mechanistic_Interpretability/logs/%x-%j.out

set -uo pipefail
export PROJ=/home/cccp/25m0834/MTP_Mechanistic_Interpretability
cd "$PROJ"

echo "=== node $(hostname -s) | partition ${SLURM_JOB_PARTITION:-?} | $(date -u) ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

source "$PROJ/envs/mech/bin/activate"
# The GPU nodes have no internet. Load by local path only.
export HF_HOME="$PROJ/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1 DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="$PROJ/.mplcache" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python - <<'PY'
import os, sys, traceback
FAIL = []
def check(name, fn):
    try:
        print(f"-- {name}")
        r = fn()
        print(f"   [ ok ] {r}")
    except Exception as e:
        print(f"   [FAIL] {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
        FAIL.append(name)

ROOT = os.environ["PROJ"]
MODEL = os.path.join(ROOT, "models", "Qwen2.5-7B-Instruct")

import torch
check("1. torch build",
      lambda: f"torch {torch.__version__}, built for CUDA {torch.version.cuda}")

def cuda_ok():
    assert torch.cuda.is_available(), (
        "torch.cuda.is_available() is False. This is THE failure mode: a CUDA 13 "
        "wheel on a CUDA 12.8 driver. Rebuild envs/mech with the cu128 pin.")
    return f"{torch.cuda.get_device_name(0)}, capability {torch.cuda.get_device_capability(0)}"
check("2. torch can USE the gpu", cuda_ok)

def matmul():
    a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    return f"bf16 matmul ran, result {tuple((a @ a).shape)}"
check("3. bf16 matmul", matmul)

def imports():
    import nnsight, transformer_lens
    return f"nnsight {nnsight.__version__}, transformer_lens {transformer_lens.__version__}"
check("4. library imports", imports)

def nns_load():
    from nnsight import LanguageModel
    # transformers renamed torch_dtype -> dtype in 4.56; accept either so the
    # probe fails on real problems, not on a keyword rename.
    for kw in ({"dtype": torch.bfloat16}, {"torch_dtype": torch.bfloat16}):
        try:
            lm = LanguageModel(MODEL, device_map="cuda", dispatch=True, **kw)
            globals()["LM"] = lm
            return (f"loaded {os.path.basename(MODEL)} offline by local path "
                    f"via {list(kw)[0]}=")
        except TypeError as e:
            last = e
    raise last
check("5. nnsight loads a model by local path", nns_load)

def trace():
    """The shape of the real experiment: read a residual stream, then patch it."""
    lm = globals()["LM"]
    n_layers = lm.config.num_hidden_layers
    with lm.trace("Arjun holds 7 pencils. Beatrix holds 3 pencils."):
        hid = lm.model.layers[n_layers // 2].output[0].save()
    return f"captured layer {n_layers // 2} residual stream, shape {tuple(hid.shape)}"
check("6. nnsight trace captures activations", trace)

def patch():
    lm = globals()["LM"]
    n_layers = lm.config.num_hidden_layers
    with lm.trace("Arjun holds 7 pencils."):
        lm.model.layers[n_layers // 2].output[0][:] = 0
        out = lm.output.logits.save()
    return f"zero-ablated a layer and got logits {tuple(out.shape)}"
check("7. nnsight can PATCH (the actual experiment)", patch)

print()
if FAIL:
    print(f"== mech probe FAILED: {', '.join(FAIL)} ==")
    sys.exit(1)
print("== mech probe PASSED -- envs/mech is cleared for the mechanistic phase ==")
PY
echo "=== exit=$? | $(date -u) ==="
