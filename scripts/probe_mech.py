"""GPU-side verification of envs/mech. Run by scripts/probe_mech.sbatch.

MUST live in a real file, not a heredoc. nnsight 0.7 builds its intervention
graph by reading the SOURCE of the `with lm.trace(...)` block via
inspect.getsource(); code arriving on stdin has no retrievable source and every
trace dies with `OSError: could not get source code`. That is what happened to
probe job 305168. The same constraint applies to every experiment script in the
mechanistic phase: no `python -c`, no heredocs, no exec'd strings.
"""
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
    import nnsight, transformer_lens  # noqa: F401
    # transformer_lens exposes no __version__; ask the metadata, not the module.
    from importlib.metadata import version
    return ", ".join(f"{p} {version(p)}" for p in ("nnsight", "transformer-lens"))
check("4. library imports", imports)

def template_parity():
    """The mech venv resolves transformers 5.x; the sweep ran on 4.57.

    If chat-template rendering differs at all, every activation would be taken
    from a prompt the behavioral results never saw, and the two phases could not
    be compared. Cheap to check, catastrophic to miss.
    """
    import json
    sys.path.insert(0, ROOT)
    from src.prompts import build_body, prompt_sha256
    from src.questions import sample_questions
    from src.simulate import generate_story
    from transformers import AutoTokenizer
    run = os.path.join(ROOT, "results/raw/2026-09-07T1341Z_304971",
                       "Qwen2.5-7B-Instruct", "prompts.jsonl")
    if not os.path.exists(run):
        return "skipped: sweep prompts.jsonl not found"
    stored = {json.loads(l)["sha256"] for l in open(run)}
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    hit = miss = 0
    for n in (8, 12):
        story = generate_story(n, n, 1000 * n)
        for q in sample_questions(story, 1000 * n):
            r = tok.apply_chat_template(
                [{"role": "user", "content": build_body(story["text"], q)}],
                add_generation_prompt=True, tokenize=False)
            hit, miss = (hit + 1, miss) if prompt_sha256(r) in stored else (hit, miss + 1)
    assert miss == 0, (
        f"{miss} prompts render differently than the sweep's. Activations would "
        f"not correspond to the behavioral results. Do NOT proceed.")
    return f"{hit}/{hit} prompts render byte-identically to the sweep"
check("5. chat-template parity with the sweep", template_parity)


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
check("6. nnsight loads a model by local path", nns_load)

def trace():
    """The shape of the real experiment: read a residual stream, then patch it."""
    lm = globals()["LM"]
    n_layers = lm.config.num_hidden_layers
    with lm.trace("Arjun holds 7 pencils. Beatrix holds 3 pencils."):
        hid = lm.model.layers[n_layers // 2].output[0].save()
    return f"captured layer {n_layers // 2} residual stream, shape {tuple(hid.shape)}"
check("7. nnsight trace captures activations", trace)

def patch():
    lm = globals()["LM"]
    n_layers = lm.config.num_hidden_layers
    with lm.trace("Arjun holds 7 pencils."):
        lm.model.layers[n_layers // 2].output[0][:] = 0
        out = lm.output.logits.save()
    return f"zero-ablated a layer and got logits {tuple(out.shape)}"
check("8. nnsight can PATCH (the actual experiment)", patch)

print()
if FAIL:
    print(f"== mech probe FAILED: {', '.join(FAIL)} ==")
    sys.exit(1)
print("== mech probe PASSED -- envs/mech is cleared for the mechanistic phase ==")
