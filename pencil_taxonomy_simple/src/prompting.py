"""Prompt rendering, chat templates, answer modes, and the hashes that pin
what was run.

Templates under prompts/ are FROZEN once a sweep starts. Every record stores
`template_sha256`, `prompt_sha256` and `answer_mode`, so "what exactly did the
model see, and was it allowed to reason?" has a factual answer afterwards.

Two answer modes, chosen in config.yaml (`generation.answer_mode`):

  direct      -- a single JSON object and nothing else. One forward pass per
                 answer; the state the model uses has to live in its activations.
                 This is the mode the earlier experiments ran, so it is the one
                 that stays comparable with them.
  scratchpad  -- the model first writes a short timestep-by-timestep tally and
                 THEN the JSON object, on its own last line. Lets an instruction
                 model do the book-keeping in text. Parsed by taking the LAST
                 JSON object in the output.

The one model-specific branch in the pipeline: Mistral-7B-Instruct-v0.3's chat
template raises on a system turn, so its format instruction is folded into the
user turn. Recorded per record as `used_system_role`, asserted in tests.
"""

import hashlib
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_DIR = os.path.join(ROOT, "prompts")

ANSWER_MODES = ("direct", "scratchpad")

FORMAT_INSTRUCTION = {
    "direct": (
        "Answer with a single JSON object and nothing else. No explanation, no "
        "markdown, no text before or after it. Use exactly this shape:"),
    "scratchpad": (
        "First work the answer out: write a brief tally of the relevant counts, "
        "one timestep per line, applying each transfer in order. Then, on the "
        "last line by itself, give the final answer as a single JSON object with "
        "no other text on that line. Use exactly this shape:"),
}
# When the instruction is hoisted into a system turn, the user turn still needs
# a lead-in or the schema dangles after a blank line. Both branches are kept
# identical apart from WHERE the instruction lives, so `used_system_role` is
# not confounded with a prompt difference.
SCHEMA_LEADIN = "Use exactly this shape:"

_CACHE = {}


def load_template(question_type):
    if question_type not in _CACHE:
        path = os.path.join(PROMPT_DIR, f"{question_type}.txt")
        with open(path, encoding="utf-8") as f:
            text = f.read()
        _CACHE[question_type] = (text, sha256(text))
    return _CACHE[question_type]


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def render_body(story_text, question, answer_mode, with_format_instruction=True):
    """The user-visible body. `with_format_instruction` is False when the
    instruction has been hoisted into a system turn instead."""
    if answer_mode not in ANSWER_MODES:
        raise ValueError(f"answer_mode must be one of {ANSWER_MODES}, got {answer_mode!r}")
    template, _ = load_template(question["question_type"])
    fields = {
        "story_text": story_text,
        "person": question.get("target_person"),
        "receiver": question.get("target_receiver"),
        "timestep": question.get("target_timestep"),
        "n_people": question.get("N"),
        "t_max": question.get("T"),
        "n_values": (question.get("T") or 0) + 1,
        "format_instruction": (FORMAT_INSTRUCTION[answer_mode]
                               if with_format_instruction else SCHEMA_LEADIN),
    }
    # The templates contain literal JSON braces in the schema line, so
    # str.format is not usable -- substitute placeholders explicitly instead.
    out = template
    for k, v in fields.items():
        out = out.replace("{" + k + "}", str(v))
    return out.strip()


def build_prompt(tokenizer, story_text, question, use_system_role, answer_mode):
    """Fully rendered, chat-templated prompt string plus its provenance."""
    _, tpl_sha = load_template(question["question_type"])
    if use_system_role:
        body = render_body(story_text, question, answer_mode,
                           with_format_instruction=False)
        system = FORMAT_INSTRUCTION[answer_mode].replace(
            " Use exactly this shape:", "")
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": body}]
    else:
        # Mistral: no system turn. The instruction stays inside the user body.
        body = render_body(story_text, question, answer_mode,
                           with_format_instruction=True)
        messages = [{"role": "user", "content": body}]

    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False)
    return prompt, tpl_sha, sha256(prompt)


def max_tokens_for(question, cfg, context_left=None):
    """Answer budget.

    Scratchpad mode adds room for the tally, sized per TRANSFER in the story
    (the count is in story_meta) rather than per timestep: a full replay writes
    one line per transfer, and that is what a model doing the book-keeping in
    text will produce. `context_left` caps the budget at what the model's window
    can still hold; a cap that bites shows up downstream as `truncated`, never
    as a silently shortened answer.
    """
    gen = cfg["generation"]
    mt = gen["max_tokens"]
    qt = question["question_type"]
    T, N = question["T"], question["N"]
    if qt == "state_snapshot":
        base = mt["state_snapshot_base"] + mt["per_unit"] * N
    elif qt == "trajectory":
        base = mt["trajectory_base"] + mt["per_unit"] * (T + 1)
    else:
        base = mt["scalar"]
    if gen.get("answer_mode", "direct") == "scratchpad":
        sp = mt["scratchpad"]
        n_transfers = len((question.get("story_meta") or {}).get("transfers", [])) or T
        base += sp["base"] + sp["per_transfer"] * n_transfers
    if context_left is not None:
        base = max(16, min(base, context_left))
    return base
