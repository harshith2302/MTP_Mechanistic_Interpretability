"""Prompt rendering, chat templates, and the hashes that pin what was run.

Templates under prompts/ are FROZEN once the sweep starts. Every record stores
`template_sha256` and `prompt_sha256`, so "what exactly did the model see?" has a
factual answer afterwards rather than a reconstruction.

The one model-specific branch in the whole pipeline lives here:
Mistral-7B-Instruct-v0.3's chat template raises on a system turn, so its schema
and format instruction are folded into the user turn instead. That is recorded
per record as `used_system_role`, and asserted in tests -- silently losing the
format instruction for one model depresses its numbers and looks like a finding.
"""

import hashlib
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_DIR = os.path.join(ROOT, "prompts")

FORMAT_INSTRUCTION = (
    "Answer with a single JSON object and nothing else. No explanation, no "
    "markdown, no text before or after it. Use exactly this shape:"
)
# When the full instruction is hoisted into a system turn, the user turn still
# needs a lead-in or the schema dangles after a blank line with nothing
# introducing it. Keeping both branches identical apart from WHERE the
# instruction lives is the point -- otherwise `used_system_role` would be
# confounded with a prompt difference.
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


def render_body(story_text, question, with_format_instruction=True):
    """The user-visible body. `with_format_instruction` is False when the
    instruction has been hoisted into a system turn instead."""
    template, _ = load_template(question["question_type"])
    fields = {
        "story_text": story_text,
        "person": question.get("target_person"),
        "receiver": question.get("target_receiver"),
        "timestep": question.get("target_timestep"),
        "n_people": question.get("N"),
        "t_max": question.get("T"),
        "n_values": (question.get("T") or 0) + 1,
        "format_instruction": (FORMAT_INSTRUCTION if with_format_instruction
                               else SCHEMA_LEADIN),
    }
    # The templates contain literal JSON braces in the schema line, so
    # str.format is not usable -- substitute placeholders explicitly instead.
    out = template
    for k, v in fields.items():
        out = out.replace("{" + k + "}", str(v))
    return out.strip()


def build_prompt(tokenizer, story_text, question, use_system_role):
    """Fully rendered, chat-templated prompt string plus its provenance."""
    _, tpl_sha = load_template(question["question_type"])
    if use_system_role:
        body = render_body(story_text, question, with_format_instruction=False)
        messages = [
            {"role": "system", "content": FORMAT_INSTRUCTION.replace(
                " Use exactly this shape:", "")},
            {"role": "user", "content": body},
        ]
    else:
        # Mistral: no system turn. The instruction stays inside the user body.
        body = render_body(story_text, question, with_format_instruction=True)
        messages = [{"role": "user", "content": body}]

    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False)
    return prompt, tpl_sha, sha256(prompt)


def max_tokens_for(question, cfg):
    mt = cfg["generation"]["max_tokens"]
    qt = question["question_type"]
    if qt == "state_snapshot":
        return mt["state_snapshot_base"] + mt["per_unit"] * question["N"]
    if qt == "trajectory":
        return mt["trajectory_base"] + mt["per_unit"] * (question["T"] + 1)
    return mt["scalar"]
