"""Prompt assembly.

Body is model-independent; the chat wrapper comes from each model's own
tokenizer template — never a hand-written [INST] string. The rendered prompt is
stored once in prompts.jsonl keyed by SHA-256; records carry only the hash.
"""

import argparse
import hashlib
import json

INSTRUCTION = (
    "Answer ONLY in valid JSON matching this exact schema. Do not include "
    "explanations, markdown, or any text outside the JSON object."
)


def build_body(story_text: str, question: dict) -> str:
    """The model-independent user message."""
    return (
        f"Below is a description of a pencil exchange.\n\n"
        f"{story_text}\n\n"
        f"Question:\n{question['question_text']}\n\n"
        f"{INSTRUCTION}\n"
        f"{json.dumps(question['schema'], indent=2)}"
    )


def apply_chat_template(tokenizer, body: str) -> str:
    """Wrap with the model's own chat template. Falls back to raw body."""
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return body
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": body}],
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception:
        return body


def build_prompt(story_text: str, question: dict, tokenizer=None) -> str:
    return apply_chat_template(tokenizer, build_body(story_text, question))


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


class PromptStore:
    """Append-only sidecar mapping sha256 -> rendered prompt."""

    def __init__(self, path):
        self.path = path
        self.seen = set()
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.seen.add(json.loads(line)["sha256"])
        except FileNotFoundError:
            pass
        self._fh = open(path, "a", encoding="utf-8")

    def add(self, prompt: str) -> str:
        h = prompt_sha256(prompt)
        if h not in self.seen:
            self._fh.write(json.dumps({"sha256": h, "prompt": prompt}) + "\n")
            self.seen.add(h)
        return h

    def flush(self):
        self._fh.flush()

    def close(self):
        self._fh.close()


def main():
    from src.questions import sample_questions
    from src.simulate import generate_story

    ap = argparse.ArgumentParser(description="Render example prompts.")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=4000)
    ap.add_argument("--type", help="restrict to one question type")
    ap.add_argument("--limit", type=int, default=2)
    ap.add_argument("--model-path", help="load this tokenizer and apply its template")
    args = ap.parse_args()

    tok = None
    if args.model_path:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)

    story = generate_story(args.n, args.n, args.seed)
    qs = sample_questions(story, args.seed, types=[args.type] if args.type else None)
    for q in qs[: args.limit]:
        prompt = build_prompt(story["text"], q, tok)
        print("=" * 78)
        print(f"{q['question_type']}  sha={prompt_sha256(prompt)[:12]}  "
              f"chars={len(prompt)}")
        print("=" * 78)
        print(prompt)
        print()


if __name__ == "__main__":
    main()
