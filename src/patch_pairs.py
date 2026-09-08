"""Build token-aligned minimal pairs for the activation-patching experiment.

The behavioral phase localised the failure to the FIRST state update: models
answer `initial_state_lookup` at 100% and diverge from the true trajectory at
index 1 (REPORT.md sections 1 and 7). This module builds the stimuli that let us
ask *where* that update happens.

Each pair is one story, one person P, one timestep (t=1):

    clean    "... Arjun gave 4 pencils to Beatrix. ..."   -> P holds v_clean
    corrupt  "... Arjun gave 7 pencils to Beatrix. ..."   -> P holds v_corrupt

Exactly one number differs. Everything else -- names, templates, sentence order,
the question, the chat template -- is identical, so the two prompts tokenise to
the same length and differ at exactly one position. That is what makes a
residual-stream patch interpretable: any recovery of v_clean must have travelled
through the activations we moved, not through some incidental prompt difference.

Design constraints, each of which throws pairs away rather than weakening them:

- **Digits, not number words** (`number_word_probability=0.0`). "seven" and
  "four" are different token counts; "7" and "4" are one token each. The Phase 9
  ablation measured this surface form as a clean null (all |delta| < 2 points,
  every McNemar p > 0.09), so forcing digits costs no generality.
- **P is touched by exactly one transfer at t=1.** A single accumulation step is
  the thing we are localising; two transfers would confound it with composition.
- **Both amounts >= 2**, so `_pencil_word` stays "pencils" and the sentence
  differs by one token rather than two.
- **The counterfactual amount must be legal** -- the giver has to hold it at that
  point in the t=1 sequence, or the story stops being a valid simulation.
- **v_clean != v_corrupt**, both single tokens under the model's tokenizer, so
  the readout is one logit against one logit.

Standalone:  python -m src.patch_pairs --help
"""

import argparse
import json
import os

from src.prompts import apply_chat_template, build_body
from src.simulate import TRANSFER_SENTENCE_TEMPLATES, _pencil_word, generate_story

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The behavioral phase's schema shows values quoted ("answer": "<INTEGER>"), and
# models split between quoted and bare. Teacher-force up to and including the
# opening quote so the very next token is the count itself, identically for both
# members of a pair. scripts/verify_pairs.py checks empirically that the model
# actually puts its mass on a digit here.
ANSWER_PREFIX_QUOTED = '{{\n  "question_type": "person_timestep_lookup",\n  "person": "{name}",\n  "timestep": 1,\n  "answer": "'
ANSWER_PREFIX_BARE = '{{\n  "question_type": "person_timestep_lookup",\n  "person": "{name}",\n  "timestep": 1,\n  "answer": '


def render(template, a_name, b_name, amount):
    return template.format(A=a_name, B=b_name, n=str(amount),
                           pencils=_pencil_word(amount))


def locate_sentence(text, a_name, b_name, amount):
    """Recover the rendered sentence for one transfer.

    simulate.py picks the template with rng.choice and stores only the structured
    fields, so the sentence is not recoverable from `narration` alone. Format all
    eight templates and keep the one that is in the text -- requiring EXACTLY one
    template to match and EXACTLY one occurrence of it, so a coincidental repeat
    elsewhere in the story can never be edited by accident.
    """
    hits = []
    for tpl in TRANSFER_SENTENCE_TEMPLATES:
        s = render(tpl, a_name, b_name, amount)
        if text.count(s) == 1:
            hits.append((tpl, s))
    return hits[0] if len(hits) == 1 else (None, None)


def t1_holdings_before(story, upto_index):
    """Holdings as they stand just before the `upto_index`-th t=1 transfer.

    Needed to check that a counterfactual amount is one the giver could actually
    have handed over. t=1 transfers are applied in narration order.
    """
    held = dict(story["states"][0])
    t1 = [e for e in story["narration"] if e["timestep"] == 1]
    for e in t1[:upto_index]:
        held[e["giver"]] -= e["amount"]
        held[e["receiver"]] += e["amount"]
    return held


def build_pair(story, event, index, tokenizer, quoted=True):
    """One minimal pair from one t=1 transfer, or None if it fails a constraint."""
    names = story["id_to_name"]
    giver, receiver, amount = event["giver"], event["receiver"], event["amount"]
    # Both amounts must be single-digit: >=2 keeps `_pencil_word` at "pencils",
    # <=9 keeps the amount one token. Capping only the counterfactual is not
    # enough -- an original amount of 12 is two tokens and silently breaks
    # alignment (16 of the first 200 pairs did exactly that).
    if not 2 <= amount <= 9:
        return None

    t1 = [e for e in story["narration"] if e["timestep"] == 1]
    # Trying giver first every time skews the set (156/44 on the first build).
    # Alternate the preference by seed parity so both roles are represented --
    # giver and receiver are different computations (subtract vs add) and a
    # patching result that only holds for one of them would be a weaker claim.
    order = (giver, receiver) if story["config"]["seed"] % 2 else (receiver, giver)
    for pid in order:
        touched = [e for e in t1 if pid in (e["giver"], e["receiver"])]
        if len(touched) != 1:
            continue

        tpl, clean_s = locate_sentence(story["text"], names[giver], names[receiver],
                                       amount)
        if tpl is None:
            continue

        held = t1_holdings_before(story, index)
        cap = min(9, held[giver])
        for alt in range(2, cap + 1):
            if alt == amount:
                continue
            v_clean = story["states"][1][pid]
            delta = alt - amount
            v_corrupt = v_clean - delta if pid == giver else v_clean + delta
            if v_corrupt < 0 or v_corrupt == v_clean:
                continue
            if tokenizer is not None:
                toks = [tokenizer.encode(str(v), add_special_tokens=False)
                        for v in (v_clean, v_corrupt)]
                if any(len(t) != 1 for t in toks):
                    continue

            corrupt_s = render(tpl, names[giver], names[receiver], alt)
            corrupt_text = story["text"].replace(clean_s, corrupt_s, 1)
            if corrupt_text == story["text"]:
                continue
            question = {
                "question_type": "person_timestep_lookup",
                "queried_person": pid,
                "queried_timestep": 1,
                "question_text": (f"How many pencils did {names[pid]} have at "
                                  f"timestep 1?"),
                # Must match src/questions.py:q_person_timestep_lookup exactly --
                # the schema is part of the prompt, so a different one would make
                # these stimuli incomparable with the behavioral records.
                "schema": {"question_type": "person_timestep_lookup",
                           "person": "<PERSON_NAME>", "timestep": "<INTEGER>",
                           "answer": "<INTEGER>"},
                "gold": {"answer": v_clean},
            }
            pre = (ANSWER_PREFIX_QUOTED if quoted else ANSWER_PREFIX_BARE)
            suffix = pre.format(name=names[pid])
            # The prefix must sit in the ASSISTANT turn. Appending it to the user
            # body instead puts the model's own partial answer inside the user's
            # question, which is a different task and would invalidate the
            # readout. apply_chat_template(add_generation_prompt=True) ends at the
            # assistant header, so the prefix continues from there.
            clean_body = (apply_chat_template(tokenizer,
                                              build_body(story["text"], question))
                          + suffix)
            corrupt_body = (apply_chat_template(tokenizer,
                                                build_body(corrupt_text, question))
                            + suffix)
            # The constraints above SHOULD guarantee alignment; verify it anyway
            # and drop the pair if not. A misaligned pair does not just add noise
            # -- patching position i in one prompt would hit a different token in
            # the other, so every downstream number would be quietly wrong.
            diff_pos = None
            if tokenizer is not None:
                ct = tokenizer.encode(clean_body, add_special_tokens=False)
                xt = tokenizer.encode(corrupt_body, add_special_tokens=False)
                if len(ct) != len(xt):
                    continue
                d = [i for i, (u, v) in enumerate(zip(ct, xt)) if u != v]
                if len(d) != 1:
                    continue
                diff_pos = d[0]
            return {
                "n_tokens": len(ct) if tokenizer is not None else None,
                "diff_token_index": diff_pos,
                "story_seed": story["config"]["seed"],
                "n_people": story["config"]["n_people"],
                "person_id": pid,
                "person_name": names[pid],
                "role": "giver" if pid == giver else "receiver",
                "clean_amount": amount, "corrupt_amount": alt,
                "v_clean": v_clean, "v_corrupt": v_corrupt,
                "clean_sentence": clean_s, "corrupt_sentence": corrupt_s,
                "clean_body": clean_body,
                "corrupt_body": corrupt_body,
            }
    return None


def iter_pairs(n, n_stories, seed0, tokenizer, quoted=True, limit=None):
    out = []
    for i in range(n_stories):
        seed = seed0 + i
        story = generate_story(n, n, seed, number_word_probability=0.0)
        t1 = [e for e in story["narration"] if e["timestep"] == 1]
        for idx, ev in enumerate(t1):
            p = build_pair(story, ev, idx, tokenizer, quoted)
            if p:
                out.append(p)
                if limit and len(out) >= limit:
                    return out
                break          # one pair per story keeps stories independent
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=10, help="N = T (the 8-12 regime)")
    ap.add_argument("--stories", type=int, default=400)
    ap.add_argument("--seed0", type=int, default=900000)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--model", default="models/Qwen2.5-7B-Instruct")
    ap.add_argument("--bare", action="store_true",
                    help="teacher-force an unquoted answer instead of a quoted one")
    ap.add_argument("--out", default="results/patching/pairs_N10.jsonl")
    a = ap.parse_args()

    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, a.model))
    except Exception as e:                       # noqa: BLE001
        print(f"[patch_pairs] no tokenizer ({e}); skipping single-token filter")

    pairs = iter_pairs(a.n, a.stories, a.seed0, tok, not a.bare, a.limit)
    out = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"[patch_pairs] {len(pairs)} pairs from {a.stories} stories -> {out}")
    if tok is not None:
        enc = lambda x: tok.encode(x, add_special_tokens=False)   # noqa: E731
        bad = [p for p in pairs
               if len(enc(p["clean_body"])) != len(enc(p["corrupt_body"]))]
        print(f"[patch_pairs] token-length mismatches: {len(bad)} "
              f"{'<-- PROBLEM' if bad else '(all aligned)'}")


if __name__ == "__main__":
    main()
