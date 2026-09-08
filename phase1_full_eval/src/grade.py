"""Tolerant JSON extraction + exact-match grading for all 12 question types.

The four original comparison rules are ported from Pencil_Exchange/verifier.py
unchanged. Parsing is deliberately tolerant (fences, prose around the object,
"7" for 7) because `format_error` is a measured taxonomy category and we want it
to mean "the model really could not produce JSON", not "we were strict".
"""

import json
import re

PARSE_OK = "ok"
PARSE_OK_REPAIRED = "ok_repaired"
# Both mean "we have a dict to grade". Anything else is a format failure.
OK_STATUSES = frozenset({PARSE_OK, PARSE_OK_REPAIRED})
REFUSAL_PATTERNS = re.compile(
    r"\b(i'm sorry|i am sorry|i cannot|i can't|as an ai|unable to (?:answer|determine)"
    r"|there is not enough information|cannot be determined)\b",
    re.IGNORECASE,
)


# --- extraction ---------------------------------------------------------------
def _strip_fences(text):
    text = re.sub(r"^\s*```(?:json)?\s*", "", text.strip())
    return re.sub(r"\s*```\s*$", "", text)


def _first_balanced_object(text, allow_repair=False):
    """First balanced {...} substring, respecting strings and escapes.

    -> (substring, was_repaired) or (None, False).

    With `allow_repair`, an object left open at end of text has its missing
    closers appended. Mistral-7B-v0.3 ends a great many otherwise-perfect
    answers one "}" short with finish_reason=stop; scoring those as
    format_error would have cost it ~37% of its records and filled its
    taxonomy with a category it did not earn.

    Repair is refused when the text ends INSIDE a string, because a cut-off
    string value is genuinely lost content, not a missing delimiter.
    """
    stack = []
    start = None
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            if ch == "{" and not stack:
                start = i
            if start is not None:
                stack.append(ch)
        elif ch in "}]":
            if stack and stack[-1] == ("{" if ch == "}" else "["):
                stack.pop()
                if not stack and start is not None:
                    return text[start:i + 1], False
    if not (allow_repair and stack and start is not None and not in_str):
        return None, False
    body = text[start:].rstrip()
    body = re.sub(r",\s*$", "", body)          # a dangling comma would be invalid
    closers = "".join("}" if ch == "{" else "]" for ch in reversed(stack))
    return body + closers, True


def parse_answer(raw_text, finish_reason=None):
    """-> (parsed_dict_or_None, parse_status)."""
    if raw_text is None or not raw_text.strip():
        return None, "empty"
    if REFUSAL_PATTERNS.search(raw_text) and "{" not in raw_text:
        return None, "refusal"
    body = _strip_fences(raw_text)
    # finish_reason "length" means the token budget cut the answer off, so the
    # content itself is missing and repairing delimiters would invent an answer.
    # "stop" means the model chose to end: only delimiters can be absent.
    truncated_by_budget = finish_reason == "length"
    candidate, repaired = _first_balanced_object(
        body, allow_repair=not truncated_by_budget)
    if candidate is None:
        return None, "truncated" if truncated_by_budget else "no_json_found"
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        if truncated_by_budget:
            return None, "truncated"
        return None, "invalid_json"
    if not isinstance(parsed, dict):
        return None, "invalid_json"
    return parsed, PARSE_OK_REPAIRED if repaired else PARSE_OK


# --- coercion -----------------------------------------------------------------
def as_int(value):
    """int-ify tolerantly: 7, "7", 7.0, " 7 ", "+7", "-7" -> int; else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value == int(value) else None
    if isinstance(value, str):
        s = value.strip().replace(",", "")
        try:
            return int(s)
        except ValueError:
            try:
                f = float(s)
                return int(f) if f == int(f) else None
            except ValueError:
                return None
    return None


def as_int_list(value):
    if not isinstance(value, (list, tuple)):
        return None
    out = []
    for v in value:
        i = as_int(v)
        if i is None:
            return None
        out.append(i)
    return out


def get_person_id(name_or_id, story):
    if name_or_id in story["name_to_id"]:
        return story["name_to_id"][name_or_id]
    if name_or_id in story["id_to_name"]:
        return name_or_id
    if isinstance(name_or_id, str):
        low = {k.lower(): v for k, v in story["name_to_id"].items()}
        return low.get(name_or_id.strip().lower())
    return None


def _norm_name(s):
    return s.strip().lower() if isinstance(s, str) else s


# --- grading ------------------------------------------------------------------
_REQUIRED = {
    "person_timestep_lookup": ["answer"],
    "person_value_timesteps": ["timesteps"],
    "population_condition": ["answer"],
    "duration_condition": ["answer"],
    "trajectory": ["counts"],
    "state_snapshot": ["counts"],
    "initial_state_lookup": ["answer"],
    "transfer_recall": ["answer"],
    "pairwise_comparison": ["answer"],
    "argmax_person": ["answer"],
    "total_conservation": ["answer"],
    "net_delta": ["answer"],
}

_SCALAR_INT_TYPES = {
    "person_timestep_lookup", "population_condition", "duration_condition",
    "initial_state_lookup", "transfer_recall", "total_conservation", "net_delta",
}


def grade(question, parsed, story=None, parse_status=None):
    """-> {correct, gold, pred, status, detail}. `status` refines parse_status.

    `parse_status` is threaded through only so the classifier can tell a list
    that the model actually wrote short from one our delimiter repair closed
    early -- the two deserve different labels.
    """
    qtype = question["question_type"]
    gold = question["gold"]
    out = {"correct": False, "gold": gold, "pred": None,
           "status": PARSE_OK,
           "detail": {"repaired": parse_status == PARSE_OK_REPAIRED}}

    if parsed is None:
        out["status"] = "missing"
        return out

    declared = parsed.get("question_type")
    if declared is not None and declared != qtype:
        out["status"] = "wrong_question_type"
        out["detail"]["declared_type"] = declared
        return out

    missing = [f for f in _REQUIRED[qtype] if f not in parsed]
    if missing:
        out["status"] = "missing_fields"
        out["detail"]["missing"] = missing
        return out

    if qtype in _SCALAR_INT_TYPES:
        pred = as_int(parsed["answer"])
        if pred is None:
            out["status"] = "bad_types"
            return out
        out["pred"] = pred
        out["correct"] = pred == gold["answer"]

    elif qtype == "person_value_timesteps":
        pred = as_int_list(parsed["timesteps"])
        if pred is None:
            out["status"] = "bad_types"
            return out
        out["pred"] = sorted(pred)
        out["correct"] = sorted(pred) == sorted(gold["timesteps"])
        out["detail"]["missed"] = sorted(set(gold["timesteps"]) - set(pred))
        out["detail"]["spurious"] = sorted(set(pred) - set(gold["timesteps"]))

    elif qtype == "trajectory":
        pred = as_int_list(parsed["counts"])
        if pred is None:
            out["status"] = "bad_types"
            return out
        out["pred"] = pred
        g = gold["counts"]
        out["correct"] = pred == g
        out["detail"]["length_mismatch"] = len(pred) != len(g)
        div = None
        for i in range(min(len(pred), len(g))):
            if pred[i] != g[i]:
                div = i
                break
        if div is None and len(pred) != len(g):
            div = min(len(pred), len(g))
        out["detail"]["first_divergence_index"] = div
        if div is not None:
            tail_g, tail_p = g[div:], pred[div:]
            n = min(len(tail_g), len(tail_p))
            out["detail"]["recovered"] = any(
                tail_p[i] == tail_g[i] for i in range(n)
            )
            errs = [abs(tail_p[i] - tail_g[i]) for i in range(n)]
            out["detail"]["error_growth"] = (
                len(errs) > 1 and all(errs[i] <= errs[i + 1] for i in range(len(errs) - 1))
            )
            out["detail"]["mean_abs_error"] = sum(errs) / len(errs) if errs else 0.0

    elif qtype == "state_snapshot":
        raw = parsed["counts"]
        if not isinstance(raw, dict):
            out["status"] = "bad_types"
            return out
        pred = {}
        for k, v in raw.items():
            i = as_int(v)
            if i is None:
                out["status"] = "bad_types"
                return out
            pred[_norm_name(k)] = i
        g = {_norm_name(k): v for k, v in gold["counts"].items()}
        out["pred"] = pred
        out["correct"] = pred == g
        matched = sum(1 for k, v in g.items() if pred.get(k) == v)
        out["detail"]["partial_correct_fraction"] = matched / len(g) if g else 0.0
        out["detail"]["n_people_answered"] = len(pred)
        out["detail"]["missing_people"] = sorted(set(g) - set(pred))
        total = story["config"]["total_pencils"] if story else sum(g.values())
        out["detail"]["pred_sum"] = sum(pred.values())
        out["detail"]["conservation_violation"] = sum(pred.values()) != total
        out["detail"]["permutation"] = (
            not out["correct"]
            and set(pred) == set(g)
            and sorted(pred.values()) == sorted(g.values())
        )

    elif qtype in ("pairwise_comparison", "argmax_person"):
        pred = parsed["answer"]
        if not isinstance(pred, str):
            out["status"] = "bad_types"
            return out
        out["pred"] = pred.strip()
        out["correct"] = _norm_name(pred) == _norm_name(gold["answer"])

    else:  # unreachable while _REQUIRED covers every type
        out["status"] = "unknown_question_type"

    return out
