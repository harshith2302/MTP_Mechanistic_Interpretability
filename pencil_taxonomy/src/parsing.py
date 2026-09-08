"""Lenient JSON parsing and answer normalisation.

Both halves of this file exist because of a specific, expensive failure:

  * PARSING. A model that ends its JSON one closing brace short is not wrong --
    it is unparsed. In the source experiment that single missing character had a
    model scored at 9.4% when its true accuracy was 28.4%. `raw_output` is stored
    verbatim precisely so a parser bug can be fixed by re-grading rather than by
    re-running a GPU sweep.

  * NORMALISATION. `"5"` and `5` are the same answer; `" Ram"` and `"ram"` are the
    same key. But a grader that is quietly forgiving inflates the headline
    number, so BOTH readings are kept: `correct_strict` (byte-exact) and
    `correct` (normalised). Their difference is reported, never hidden.

Every repair step that fires is recorded in `parse_path`, so the cost of
leniency is measurable rather than assumed.
"""

import json
import re

FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


def parse(raw, truncated=False):
    """-> (parsed_or_None, parse_path, format_error).

    `truncated` suppresses brace repair: if the token budget cut the output off
    mid-object, the content is genuinely incomplete and repairing it would
    invent an answer the model never finished forming.
    """
    steps = []
    if raw is None:
        return None, "no_output", True
    text = raw.strip()
    if not text:
        return None, "empty", True

    m = FENCE.search(text)
    if m:
        text = m.group(1).strip()
        steps.append("fence_stripped")

    start = text.find("{")
    if start == -1:
        return None, "+".join(steps + ["no_brace"]), True
    if start > 0:
        text = text[start:]
        steps.append("prose_stripped")

    end = text.rfind("}")
    if end != -1 and end < len(text) - 1:
        text = text[:end + 1]
        steps.append("trailing_stripped")

    obj, err = _try(text)
    if obj is not None:
        return obj, "+".join(steps + ["direct"]) if steps else "direct", False

    # One closing brace short, and the model chose to stop -> repair.
    if not truncated and text.count("{") == text.count("}") + 1:
        obj, _ = _try(text + "}")
        if obj is not None:
            return obj, "+".join(steps + ["brace_repaired"]), False

    return None, "+".join(steps + ["unparseable"]), True


def _try(text):
    try:
        obj = json.loads(text)
        return (obj, None) if isinstance(obj, dict) else (None, "not_an_object")
    except Exception as e:                                   # noqa: BLE001
        return None, str(e)


# --- normalisation -----------------------------------------------------------

def _int(v):
    """int, integral float, or numeric string -> int. Otherwise None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v) if v.is_integer() else None   # reject 5.5, never round it
    if isinstance(v, str):
        s = v.strip().replace(",", "")
        try:
            f = float(s)
        except ValueError:
            return None
        return int(f) if f.is_integer() else None
    return None


def normalise(value, question_type):
    """Comparable form of an answer, or None if it is not of the right shape."""
    if question_type == "state_snapshot":
        if not isinstance(value, dict):
            return None
        out = {}
        for k, v in value.items():
            iv = _int(v)
            if iv is None:
                return None
            out[str(k).strip().casefold()] = iv
        return out                       # dict compare ignores key order
    if question_type == "trajectory":
        if not isinstance(value, (list, tuple)):
            return None
        out = [_int(v) for v in value]
        return None if any(v is None for v in out) else out   # order matters
    return _int(value)


def extract_answer(parsed, question_type):
    """Pull the answer out of a parsed object, tolerating a couple of shapes."""
    if not isinstance(parsed, dict):
        return None
    if "answer" in parsed:
        return parsed["answer"]
    # Some models drop the wrapper and return the object/list directly.
    if question_type == "state_snapshot" and parsed:
        if all(_int(v) is not None for v in parsed.values()):
            return parsed
    return None
