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


def parse(raw, truncated=False, prefer_last=False):
    """-> (parsed_or_None, parse_path, format_error).

    `truncated` suppresses brace repair: if the token budget cut the output off
    mid-object, the content is genuinely incomplete and repairing it would
    invent an answer the model never finished forming.

    `prefer_last` is the scratchpad-mode reading: the model was told to reason
    first and put the JSON on the last line, so the answer is the LAST balanced
    object in the text, not the first. Direct mode keeps the first-object
    reading so it grades exactly as the earlier experiments did.
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

    if prefer_last:
        obj = _last_object(text)
        if obj is not None:
            return obj, "+".join(steps + ["last_object"]), False
        # fall through: maybe there is exactly one object, possibly unclosed

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


def _last_object(text):
    """The last balanced {...} that parses as a JSON object, or None.

    Scans "{" positions from the end, so reasoning text that happens to contain
    braces before the answer cannot shadow it. Strings and escapes are
    respected so a "}" inside a quoted name does not end the object early.

    Prefers the last object that carries the "answer" key: for a nested answer
    like {"answer": {"Ana": 5}} the very last "{" opens the INNER dict, and
    returning that would drop the wrapper. An object without the key is kept
    only as a fallback.
    """
    starts = [i for i, ch in enumerate(text) if ch == "{"]
    fallback = None
    for s in reversed(starts):
        depth, in_str, esc = 0, False, False
        for j in range(s, len(text)):
            ch = text[j]
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
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    obj, _ = _try(text[s:j + 1])
                    if obj is not None:
                        if "answer" in obj:
                            return obj
                        if fallback is None:
                            fallback = obj
                    break
    return fallback


def _try(text):
    try:
        obj = json.loads(text)
        return (obj, None) if isinstance(obj, dict) else (None, "not_an_object")
    except Exception as e:                                   # noqa: BLE001
        return None, str(e)


# --- normalisation -----------------------------------------------------------

def _number_words():
    """'seven' -> 7, for 0..999: the story writes ~45% of its numbers as words,
    and the prompt says the two forms are equivalent, so a model that copies
    the word back has answered correctly. Built from num2words so the mapping
    is exactly the one the generator used."""
    from num2words import num2words
    out = {}
    for i in range(1000):
        w = num2words(i)
        out[w] = i
        out[w.replace("-", " ")] = i
        out[w.replace(" and ", " ")] = i
    return out


_WORDS = None


def _int(v):
    """int, integral float, numeric string, or number WORD -> int. Otherwise None.

    A number word is a normalisation, not a strict match: `correct_strict`
    stays False and `normalisation_rescued` records it, so the cost of this
    leniency is measurable. In the 2026-09-14 sweep it was 237 answers, all
    OLMo-2, 232 of them on the initial_state_lookup control."""
    global _WORDS
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
            if _WORDS is None:
                _WORDS = _number_words()
            return _WORDS.get(s.lower())
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
