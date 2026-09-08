"""Grading: exact match, plus one secondary metric for the vector question.

Exact match, no partial credit -- the question is whether the model can do the
book-keeping, not whether it is close.

The one exception is `state_snapshot`, where exact match on an N-element vector
goes to zero and stops discriminating. Per-element accuracy is reported
ALONGSIDE exact match, never in place of it.

Denominators: everything except `context_overflow` counts. Format errors ARE
wrong answers and stay in the denominator, but are also reported separately as
`format_error_rate` so both readings are available.
"""

from src.parsing import extract_answer, normalise, parse


def grade_record(rec):
    """Fill every derived field of a record from its stored `raw_output`.

    Pure function of the record's raw fields -- this is what makes re-grading
    without a GPU possible, and it is the reason nothing downstream may read the
    model or the tokenizer.
    """
    qt = rec["question_type"]
    gt = rec["ground_truth"]

    if rec.get("context_overflow"):
        rec.update(parsed=None, parse_path="not_generated", format_error=False,
                   correct=False, correct_strict=False,
                   normalisation_rescued=False, per_element_accuracy=None)
        return rec

    parsed, path, fmt_err = parse(rec.get("raw_output"), rec.get("truncated", False))
    rec["parsed"] = parsed
    rec["parse_path"] = path
    rec["format_error"] = fmt_err

    if fmt_err:
        rec.update(correct=False, correct_strict=False,
                   normalisation_rescued=False, per_element_accuracy=None)
        return rec

    ans = extract_answer(parsed, qt)
    strict = _strict_equal(ans, gt, qt)
    norm_ans = normalise(ans, qt)
    norm_gt = normalise(gt, qt)
    loose = norm_ans is not None and norm_ans == norm_gt

    rec["correct_strict"] = bool(strict)
    rec["correct"] = bool(loose)
    rec["normalisation_rescued"] = bool(loose and not strict)
    rec["per_element_accuracy"] = (
        _per_element(norm_ans, norm_gt) if qt == "state_snapshot" else None)
    return rec


def _strict_equal(ans, gt, qt):
    """Byte-exact: same types, same keys, same order. No coercion at all."""
    if qt == "trajectory":
        return isinstance(ans, list) and ans == gt
    if qt == "state_snapshot":
        return isinstance(ans, dict) and ans == gt
    return isinstance(ans, int) and not isinstance(ans, bool) and ans == gt


def _per_element(norm_ans, norm_gt):
    """Fraction of people whose count is right. None if the shape is unusable."""
    if not isinstance(norm_ans, dict) or not isinstance(norm_gt, dict) or not norm_gt:
        return None
    hit = sum(1 for k, v in norm_gt.items() if norm_ans.get(k) == v)
    return round(hit / len(norm_gt), 4)


def first_divergence(norm_ans, norm_gt):
    """Index where a trajectory answer first leaves the truth, or None."""
    if not isinstance(norm_ans, list) or not isinstance(norm_gt, list):
        return None
    for i in range(min(len(norm_ans), len(norm_gt))):
        if norm_ans[i] != norm_gt[i]:
            return i
    if len(norm_ans) != len(norm_gt):
        return min(len(norm_ans), len(norm_gt))
    return None


def is_formulaic(seq):
    """Constant list or perfect arithmetic progression.

    These diverge at index 0-1 by construction -- they are pattern completion,
    not tracking -- so figure 4 excludes them. Length >= 3 required, otherwise
    every short sequence is trivially a progression.
    """
    if not isinstance(seq, list) or len(seq) < 3:
        return False
    diffs = {seq[i + 1] - seq[i] for i in range(len(seq) - 1)}
    return len(diffs) == 1
