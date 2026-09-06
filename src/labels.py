"""Failure-label vocabulary: fine label -> coarse category, and the tie-break
priority used to pick a single winning label when several hypotheses match.

Split out of classify.py so that module stays readable; this file is data only.
Coarse categories are the 9 from Papers.md §7 plus Conservation and Unexplained.
"""

COARSE = {
    "format_error": "Format", "refusal_or_truncation": "Format",
    "wrong_question_type": "Format", "missing_fields": "Format",
    "bad_types": "Format",
    "out_of_range": "Conservation", "conservation_violation": "Conservation",
    "invariant_violation": "Conservation",
    "binding_person": "Binding", "permutation": "Binding",
    "transfer_binding_other": "Binding",
    "temporal_initial": "Temporal", "temporal_off_by_one": "Temporal",
    "temporal_other": "Temporal", "included_t0": "Temporal",
    "excluded_t0": "Temporal",
    "omission_1": "Omission", "omission_2": "Omission", "omission_3": "Omission",
    "over_application": "Over-application",
    "direction_flip_one": "Direction", "direction_flip_all": "Direction",
    "direction_confusion": "Direction",
    "sign_error_last": "Arithmetic", "sign_error": "Arithmetic",
    "off_by_small": "Arithmetic", "boundary_error": "Arithmetic",
    "condition_flip": "Arithmetic",
    "digit_error": "Digit",
    "referent_adjacent": "Referent", "comparison_error": "Referent",
    "unexplained": "Unexplained", "correct": "Correct",
}

PRIORITY = [
    "format_error", "refusal_or_truncation", "wrong_question_type",
    "missing_fields", "bad_types",
    "out_of_range",
    "permutation", "omission_1", "omission_2", "omission_3",
    "over_application", "direction_flip_one", "direction_flip_all",
    "direction_confusion", "included_t0", "excluded_t0", "condition_flip",
    "boundary_error", "sign_error", "sign_error_last",
    "transfer_binding_other", "binding_person",
    "temporal_initial", "temporal_off_by_one", "temporal_other",
    "referent_adjacent", "comparison_error",
    "invariant_violation", "conservation_violation",
    "digit_error", "off_by_small", "unexplained",
]
FORMAT_STATUSES = {
    "empty": "refusal_or_truncation", "refusal": "refusal_or_truncation",
    "truncated": "refusal_or_truncation", "no_json_found": "format_error",
    "invalid_json": "format_error", "missing": "format_error",
    "wrong_question_type": "wrong_question_type",
    "missing_fields": "missing_fields", "bad_types": "bad_types",
}
