"""python -m pytest -q test_data.py    (after generate.py; DATA_DIR=results by default)"""
import json, os, re, subprocess, sys
import pytest
import yaml
from num2words import num2words

DATA = os.environ.get("DATA_DIR", "results")
CFG = yaml.safe_load(open("config.yaml"))
CUES = ["friends", "between them", "altogether", "all together", "in total", "everyone", "each of"]
WORD_TO_NUM = {num2words(v): v for v in range(0, 200)}


@pytest.fixture(scope="module")
def stories():
    return [json.loads(l) for l in open(os.path.join(DATA, "data.jsonl"))]


@pytest.fixture(scope="module")
def prompts():
    return [json.loads(l) for l in open(os.path.join(DATA, "prompts.jsonl"))]


def numbers_in(text):
    out = [int(d) for d in re.findall(r"\d+", text)]
    for w in re.findall(r"[a-z]+(?:-[a-z]+)?", text.lower()):
        if w in WORD_TO_NUM:
            out.append(WORD_TO_NUM[w])
    return out


def test_conservation_and_no_negative(stories):
    for s in stories:
        table = s["holdings_by_timestep"]
        assert len(table) == s["T"] + 1 and table[0] == s["initial_holdings"]
        assert min(s["initial_holdings"]) >= 1
        for step in table:
            assert sum(step) == s["total"] and min(step) >= 0, s["story_id"]


def test_one_transfer_per_timestep(stories):
    for s in stories:
        assert len(s["transfers"]) == s["T"]
        hold = dict(zip(s["names"], s["initial_holdings"]))
        for k, (step, giver, receiver, amount) in enumerate(s["transfers"], 1):
            assert step == k and giver != receiver and 1 <= amount <= hold[giver]
            hold[giver] -= amount; hold[receiver] += amount
            assert list(hold.values()) == s["holdings_by_timestep"][k]


def test_text_never_states_the_total_or_the_count(stories):
    """Every number in the text is a timestep label, a holding or a transfer
    amount. The total is never one of them: with N >= 5 people each holding at
    least 1, no single holding or amount can reach it."""
    for s in stories:
        allowed = sorted(list(range(s["T"] + 1)) + s["initial_holdings"] + [t[3] for t in s["transfers"]])
        assert sorted(numbers_in(s["story"])) == allowed, s["story_id"]
        assert s["total"] not in s["initial_holdings"] and s["N"] not in (s["total"],)
        low = s["story"].lower()
        assert not any(c in low for c in CUES), s["story_id"]


def test_totals_cover_the_range(stories):
    tot = [s["total"] for s in stories]
    assert min(tot) >= CFG["total_min"] and max(tot) <= CFG["total_max"]
    if len(stories) >= 500:                       # a pilot need not hit every value
        assert len(set(tot)) >= 0.8 * (CFG["total_max"] - CFG["total_min"] + 1)
    for v in CFG["held_out_totals"]:
        assert CFG["total_min"] <= v <= CFG["total_max"]
    assert CFG["total_min"] < CFG["extrapolation_train_max"] < CFG["total_max"]


def test_names(stories):
    for s in stories:
        names = s["names"]
        assert len(names) == s["N"] == len(set(names))
        assert len({n[0] for n in names}) == len(names), names
        assert not any(a != b and a in b for a in names for b in names), names
        assert all(n in s["story"] for n in names)


def test_anchors_point_where_they_should(prompts, stories):
    """The four character offsets must land on the sentence ends they name."""
    by_id = {s["story_id"]: s for s in stories}
    for p in prompts[:200]:
        s, a, text = by_id[p["story_id"]], p["anchors"], p["prompt"]
        assert text.endswith("Answer: ") and a["answer"] == len(text)
        assert text[:a["allocation_end"]].endswith("pencils.") or text[:a["allocation_end"]].endswith("pencil.")
        assert text[:a["last_transfer"]].endswith(s["story"][-30:])
        assert text[:a["question_end"]].endswith("in total?")
        if s["T"] == 0:
            assert a["allocation_end"] == a["last_transfer"]
        else:
            assert a["allocation_end"] < a["last_transfer"] < a["question_end"] < a["answer"]


def test_splits_and_determinism(stories, prompts, tmp_path):
    frac = {k: sum(s["split"] == k for s in stories) / len(stories) for k in CFG["splits"]}
    assert all(abs(frac[k] - v) < 0.03 for k, v in CFG["splits"].items()), frac
    split = {s["story_id"]: s["split"] for s in stories}
    assert all(p["split"] == split[p["story_id"]] for p in prompts) and len(prompts) == len(stories)
    subprocess.run([sys.executable, "generate.py", "--n", "20", "--out", str(tmp_path)], check=True)
    assert [json.loads(l) for l in open(tmp_path / "data.jsonl")] == stories[:20]


def test_answer_tokens(prompts):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CFG["models"][os.environ.get("MODEL", CFG["model"])]["path"])
    for p in prompts[:20]:
        base = tok(p["prompt"], add_special_tokens=False).input_ids
        for v in range(CFG["total_min"], CFG["total_max"] + 1):
            ext = tok(p["prompt"] + str(v), add_special_tokens=False).input_ids
            assert ext[:len(base)] == base, v          # appending never re-tokenises the prompt
            assert 1 <= len(ext) - len(base) <= 2, (v, len(ext) - len(base))
