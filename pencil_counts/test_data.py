"""python -m pytest -q test_data.py    (after generate.py; DATA_DIR=results by default)"""
import json, os, re, subprocess, sys
import pytest
import yaml
from num2words import num2words

DATA = os.environ.get("DATA_DIR", "results")
CFG = yaml.safe_load(open("config.yaml"))
CUES = ["friends", "between them", "altogether", "all together", "in total", "everyone", "each of"]
WORD_TO_NUM = {num2words(v): v for v in range(0, 60)}


@pytest.fixture(scope="module")
def stories():
    return [json.loads(l) for l in open(os.path.join(DATA, "data.jsonl"))]


@pytest.fixture(scope="module")
def prompts():
    return [json.loads(l) for l in open(os.path.join(DATA, "prompts.jsonl"))]


def numbers_in(text):
    """Every number in the text, digits or words, as ints."""
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
            assert step == k and giver != receiver
            assert 1 <= amount <= hold[giver], "a transfer moves at most what the giver holds"
            hold[giver] -= amount; hold[receiver] += amount
            assert list(hold.values()) == s["holdings_by_timestep"][k]
        assert s["story"].count("At timestep") == s["T"] + 1


def test_text_never_states_n_or_total(stories):
    """Every number in the text is a timestep label, an initial holding or a
    transfer amount -- nothing else, so neither N nor the total is ever stated."""
    for s in stories:
        allowed = sorted(list(range(s["T"] + 1)) + s["initial_holdings"] + [tr[3] for tr in s["transfers"]])
        assert sorted(numbers_in(s["story"])) == allowed, s["story"]
        low = s["story"].lower()
        assert not any(c in low for c in CUES), s["story"]
        assert not re.search(r"\b(are|were)\s+\S+\s+people\b", low), s["story"]


def test_names(stories):
    for s in stories:
        names = s["names"]
        assert len(names) == s["N"] == len(set(names))
        assert len({n[0] for n in names}) == len(names), names
        assert not any(a != b and a in b for a in names for b in names), names
        assert all(n.isalpha() for n in names) and all(n in s["story"] for n in names)


def test_number_words_mixed(stories):
    text = " ".join(s["story"] for s in stories)
    digits, words = len(re.findall(r"holds \d", text)), len(re.findall(r"holds [a-z]", text))
    assert abs(words / (digits + words) - CFG["number_word_probability"]) < 0.05


def test_splits_by_story(stories, prompts):
    frac = {k: sum(s["split"] == k for s in stories) / len(stories) for k in CFG["splits"]}
    assert all(abs(frac[k] - v) < 0.03 for k, v in CFG["splits"].items()), frac
    split = {s["story_id"]: s["split"] for s in stories}
    assert all(p["split"] == split[p["story_id"]] for p in prompts) and len(prompts) == 2 * len(stories)


def test_deterministic(stories, tmp_path):
    subprocess.run([sys.executable, "generate.py", "--n", "20", "--out", str(tmp_path)], check=True)
    assert [json.loads(l) for l in open(tmp_path / "data.jsonl")] == stories[:20]


def test_answer_tokens(prompts):
    """Appending an answer must extend the prompt, never re-tokenise it. The people
    answers are one token for every model here; Qwen and Mistral write the totals as
    two, which is why the model's own answer is scored by summing token log-probs."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CFG["models"][os.environ.get("MODEL", CFG["model"])]["path"])
    for p in prompts[:40]:
        assert p["prompt"].endswith("Answer: ")
        base = tok(p["prompt"], add_special_tokens=False).input_ids
        for ans in CFG["n_people"] + CFG["totals"]:
            ext = tok(p["prompt"] + str(ans), add_special_tokens=False).input_ids
            extra = len(ext) - len(base)
            assert ext[:len(base)] == base, (ans, tok.convert_ids_to_tokens(ext[-3:]))
            assert extra == 1 or (ans in CFG["totals"] and extra == 2), (ans, extra)
        for ans in CFG["n_people"]:
            assert len(tok(p["prompt"] + str(ans), add_special_tokens=False).input_ids) == len(base) + 1
