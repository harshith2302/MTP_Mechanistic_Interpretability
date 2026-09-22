"""Stories and prompts.   python generate.py [--n 500] [--out results]

Story wording, transfer verbs, number-word mixing and the name list are those of
pencil_taxonomy_simple; the one change is that the opening sentence no longer
states how many people there are. Writes <out>/data.jsonl (one story per line,
with the holdings table at every timestep) and <out>/prompts.jsonl (two prompts
per story). Deterministic from config.yaml.
"""
import argparse, hashlib, json, os, random
import yaml
from num2words import num2words

NAMES = """Arjun Beatrix Chandra Dmitri Eleanor Farhan Gwendolyn Hiroshi Ingrid Jamal Kavitha
Lorenzo Mateo Nadia Oleg Priya Quentin Rosalind Sanjay Theodora Ulrich Valentina Wesley Ximena
Yusuf Zubeida Anika Bartholomew Clementine Desmond Esperanza Fionnuala Gustavo Harriet Isadora
Joaquin Katarina Leopold Marguerite Nikolai""".split()
TRANSFER = ["{A} gave {n} {pencils} to {B}.", "{A} handed {n} {pencils} over to {B}.",
            "{A} transferred {n} {pencils} to {B}.", "{A} donated {n} {pencils} to {B}.",
            "{A} passed {n} {pencils} to {B}.", "{A} sent {n} {pencils} to {B}.",
            "{A} dropped {n} {pencils} into {B}'s bag.", "{A} let {B} have {n} {pencils}."]
QUESTIONS = {"people": "How many people are participating in the pencil exchange?",
             "total": "How many pencils are there in total?"}


def names_ok(names):
    """No two share a first letter, none is a substring of another."""
    return (len({n[0] for n in names}) == len(names)
            and not any(a != b and a in b for a in names for b in names))


def partition(rng, total, parts):
    """Random composition of `total` into `parts` positive integers (every holding >= 1)."""
    cuts = sorted(rng.sample(range(1, total), parts - 1))
    return [b - a for a, b in zip([0] + cuts, cuts + [total])]


def make_story(story_id, n, total, t, cfg, rng):
    def num(v):
        return num2words(v) if rng.random() < cfg["number_word_probability"] else str(v)
    def pencils(v):
        return "pencil" if v == 1 else "pencils"
    while True:
        names = rng.sample(NAMES, n)
        if names_ok(names):
            break
    hold = partition(rng, total, n)
    table, transfers, sentences = [list(hold)], [], []
    for step in range(1, t + 1):                              # exactly one transfer per timestep
        giver = rng.choice([i for i in range(n) if hold[i] > 0])
        receiver = rng.choice([i for i in range(n) if i != giver])
        amount = rng.randint(1, hold[giver])                  # at most what the giver holds
        hold[giver] -= amount; hold[receiver] += amount
        assert sum(hold) == total and min(hold) >= 0
        table.append(list(hold)); transfers.append([step, names[giver], names[receiver], amount])
        sentences.append(f"At timestep {step}, " + rng.choice(TRANSFER).format(
            A=names[giver], B=names[receiver], n=num(amount), pencils=pencils(amount)))
    holdings = [f"{nm} holds {num(h)} {pencils(h)}" for nm, h in zip(names, table[0])]
    intro = (f"The following people are participating in the pencil exchange: "
             f"{', '.join(names[:-1])} and {names[-1]}. At timestep 0, their initial pencil "
             f"holdings are as follows: {', '.join(holdings[:-1])} and {holdings[-1]}.")
    return {"story_id": story_id, "N": n, "total": total, "T": t, "names": names,
            "initial_holdings": table[0], "transfers": transfers, "holdings_by_timestep": table,
            "story": " ".join([intro] + sentences), "split": split_of(story_id, cfg)}


def split_of(story_id, cfg):
    x = int(hashlib.sha256(f"{cfg['split_seed']}|{story_id}".encode()).hexdigest(), 16) / 16 ** 64
    s = cfg["splits"]
    return "train" if x < s["train"] else "val" if x < s["train"] + s["val"] else "test"


def render(story, task, tok, cfg):
    """Prompt text ending exactly where the answer token goes (trailing space included)."""
    user = f"{story}\nQuestion: {QUESTIONS[task]}"
    if cfg["use_chat_template"]:
        return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                       add_generation_prompt=True) + "Answer: "
    return user + "\nAnswer: "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="number of stories (default: config)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--model", default=None, help="key in config models (default: config model)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open("config.yaml"))
    mcfg = cfg["models"][a.model or cfg["model"]]
    rng = random.Random(cfg["seed"])
    os.makedirs(a.out, exist_ok=True)
    tok = None
    if cfg["use_chat_template"]:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(mcfg["path"])
    stories, prompts = [], []
    for i in range(a.n or cfg["n_stories"]):
        n, total, t = rng.choice(cfg["n_people"]), rng.choice(cfg["totals"]), rng.choice(cfg["timesteps"])
        s = make_story(f"s{i:05d}", n, total, t, cfg, rng)
        stories.append(s)
        for task in QUESTIONS:
            prompts.append({"prompt_id": f"{s['story_id']}_{task}", "story_id": s["story_id"], "task": task,
                            "gold": s["N"] if task == "people" else s["total"], "N": s["N"],
                            "total": s["total"], "T": s["T"], "split": s["split"],
                            "prompt": render(s["story"], task, tok, cfg)})
    with open(os.path.join(a.out, "data.jsonl"), "w") as f:
        f.writelines(json.dumps(s) + "\n" for s in stories)
    with open(os.path.join(a.out, "prompts.jsonl"), "w") as f:
        f.writelines(json.dumps(p) + "\n" for p in prompts)
    from collections import Counter
    print(f"[generate] {len(stories)} stories, {len(prompts)} prompts -> {a.out}/ ; "
          f"splits {dict(Counter(s['split'] for s in stories))}")


if __name__ == "__main__":
    main()
