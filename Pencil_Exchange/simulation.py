import random
import json
from num2words import num2words

# Configuration
NUM_PEOPLE = 10
NUM_TIMESTEPS = 10
TOTAL_PENCIL_COUNT= NUM_PEOPLE * 5   
NUMBER_WORD_PROBABILITY = 0.45
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

def load_name_bag(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def sample_display_names(name_bag, num_people):
    if num_people > len(name_bag):
        raise ValueError(
            f"Name bag has only {len(name_bag)} names, "
            f"but {num_people} were requested."
        )
    return random.sample(name_bag, num_people)

name_bag      = load_name_bag("names.txt")
display_names = sample_display_names(name_bag, NUM_PEOPLE)

person_ids = [f"A{i}" for i in range(1, NUM_PEOPLE + 1)]
id_to_name = {pid: name for pid, name in zip(person_ids, display_names)}
name_to_id = {name: pid for pid, name in id_to_name.items()}


# CLuster Generation  
def create_person_clusters(ids):
    n = len(ids)
    if n < 4:
        return [ids[:]]
    max_possible_splits = n // 2 - 1
    num_splits = random.randint(1, max(1, max_possible_splits))
    split_positions = sorted(
        random.sample(range(2, n - 1), min(num_splits, n - 3))
    )
    clusters, prev = [], 0
    for s in split_positions:
        clusters.append(ids[prev:s])
        prev = s
    clusters.append(ids[prev:])
    return clusters

person_clusters = create_person_clusters(person_ids)

# Initial distribution 
def random_integer_partition(total_value, parts):
    if total_value < parts:
        raise ValueError(
            f"Cannot distribute {total_value} pencils among {parts} people: "
            f"each person needs at least 1. Increase TOTAL_PENCIL_COUNT."
        )
    cut_positions = sorted(random.sample(range(1, total_value), parts - 1))
    partitions, prev = [], 0
    for cut in cut_positions:
        partitions.append(cut - prev)
        prev = cut
    partitions.append(total_value - prev)
    return partitions

initial_distribution = random_integer_partition(TOTAL_PENCIL_COUNT, NUM_PEOPLE)
current_pencil_distribution = {
    person_ids[i]: initial_distribution[i] for i in range(NUM_PEOPLE)
}

# Number To Word 
def format_number_token(value):
    return num2words(value) if random.random() < NUMBER_WORD_PROBABILITY else str(value)

def pencil_word(count):
    return "pencil" if count == 1 else "pencils"

TRANSFER_SENTENCE_TEMPLATES = [
    "{A} gave {n} {pencils} to {B}.",
    "{A} handed {n} {pencils} over to {B}.",
    "{A} transferred {n} {pencils} to {B}.",
    "{A} donated {n} {pencils} to {B}.",
    "{A} passed {n} {pencils} to {B}.",
    "{A} sent {n} {pencils} to {B}.",
    "{A} dropped {n} {pencils} into {B}'s bag.",
    "{A} let {B} have {n} {pencils}.",
    # "{A} forwarded {n} {pencils} to {B}.",
    # "{A} relinquished {n} {pencils} to {B}.",
    # "{A} gave away {n} {pencils} to {B}.",
    # "{A} offered {n} {pencils} to {B}.",
    # "{A} deposited {n} {pencils} with {B}.",
    # "{A} slid {n} {pencils} across to {B}.",
    # "{A} contributed {n} {pencils} to {B}.",
    # "{A} entrusted {n} {pencils} to {B}.",
    # "{A} delivered {n} {pencils} to {B}.",
    # "{A} handed off {n} {pencils} to {B}.",
    # "{A} gave a messenger {n} {pencils} to deposit with {B}.",
    # "{A} set aside {n} {pencils} for {B}.",
    # "{A} pushed {n} {pencils} towards {B}.",
    # "{A} parcelled out {n} {pencils} to {B}.",
    # "{B} received {n} {pencils} from {A}.",
    # "{B} took {n} {pencils} from {A}.",
    # "{B} collected {n} {pencils} from {A}.",
    # "{B} acquired {n} {pencils} from {A}.",
    # "{B} accepted {n} {pencils} from {A}.",
    # "{B} obtained {n} {pencils} from {A}.",
    # "{B} picked up {n} {pencils} from {A}.",
    # "{B} walked away with {n} {pencils} from {A}.",
]

def generate_transfer_sentence(giver_id, receiver_id, amount):
    template = random.choice(TRANSFER_SENTENCE_TEMPLATES)
    return template.format(
        A=id_to_name[giver_id],
        B=id_to_name[receiver_id],
        n=format_number_token(amount),
        pencils=pencil_word(amount),
    )

def build_timestep_description(timestep, events):
    if not events:
        return ""
    sentences = [generate_transfer_sentence(g, r, amt) for (g, r, amt) in events]
    return f"At timestep {timestep}, " + " ".join(sentences)

#simulation
ground_truth_pencil_counts = {0: dict(current_pencil_distribution)}

transfer_events_by_cluster = [
    [[] for _ in range(NUM_TIMESTEPS + 1)]
    for _ in person_clusters
]

for timestep in range(1, NUM_TIMESTEPS + 1):
    updated_distribution = dict(current_pencil_distribution)

    for cluster_index, cluster_members in enumerate(person_clusters):
        local = {p: updated_distribution[p] for p in cluster_members}
        num_transfers = random.randint(1, len(cluster_members))

        for _ in range(num_transfers):
            possible_givers = [p for p in cluster_members if local[p] > 0]
            if not possible_givers:
                break
            giver = random.choice(possible_givers)
            possible_receivers = [p for p in cluster_members if p != giver]
            if not possible_receivers:
                break
            receiver = random.choice(possible_receivers)
            amount = random.randint(1, local[giver])
            local[giver]    -= amount
            local[receiver] += amount
            transfer_events_by_cluster[cluster_index][timestep].append(
                (giver, receiver, amount)
            )

        for p in cluster_members:
            updated_distribution[p] = local[p]

    current_pencil_distribution = updated_distribution
    ground_truth_pencil_counts[timestep] = dict(current_pencil_distribution)

# Conservation assertion
for t in range(NUM_TIMESTEPS + 1):
    total = sum(ground_truth_pencil_counts[t].values())
    assert total == TOTAL_PENCIL_COUNT, (
        f"Conservation broken at t={t}: found {total}, expected {TOTAL_PENCIL_COUNT}."
    )

schedulable_nodes  = [(ci, 1) for ci in range(len(person_clusters))]
execution_schedule = []

while schedulable_nodes:
    selected = random.choice(schedulable_nodes)
    schedulable_nodes.remove(selected)
    execution_schedule.append(selected)
    cluster_index, timestep = selected
    if timestep + 1 <= NUM_TIMESTEPS:
        schedulable_nodes.append((cluster_index, timestep + 1))

# Text generation
timestep_paragraphs = []
for cluster_index, timestep in execution_schedule:
    cluster_events = transfer_events_by_cluster[cluster_index][timestep]
    if cluster_events:
        timestep_paragraphs.append(
            build_timestep_description(timestep, cluster_events)
        )

names_text = (
    ", ".join(id_to_name[p] for p in person_ids[:-1])
    + " and " + id_to_name[person_ids[-1]]
)
initial_state_sentences = [
    f"{id_to_name[p]} holds "
    f"{format_number_token(ground_truth_pencil_counts[0][p])} "
    f"{pencil_word(ground_truth_pencil_counts[0][p])}"
    for p in person_ids
]
initial_state_text = (
    ", ".join(initial_state_sentences[:-1]) + " and " + initial_state_sentences[-1]
)
intro_text = (
    f"There are {format_number_token(NUM_PEOPLE)} people participating "
    f"in the pencil exchange: {names_text}. "
    f"At timestep 0, their initial pencil holdings are as follows: "
    f"{initial_state_text}."
)
full_text_description = intro_text + " " + " ".join(timestep_paragraphs)


# Save outputs
ground_truth_output = {
    "config": {
        "num_people"   : NUM_PEOPLE,
        "num_timesteps": NUM_TIMESTEPS,
        "total_pencils": TOTAL_PENCIL_COUNT,
    },
    "persons"   : person_ids,
    "id_to_name": id_to_name,
    "name_to_id": name_to_id,
    "clusters"  : {
        f"cluster_{i+1}": cluster
        for i, cluster in enumerate(person_clusters)
    },
    "ground_truth_pencil_counts": {
        f"t={t}": ground_truth_pencil_counts[t]
        for t in range(NUM_TIMESTEPS + 1)
    },
}
with open("ground_truth_pencil_counts.json", "w") as f:
    json.dump(ground_truth_output, f, indent=2)
with open("pencil_text_description.txt", "w") as f:
    f.write(full_text_description.replace("\n", " "))
print("Saved ground_truth_pencil_counts.json")
print("Saved pencil_text_description.txt")