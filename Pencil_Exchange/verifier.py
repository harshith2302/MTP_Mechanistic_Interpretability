import json

# Paste the LLM's JSON answer here
llm_answer_json = '''
{
  "question_type": "duration_condition",
  "person": "Hanuman",
  "condition": "<10",
  "answer": 91
}
'''
# Parse and load JSON
try:
    llm_answer = json.loads(llm_answer_json)
except json.JSONDecodeError as e:
    print(f"ERROR: LLM response is not valid JSON. Details: {e}")
    exit()

with open("ground_truth_pencil_counts.json") as f:
    ground_truth_file = json.load(f)

pencil_count_matrix = ground_truth_file["ground_truth_pencil_counts"]

# Name <-> ID mappings 
name_to_id = ground_truth_file["name_to_id"]   
id_to_name = ground_truth_file["id_to_name"]   

def get_person_id(name_or_id):
    if name_or_id in name_to_id:
        return name_to_id[name_or_id]  
    if name_or_id in id_to_name:
        return name_or_id               
    return None                         

# Helper: parse a condition like ">=5" into operator and number
def split_condition(condition_string):
    for operator in (">=", "<=", ">", "<"):
        if condition_string.startswith(operator):
            number_part = condition_string[len(operator):]
            try:
                return operator, int(number_part)
            except ValueError:
                return None, None
    return None, None


# Helper: check whether a pencil count satisfies a condition

def satisfies_condition(pencil_count, operator, threshold):
    if operator == ">=": return pencil_count >= threshold
    if operator == "<=": return pencil_count <= threshold
    if operator == ">":  return pencil_count >  threshold
    if operator == "<":  return pencil_count <  threshold


# Validate that the LLM gave all required fields

required_fields_per_type = {
    "person_timestep_lookup": ["person", "timestep", "answer"],
    "person_value_timesteps": ["person", "pencil_count", "timesteps"],
    "population_condition"  : ["timestep", "condition", "answer"],
    "duration_condition"    : ["person", "condition", "answer"],
}

question_type = llm_answer.get("question_type")

if question_type not in required_fields_per_type:
    print(f"ERROR: Unknown question_type '{question_type}'.")
    print(f"Valid types: {list(required_fields_per_type.keys())}")
    exit()

missing_fields = [
    field for field in required_fields_per_type[question_type]
    if field not in llm_answer
]
if missing_fields:
    print(f"ERROR: LLM answer is missing these fields: {missing_fields}")
    exit()


# Verify the answer based on question type
answer_is_correct = False

# Type 1: How many pencils did <person> have at timestep <T>? 
if question_type == "person_timestep_lookup":

    person_id       = get_person_id(llm_answer["person"])
    timestep_number = int(llm_answer["timestep"])
    timestep_key    = f"t={timestep_number}"
    correct_count   = pencil_count_matrix[timestep_key][person_id]
    llm_count       = int(llm_answer["answer"])

    print(f"Person        : {llm_answer['person']} ({person_id})")
    print(f"Timestep      : {timestep_number}")
    print(f"Correct count : {correct_count}")
    print(f"LLM answer    : {llm_count}")
    answer_is_correct = correct_count == llm_count

# Type 2: At which timesteps did <person> have exactly <N> pencils?
elif question_type == "person_value_timesteps":

    person_id           = get_person_id(llm_answer["person"])
    target_pencil_count = int(llm_answer["pencil_count"])

    correct_timesteps = sorted(
        int(key.split("=")[1])
        for key, counts in pencil_count_matrix.items()
        if counts[person_id] == target_pencil_count
    )
    llm_timesteps = sorted(int(t) for t in llm_answer["timesteps"])

    print(f"Person            : {llm_answer['person']} ({person_id})")
    print(f"Pencil count      : {target_pencil_count}")
    print(f"Correct timesteps : {correct_timesteps}")
    print(f"LLM timesteps     : {llm_timesteps}")
    answer_is_correct = correct_timesteps == llm_timesteps

# Type 3: At timestep <T>, how many people had <condition> pencils? 
elif question_type == "population_condition":

    timestep_number     = int(llm_answer["timestep"])
    timestep_key        = f"t={timestep_number}"
    operator, threshold = split_condition(llm_answer["condition"])

    correct_people_count = sum(
        satisfies_condition(count, operator, threshold)
        for count in pencil_count_matrix[timestep_key].values()
    )
    llm_people_count = int(llm_answer["answer"])

    print(f"Timestep              : {timestep_number}")
    print(f"Condition             : {llm_answer['condition']}")
    print(f"Correct people count  : {correct_people_count}")
    print(f"LLM answer            : {llm_people_count}")
    answer_is_correct = correct_people_count == llm_people_count

# Type 4: During how many timesteps did <person> have <condition> pencils?
elif question_type == "duration_condition":

    person_id           = get_person_id(llm_answer["person"])
    operator, threshold = split_condition(llm_answer["condition"])
    active_timestep_keys = [
        key for key in pencil_count_matrix
        if int(key.split("=")[1]) >= 1
    ]
    correct_timestep_count = sum(
        satisfies_condition(pencil_count_matrix[key][person_id], operator, threshold)
        for key in active_timestep_keys
    )
    llm_timestep_count = int(llm_answer["answer"])

    print(f"Person                  : {llm_answer['person']} ({person_id})")
    print(f"Condition               : {llm_answer['condition']}  (t=1..T only)")
    print(f"Correct timestep count  : {correct_timestep_count}")
    print(f"LLM answer              : {llm_timestep_count}")
    answer_is_correct = correct_timestep_count == llm_timestep_count

# STEP 8 — Print final result
print("\nVerification Result:", "Correct Answer" if answer_is_correct else "Incorrect Answer")