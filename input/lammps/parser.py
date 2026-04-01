import json
import re
from config import OPENAI_MODEL_PARSER, TRAPPE_DICT_FILE, get_openai_client

def parse_query_with_llm(query: str) -> dict:
    prompt = f"""
You are an assistant that converts natural language simulation queries into JSON dictionaries 
for initializing a LAMMPSAgent. Extract the following fields:

- "MOF": the MOF structure (e.g., UiO-66, MOF-5, HKUST-1)
- "Guest": the guest molecule (e.g., CO2, H2O, CH4)
- "simulation_property": property to compute (e.g., diffusivity, adsorption, binding_energy)
- "Guest_num": number of guest molecules (integer, default=10)

Return ONLY a valid JSON object. No markdown, no code block, no explanation.
"""

    response = get_openai_client().chat.completions.create(
        model=OPENAI_MODEL_PARSER,
        messages=[
            {"role": "system", "content": "You are a molecular simulation assistant."},
            {"role": "user", "content": prompt + f"\n\nInput query:\n{query}"}
        ],
        temperature=0.0,
    )

    content = response.choices[0].message.content.strip()

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        content = match.group(0)

    try:
        query_dict = json.loads(content)
    except json.JSONDecodeError:
        raise ValueError(f"LLM output is not valid JSON:\n{content}")

    if "Guest_num" not in query_dict:
        query_dict["Guest_num"] = 10

    return query_dict


def match_trappe_abbreviation(molecule_name_origin: str,
                              dict_path: str = str(TRAPPE_DICT_FILE)) -> dict:

    
    with open(dict_path, 'r') as f:
        data = json.load(f)

    trappe_text = json.dumps(data, separators=(',', ':'), ensure_ascii=False)

    prompt = f""" 
Here is a TRAPPE abbreviation dictionary in JSON format (all in one line):

{trappe_text}

Given my molecule name "{molecule_name_origin}", please recommend the most appropriate TRAPPE abbreviation (key) from the dictionary.  
If there is no exact match, suggest the closest candidate.

Return ONLY the most appropriate abbreviation (key) as a single line string, with no explanation, value, or formatting.
"""

    response = get_openai_client().chat.completions.create(
        model=OPENAI_MODEL_PARSER,
        messages=[
            {"role": "system", "content": "You are a molecular simulation expert specializing in TRAPPE force field assignment for LAMMPS simulations."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
    )

    molecule_name = response.choices[0].message.content.strip()
    return molecule_name

def count_atoms_in_lt(file_path):
    atom_count = 0
    atoms = []
    inside_block = False

    with open(file_path, "r") as f:
        for line in f:
            stripped = line.strip()

            
            if stripped.startswith('write_once("Data Masses")'):
                inside_block = True
                continue

            
            if inside_block and stripped.startswith("}"):
                inside_block = False
                continue

            
            if inside_block and stripped.startswith("@atom:"):
                atom_count += 1
                atoms.append(stripped)

    return atom_count, atoms


def make_group_commands(mof_lt, guest_lt):
    
    mof_count, mof_atoms = count_atoms_in_lt(mof_lt)
    
    guest_count, guest_atoms = count_atoms_in_lt(guest_lt)

    
    mof_types = list(range(1, mof_count + 1))
    guest_types = list(range(mof_count + 1, mof_count + guest_count + 1))

    
    group_cmds = []
    if mof_types:
        group_cmds.append(f"group MOF type {' '.join(map(str, mof_types))}")
    if guest_types:
        group_cmds.append(f"group guest type {' '.join(map(str, guest_types))}")

    return mof_atoms, guest_atoms, group_cmds










