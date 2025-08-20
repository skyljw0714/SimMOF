import os

from openai import OpenAI

OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)

WORKFLOW_PROMPT_TEMPLATE = """
You are an expert in computational materials screening.

A researcher is interested in identifying the best metal-organic frameworks (MOFs) for the following application:

🧪 Application Goal:
"{query}"

A wide range of simulation and analysis tools are available for this screening task, including:

- MOFChecker: for filtering out MOFs with structural issues
- ASE: for extracting atomic species (such as types of metal), coordinates, and unit cell information
- MOFid: for representing nodes and linkers as SMILES strings, enabling analysis of topological patterns and linker chemistry (e.g., functional groups) via substructure matching.
- MOFSimplify: for predicting solvent removal and thermal stability using ML models
- zeo++: for calculating pore diameter, pore volume, and surface area
- RASPA: for evaluating gas adsorption properties via GCMC
- LAMMPS: for simulating diffusion, permeability, and mechanical properties via MD
- VASP: for performing DFT-based structure optimization, electronic structure, and reactivity analysis
- Gaussian: for cluster-level optimization and electronic structure analysis
- MLIP: for fast but less accurate MD and geometry optimization

Please write a **short and concise** step-by-step strategy to efficiently identify the most promising MOF candidates for the application above using an optimal combination of the tools listed.

Your workflow should minimize computational cost while maintaining prediction accuracy. It should include:
- Initial filtering steps
- Use of surrogate models or precomputed data where possible
- Application of higher-fidelity methods in later stages

Present your answer as a numbered list (4–6 steps) without any extra commentary or code.
"""

def generate_workflow(query: str, model="gpt-4.1"):
    
    full_prompt = WORKFLOW_PROMPT_TEMPLATE.format(query=query)

    response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=1.0
        )

    output = response.choices[0].message.content.strip()
    return output

def main():
    query = input("📝 Enter your simulation goal: ")
    workflow = generate_workflow(query)
    print("\n📋 Generated Workflow:\n")
    print(workflow)

    save_dir = "llm_workflows"
    os.makedirs(save_dir, exist_ok=True)

    file_path = os.path.join(save_dir, "workflow_screening.txt")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("🧪 Research Goal:\n")
        f.write(query.strip() + "\n\n")
        f.write("📋 Generated Workflow:\n\n")
        f.write(workflow.strip())

    print(f"\n💾 Workflow saved to: {file_path}")
if __name__ == "__main__":
    main()
