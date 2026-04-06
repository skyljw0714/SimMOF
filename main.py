import json
from query.agent import analyze_mof_query
from working.agent import WorkingAgent
from Zeopp.agent import ZeoppAgent
from LAMMPS.agent import LAMMPSAgent
from response.agent import ResponseAgent
from RASPA.agent import RASPAAgent
from VASP.agent import VASPAgent
from screening.agent import ScreeningAgent
from analysis.agent import AnalysisAgent
from langchain.schema import HumanMessage, SystemMessage
from config import LLM_DEFAULT


def main():
    agents = {
        "ZeoppAgent": ZeoppAgent(),
        "LAMMPSAgent": LAMMPSAgent(),
        "ResponseAgent": ResponseAgent(),
        "RASPAAgent": RASPAAgent(),
        "VASPAgent": VASPAgent(),
        "ScreeningAgent": ScreeningAgent(),
        "AnalysisAgent": AnalysisAgent(),
    }

    
    user_queries = [
        ################# Additional queries ###################
        # "I want to calculate surface area of UiO-66 and ZIF-8",
        # "What is CO2 uptake value of UiO-66 and HKUST-1 in 298K",
        # "I want to calculate the binding energy of CO2 in HKUST-1 and ZIF-8",

        # "For CO₂ adsorption in HKUST-1, compute multiple binding configurations and identify the most stable adsorption site.",
        # "For UiO-66, compute CO₂ adsorption at 298 K and 1 bar, then compute CO₂ diffusivity at the corresponding loading, and analyze whether higher loading reduces diffusivity."

        # "What is the CO2 uptake of HKUST-1 at 298 K and 1 bar?",
        # "What is the argon uptake of HKUST-1 at 298 K and 1 bar?",
        # "Determine the CO2 adsorption isotherm for HKUST-1 at 298 K from 0.01 to 10 bar using 8–12 pressure points.",
        # "Calculate the CO2 uptake of HKUST-1 at 1 bar for temperatures 273 K, 298 K, and 323 K.",
        # "Compare the adsorption of CO2 and N2 in HKUST-1 at 298 K and 1 bar.",
        # "Estimate the Henry coefficient of CO2 in HKUST-1 at 298 K.",
        # "Compute the isosteric heat (Qst) of CO2 adsorption in HKUST-1 at 298 K at pressures 0.01, 0.03, 0.1, 0.3, 1, and 10 bar."
        # "Calculate the selectivity of a CO2/N2 mixture (15/85) in HKUST-1 at 298 K and 1 bar.",
        # "Evaluate the working capacity of CO2 in HKUST-1 between 0.15 bar and 1 bar at 298 K.",
        # "Evaluate adsorption of a CO2/N2/H2O mixture in HKUST-1 at 298 K and 1 bar, with gas composition CO2/N2/H2O = 15/84/1.",
        # "Compare CO2 adsorption in HKUST-1 using two different CO2 force field models. (EPM2 and TraPPE)",

        # "Attempt a NEB calculation for CO2 migration between two adsorption sites in HKUST-1.",
        # "Compute the charge density difference for CO2 adsorption in HKUST-1.",
        # "Calculate the vibrational frequencies of CO2 adsorbed in HKUST-1.",
        # "Perform an ab initio DFT calculation to determine the adsorption energy of CO2 in HKUST-1.",
        # "Compare the adsorption energy of CO2 in HKUST-1 using different exchange–correlation functionals (PBE, PBE+D3, and HSE06).",

        ############## Simple property queries ################
        # "I want to calculate surface area of UiO-66",
        # "What is the pore volume of HKUST-1?",
        # "which MOF has the larger pore volume between HKUST-1 and ZIF-8",

        # "What is the CO2 uptake of ZIF-8 at 298 K and 1 bar?",
        # "Calculate CH4 uptake in HKUST-1 at 1 bar."

        # "I want to calculate diffusivity of CO2 in HKUST-1",
        # "I want to calculate diffusivity of CO2 in UiO-66 and ZIF-8",
        # "Calculate mean squared displacement of CH4 in UiO-66",
        # "I want to calculate the Henry constant of CO2 in UiO-66 at 298 K"

        "I want to calculate the binding energy of CO2 in ZIF-8",
        # "Compute the adsorption energy of H2 on HKUST-1",


        ############### Not-mentioned properties queries ################
        # "Calculate the heat of adsorption of N2 in UiO-66",
        # "Calculate the selectivity of CO2 and N2 in ZIF-8 at 298K",

        # "Calculate the largest cavity diameter of UiO-66",
        # "Compute the pore limiting diameter of HKUST-1",
        # "Calculate the accessible pore volume of ZIF-8",

        # "Calculate self-diffusion coefficient of CH4 in UiO-66",
        # "Simulate thermal expansion of UiO-66",

        # "Compute the optimized crystal structure of UiO-66",
        # "Calculate the electronic density of states of HKUST-1",
        # "Calculate the band gap of UiO-66",



        ################ High-level complex queries ################
        # "I want to find top 10 MOFs with the highest H2 uptake at 1 bar and 298K in the database",
        # "I want to find top 10 MOFs with the highest CH4 uptake at 1 bar and 298K in the database",
        # "Compute the CO₂ binding energies for HKUST-1 and ZIF-8 and discuss why the two MOFs show different binding strengths.",
        # "Calculate pore volume and pld, then compute CO₂ diffusivity of ZIF-8 and UiO-66 and explain how pore structure affects diffusion.",
        # "Compute CH₄ diffusivity in MOF-5 at 200 K, 300 K, and 400 K, and analyze the temperature dependence",
        # "Compute CO₂ adsorption isotherms for UiO-66 from 0.01 to 10 bar, and analyze how the low-pressure region reflects the strength of host–guest interactions",
        # "For HKUST-1 and ZIF-8, compute CO₂ uptake and binding energies, and analyze whether higher uptake correlates with stronger binding",
        # "Compare CO₂ and CH₄ adsorption in ZIF-8 and explain the origin of selectivity.",
        # "For UiO-66, compute CO₂ adsorption at infinite dilution, then compute CO₂ diffusivity at the same thermodynamic limit, and analyze whether stronger adsorption necessarily implies slower transport.",
        # "For CO₂ adsorption in ZIF-8, compute binding energy and charge transfer, and discuss whether adsorption is physisorption or chemisorption.",
        # "Compute CO₂/N₂ selectivity in UiO-66 at 0.1 bar and 1 bar and analyze the pressure dependence.",


    ]

    for user_query in user_queries:
        print("\n========================================")
        print(f"User query: {user_query}")

        
        bundle = analyze_mof_query(user_query)
        if not bundle or not bundle.get("queries"):
            print("QueryAgent parsing failed.")
            continue

        parsed = bundle["queries"]
        analysis_enabled = bundle.get("analysis_enabled", False)
        simulation_input = bundle.get("simulation_input")

        clarification_round = 0
        current_query = user_query

        while bundle.get("needs_clarification") and clarification_round < 2:
            clarification_round += 1

            print("\n[Missing Required Information]")
            print(bundle.get("clarification_question", "More information is required to continue."))

            user_reply = input("Reply: ").strip()

            current_query = (
                current_query.strip()
                + "\n"
                + f"Additional user-provided conditions: {user_reply}"
            )

            print("\n[Re-parsing query with additional conditions...]")
            bundle = analyze_mof_query(current_query)

            if not bundle or not bundle.get("queries"):
                print("QueryAgent parsing failed after clarification.")
                break

        if not bundle or not bundle.get("queries"):
            continue

        parsed = bundle["queries"]
        analysis_enabled = bundle.get("analysis_enabled", False)
        simulation_input = bundle.get("simulation_input")
            
        simulation_input_status = bundle.get("simulation_input_status", "ok")
        simulation_input_message = bundle.get("simulation_input_message", "")

        if simulation_input_status == "needs_user_confirmation":
            print("\n[Simulation Input Review]")
            print(simulation_input_message)

            user_reply = input("Reply (KEEP / REGENERATE / correction): ").strip()

            if user_reply.upper() == "KEEP":
                print("[Simulation Input] keeping original user-provided input as-is.")

            elif user_reply.upper() == "REGENERATE":
                print("[Simulation Input] discarding user-provided input and falling back to fresh generation.")
                simulation_input = {"present": False, "snippets": []}

            else:
                print("[Simulation Input] patching user-provided input based on user reply...")
                simulation_input = patch_simulation_input_with_user_reply(
                    simulation_input=simulation_input,
                    user_reply=user_reply,
                    parsed_queries=parsed,
                    llm=LLM_DEFAULT,
                )

                print("\n[Patched simulation_input]")
                print(json.dumps(simulation_input, ensure_ascii=False, indent=2))

        print("\n[Parsed Queries]")
        print(json.dumps(parsed, ensure_ascii=False, indent=2))


        
        wa = WorkingAgent(parsed, analysis_enabled=analysis_enabled, agents=agents, simulation_input=simulation_input,)
        plans = wa.plan()

        print("\n[Planned Workflows]")
        for i, p in enumerate(plans, 1):
            print(f"\n--- Plan {i} ---")
            print(json.dumps(p.model_dump(), ensure_ascii=False, indent=2))

        results = wa.run()

def llm_patch_simulation_snippet(software: str, original_text: str, user_reply: str, parsed_queries: list, llm) -> str:
    system = """You are a careful editor for simulation input snippets.
Return ONLY the patched simulation input text.
No markdown. No explanation."""

    user = f"""Patch the original simulation input snippet according to the user's reply.

HARD RULES:
1) Preserve as much of the original text as possible.
2) Apply only changes clearly requested by the user.
3) Keep the result valid for the same software type.
4) If the user pasted a full corrected replacement input, use that as the result.
5) If the user instruction is ambiguous, make the smallest reasonable change.
6) Return ONLY the patched input text.

Software:
{software}

Parsed queries:
{json.dumps(parsed_queries, ensure_ascii=False, indent=2)}

Original simulation input:
<<<{original_text}>>>

User reply:
<<<{user_reply}>>>
"""

    resp = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ]).content.strip()

    text = resp
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    return text

def patch_simulation_input_with_user_reply(simulation_input: dict, user_reply: str, parsed_queries: list, llm) -> dict:
    if not simulation_input or not simulation_input.get("present"):
        return simulation_input

    snippets = simulation_input.get("snippets", [])
    if not snippets:
        return simulation_input

    target_idx = 0
    target_snippet = snippets[target_idx]

    patched_text = llm_patch_simulation_snippet(
        software=target_snippet.get("software", ""),
        original_text=target_snippet.get("text", ""),
        user_reply=user_reply,
        parsed_queries=parsed_queries,
        llm=llm,
    )

    if patched_text and patched_text.strip():
        simulation_input["snippets"][target_idx]["text"] = patched_text.strip()

    return simulation_input

if __name__ == "__main__":
    main()
