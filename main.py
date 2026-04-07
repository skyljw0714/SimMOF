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
