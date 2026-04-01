import json
from typing import List, Optional
from langchain.schema import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError
from config import LLM_DEFAULT, AGENT_LLM_MAP
from analysis.agent import AnalysisAgent
from rag.agent import RagAgent


AGENT_DESCRIPTIONS = {
    "VASPAgent": "VASP is a DFT-based simulation package for electronic structure and binding energy.",
    "RASPAAgent": "RASPA is a Monte Carlo package for gas adsorption (uptake) and henry constants in porous materials.",
    "ZeoppAgent": "zeo++ (zeopp) calculates surface area and pore volume of porous materials.",
    "LAMMPSAgent": "LAMMPS is a molecular dynamics simulator used to compute time-dependent properties. Use this for diffusivity, diffusion coefficients, mean squared displacement (MSD), and stress/strain in atomistic and molecular systems",
    "ScreeningAgent": "ScreeningAgent performs fast, low-cost pre-screening of large material datasets using simple structural checks, heuristic rules, or approximate property estimators to select suitable candidates for subsequent detailed simulations."
}


class QueryInformation(BaseModel):
    Name: str
    Agent: str
    Property: str
    MOF: str
    Guest: Optional[str] = None
    QueryText: Optional[str] = None

class SimulationInputSnippet(BaseModel):
    software: str  
    text: str

class SimulationInputPayload(BaseModel):
    present: bool = False
    snippets: List[SimulationInputSnippet] = Field(default_factory=list)


SIM_INPUT_SYSTEM = """You extract simulation input snippets from a user message.
Return ONLY valid JSON. No markdown. No extra keys."""

SIM_INPUT_USER = """Extract any simulation input snippets the user provided.
These are pasted input contents/commands such as:
- LAMMPS: system.in Run Section commands
- RASPA: input file contents
- VASP: INCAR contents (and optionally KPOINTS/POSCAR if provided)
- Zeopp: zeo++ command line(s)

Rules:
- If the user provided no such snippet, return present=false and an empty snippets list.
- If multiple snippets exist, return multiple entries.
- Each entry must have:
  - software: EXACTLY one of ["LAMMPS","RASPA","VASP","Zeopp"] (case-sensitive)
  - text: the exact extracted snippet text only (no surrounding prose)

Output schema (exact):
{{
  "present": boolean,
  "snippets": [
    {{"software": "...", "text": "..."}}
  ]
}}

User message:
<<<{user_input}>>>
"""


ROUTER_SYSTEM = """You are a router.
Answer with ONLY one token: true or false.
No punctuation. No explanation. No other words."""

ROUTER_USER = """Decide whether the user message requires EXTRA ANALYSIS beyond running simulations/tools.
You MUST answer with only one token: true or false.

Definitions:
- EXTRA ANALYSIS = scientific interpretation/explanation such as reasons/why, mechanism, trends, correlations,
  literature-style discussion, or causal claims beyond reporting computed numbers.

Hard rules (highest priority):
1) Return false for DIRECT COMPUTATION REQUESTS.
A direct computation request is when the user primarily asks to calculate/compute/run/simulate/reproduce/get
a property, even if it sounds scientific.
These MUST be false:
- "I want to calculate the binding energy of CO2 in ZIF-8"
- "Compute Henry coefficient of CO2 in HKUST-1"
- "Run RASPA to get CO2 uptake in MOF-5"
- "Calculate pore volume / surface area of MOF-5"
- "Calculate diffusivity of CO2 in HKUST-1"

2) Return false for SIMPLE NUMERIC COMPARISONS that only require computing and comparing numbers,
as long as the user does NOT ask why/explain/mechanism/trend.
These MUST be false:
- "which MOF has the larger pore volume between A and B"
- "which has higher CO2 uptake, A or B"
- "compare surface area of A vs B" (without asking why)

3) Return true ONLY if the user explicitly asks for explanation/interpretation beyond numbers.
Triggers (any of these words/intent):
- why, explain, reason, mechanism, interpret, analyze, discuss, trend, correlation
Examples that MUST be true:
- "why is A higher than B"
- "explain the difference between A and B"
- "discuss the trend across A, B, C"
- "analyze the reason/mechanism"

User message:
{user_input}
"""

SIM_INPUT_REVIEW_SYSTEM = """You review user-provided simulation input snippets for MOF workflows.
Return ONLY valid JSON. No markdown. No extra keys."""

SIM_INPUT_REVIEW_USER = """Review the extracted simulation input snippets against the parsed MOF simulation queries.

Your task:
- Decide whether the provided simulation input looks suitable for the current requested task.
- Consider all supported software types: LAMMPS, RASPA, VASP, Zeopp.
- Be conservative.
- If there is any meaningful mismatch, ask for user confirmation.

Check for issues such as:
- the input appears intended for a different MOF/framework/system
- the input appears intended for a different property or simulation goal
- the guest/species does not match the request
- explicit conditions in the input conflict with the request
- reproduce-style reuse may be risky for the new system
- the snippet is incomplete or suspicious for the claimed software

Return JSON with exactly this schema:
{
  "status": "ok" or "needs_user_confirmation",
  "message": "string"
}

Rules:
- status="ok" if the input looks broadly usable as-is.
- status="needs_user_confirmation" if the input may not match the requested MOF, property, guest, conditions, or intent.
- The "message" must be a single user-facing message.
- If confirmation is needed, the message must clearly instruct the user to reply with one of:
  - KEEP
  - REGENERATE
  - or paste corrected simulation input directly
- Do not mention JSON.
- Keep the message concise but clear.

Parsed queries:
{queries_json}

Extracted simulation_input:
{simulation_input_json}

Original user message:
<<<{user_input}>>>
"""

class MissingInfoCheckResult(BaseModel):
    needs_clarification: bool = False
    missing_fields: List[str] = Field(default_factory=list)
    question: str = ""


MISSING_INFO_SYSTEM = """You check whether a MOF simulation request has enough information to proceed.
Return ONLY valid JSON. No markdown. No extra keys."""

MISSING_INFO_USER = """Given the original user query and the parsed MOF simulation queries, decide whether any required information is missing.

You must determine whether the simulation can proceed safely.

General guidance:
- uptake at a single state point usually requires:
  - MOF
  - guest
  - temperature
  - pressure
- isotherm usually requires:
  - MOF
  - guest
  - temperature
  - pressure range
- henry_coefficient usually requires:
  - MOF
  - guest
  - temperature
- diffusivity usually requires:
  - MOF
  - guest
  - temperature
  - and either loading or pressure
- binding_energy usually requires:
  - MOF
  - guest
  - temperature/pressure are usually NOT required
- surface_area / pore_volume usually require:
  - MOF only

Rules:
1) Be conservative but practical.
2) Ask only for truly required missing information.
3) If everything essential is present, return needs_clarification=false.
4) Ask for all missing required information in ONE combined question.
5) Keep the question short and user-facing.
6) missing_fields should use simple canonical names like:
   - temperature
   - pressure
   - pressure_range
   - guest
   - mof
   - loading
   - composition

Return JSON with exactly this schema:
{
  "needs_clarification": true or false,
  "missing_fields": ["field1", "field2"],
  "question": "string"
}

Original user query:
<<<{user_input}>>>

Parsed queries:
{queries_json}
"""

def extract_simulation_input(user_input: str, llm) -> dict:
    resp = llm.invoke([
        SystemMessage(content=SIM_INPUT_SYSTEM),
        HumanMessage(content=SIM_INPUT_USER.format(user_input=user_input)),
    ]).content.strip()

    
    text = resp
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    
    import re
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        text = m.group(0).strip()

    try:
        data = json.loads(text)
        payload = SimulationInputPayload(**data)

        
        snippets = [s for s in payload.snippets if s.text and s.text.strip()]
        present = bool(payload.present and len(snippets) > 0)

        return {
            "present": present,
            "snippets": [s.model_dump() for s in snippets],
        }
    except Exception:
        return {"present": False, "snippets": []}


def needs_analysis(user_input: str, llm) -> bool:
    resp = llm.invoke([
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=ROUTER_USER.format(user_input=user_input)),
    ]).content.strip().lower()

    resp = resp.replace(".", "").replace(",", "").strip()

    if resp == "true":
        return True
    if resp == "false":
        return False

    return False

def review_simulation_input(user_input: str, queries_list: list, simulation_input: dict, llm) -> dict:
    if not simulation_input or not simulation_input.get("present"):
        return {
            "status": "ok",
            "message": "",
        }

    try:
        resp = llm.invoke([
            SystemMessage(content=SIM_INPUT_REVIEW_SYSTEM),
            HumanMessage(content=SIM_INPUT_REVIEW_USER.format(
                queries_json=json.dumps(queries_list, ensure_ascii=False, indent=2),
                simulation_input_json=json.dumps(simulation_input, ensure_ascii=False, indent=2),
                user_input=user_input,
            )),
        ]).content.strip()

        text = resp
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].lstrip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        import re
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            text = m.group(0).strip()

        data = json.loads(text)
        status = data.get("status", "ok")
        message = (data.get("message") or "").strip()

        if status not in {"ok", "needs_user_confirmation"}:
            status = "ok"

        if status == "needs_user_confirmation" and not message:
            message = (
                "The provided simulation input may not match the current requested task. "
                "Reply with KEEP to use it as-is, REGENERATE to ignore it and create a new input, "
                "or paste a corrected command/input directly."
            )

        return {
            "status": status,
            "message": message,
        }

    except Exception:
        return {
            "status": "needs_user_confirmation",
            "message": (
                "The provided simulation input could not be verified automatically. "
                "Reply with KEEP to use it as-is, REGENERATE to ignore it and create a new input, "
                "or paste a corrected command/input directly."
            ),
        }

def _plan_to_json_text(plan_obj) -> str:
    
    if hasattr(plan_obj, "model_dump_json"):  
        return plan_obj.model_dump_json(indent=2)
    if hasattr(plan_obj, "json"):  
        try:
            return plan_obj.json(indent=2)
        except TypeError:
            return plan_obj.json()
    if isinstance(plan_obj, dict):
        return json.dumps(plan_obj, ensure_ascii=False, indent=2)
    return str(plan_obj)

def check_missing_info(user_input: str, queries_list: list, llm) -> dict:
    try:
        resp = llm.invoke([
            SystemMessage(content=MISSING_INFO_SYSTEM),
            HumanMessage(content=MISSING_INFO_USER.format(
                user_input=user_input,
                queries_json=json.dumps(queries_list, ensure_ascii=False, indent=2),
            )),
        ]).content.strip()

        text = resp
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].lstrip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        import re
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            text = m.group(0).strip()

        data = json.loads(text)
        parsed = MissingInfoCheckResult(**data)

        
        if parsed.needs_clarification and not parsed.question.strip():
            missing = ", ".join(parsed.missing_fields) if parsed.missing_fields else "some required information"
            question = f"I need the following information to continue: {missing}. Please provide it."
        else:
            question = parsed.question.strip()

        return {
            "needs_clarification": parsed.needs_clarification,
            "missing_fields": parsed.missing_fields,
            "question": question,
        }

    except Exception:
        
        return {
            "needs_clarification": False,
            "missing_fields": [],
            "question": "",
        }

def analyze_mof_query(user_input: str, llm=None):
    analysis_enabled = False
    analysis_recommendation_json = ""

    if llm is None:
        llm = AGENT_LLM_MAP.get("QueryAgent", LLM_DEFAULT)

    simulation_input = extract_simulation_input(user_input, llm)

    if needs_analysis(user_input, llm):
        
        miner = RagAgent(agent_name="RagAgent")
        mined = miner.run(
            user_input,
            parsed_query={},   
            k_papers=5,
        )

        
        context = {
            "query_text": user_input,   
            "results": {
                "rag": {
                    "metric": mined.get("metric"),
                    "queries": mined.get("queries"),
                    "top_papers": mined.get("top_papers"),
                    "evidence_block": mined.get("evidence_block"),
                },
                
                "evidence_block": mined.get("evidence_block", ""),
                "top_papers": mined.get("top_papers", []),
            },
        }

        aa = AnalysisAgent()
        plan_obj = aa.recommend_analysis_tasks(context)

        
        analysis_recommendation_json = _plan_to_json_text(plan_obj)
        analysis_enabled = True

    


    
    tool_desc_text = "=== Simulation Software Descriptions ===\n"
    for tool, desc in AGENT_DESCRIPTIONS.items():
        tool_desc_text += f"- {tool}: {desc}\n"
    tool_desc_text += "\n"

    
    examples = [
        {
            "input": "I want to reproduce Binding energy of CO2 in HKUST-1 MOF",
            "output": [{
                "Name": "HKUST-1-CO2-binding_energy",
                "Agent": "VASPAgent",
                "Property": "binding_energy",
                "MOF": "HKUST-1",
                "Guest": "CO2",
            }]
        },
        {
            "input": "which MOF has the larger pore volume between HKUST-1 and MOF-5",
            "output": [
                {
                    "Name": "HKUST-1-pore_volume",
                    "Agent": "ZeoppAgent",
                    "Property": "pore_volume",
                    "MOF": "HKUST-1",
                    "Guest": None
                },
                {
                    "Name": "MOF-5-pore_volume",
                    "Agent": "ZeoppAgent",
                    "Property": "pore_volume",
                    "MOF": "MOF-5",
                    "Guest": None
                }
            ]
        },
        {
            "input": "I want to calculate diffusivity of CO2 in HKUST-1",
            "output": [{
                "Name": "HKUST-1-CO2-diffusivity",
                "Agent": "LAMMPSAgent",
                "Property": "diffusivity",
                "MOF": "HKUST-1",
                "Guest": "CO2"
            }]
        },
        {
            "input": "I want to find MOFs with the highest CO2 uptake at 1 bar and 298K in the database",
            "output": [
                {
                "Name": "allMOFs-CO2-uptake-1bar-298K-screen",
                "Agent": "ScreeningAgent",
                "Property": "uptake",
                "MOF": "database",
                "Guest": "CO2"
                },
                {
                "Name": "allMOFs-CO2-uptake-1bar-298K-raspa",
                "Agent": "RASPAAgent",
                "Property": "uptake",
                "MOF": "database",
                "Guest": "CO2"
                }
            ]
        }
    ]

    examples_text = ""
    for i, example in enumerate(examples, 1):
        examples_text += f"Example {i}:\n"
        examples_text += f"Input: {example['input']}\n"
        examples_text += f"Output: {json.dumps(example['output'], indent=2)}\n\n"



    structured_prompt = f"""
{tool_desc_text}
You are a MOF simulation expert.
You may be provided with an analysis recommendation produced by AnalysisAgent.
If analysis_recommendation_json is provided, you MUST generate parsed queries that cover every recommended step/method in it (do not omit any).

Analysis recommendation (JSON):
{analysis_recommendation_json}

Hard rule:
- NEVER output Agent="AnalysisAgent" or Agent="ResponseAgent".
- Output only simulation/tool agents (VASPAgent, RASPAAgent, ZeoppAgent, LAMMPSAgent, ...).
- "identify/choose/rank/find the most stable/best" is NOT a separate simulation property. It must be handled in final_response using results from computed energies. Do NOT output a separate query for it.

CONDITION ATTACHMENT RULE (VERY IMPORTANT):

- Do NOT invent or infer simulation conditions.
- Do NOT infer or assume conditions based on analysis goals, scientific reasoning,
  consistency requirements, or common practice.

NAME FORMAT RULE (STRICT):

- By default, Name MUST NOT include any conditions
  (no "-298K", "-1bar", "-eq_loading", "-lowP", "-Henry", etc.).

- You are NOT allowed to append default or assumed conditions to Name.
  (e.g., do NOT add "298K" or "1bar" unless the user explicitly wrote them.)

- Only append conditions to Name if the user explicitly provides the condition values
  in the user input text (numerical values like 200 K, 300 K, 1 bar, 10 bar, etc.).

- Never append method/regime words to Name
  (e.g., "eq_loading", "equilibrium", "Henry", "Widom", "GCMC")
  unless the user explicitly requests that method.

- If the user did NOT provide any explicit numerical conditions,
  Name must be exactly:
  "<MOF>-<Guest>-<Property>"  (or "<MOF>-<Property>" if Guest is null).

Return your answer *strictly* as a JSON array (not an object).
Each element in the list must follow this schema:

{{
  "Name": "string",
  "Agent": "string",
  "Property": "string",
  "MOF": "string",
  "Guest": "string or null",
}}

Examples:
{examples_text}

User input: "{user_input}"

Return ONLY the JSON array, e.g.:
[
  {{...}},
  {{...}}
]
"""

    messages = [
        SystemMessage(content="You are a MOF simulation expert. Output must be a JSON array."),
        HumanMessage(content=structured_prompt)
    ]

    
    response = llm.invoke(messages)
    raw = response.content

    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("Output is not a list.")
        queries = [QueryInformation(**item) for item in data]

        for q in queries:
            q.QueryText = user_input

        print("=== Parsed Queries ===")
        for q in queries:
            print(f"- {q.Name}: {q.Agent} → {q.Property} ({q.MOF}, guest={q.Guest})")

        queries_list = [q.model_dump() for q in queries]  

        sim_input_review = review_simulation_input(
            user_input=user_input,
            queries_list=queries_list,
            simulation_input=simulation_input,
            llm=llm,
        )

        missing_info = check_missing_info(
            user_input=user_input,
            queries_list=queries_list,
            llm=llm,
        )

        return {
            "queries": queries_list,
            "analysis_enabled": analysis_enabled,
            "simulation_input": simulation_input,
            "simulation_input_status": sim_input_review["status"],
            "simulation_input_message": sim_input_review["message"],
            "needs_clarification": missing_info["needs_clarification"],
            "missing_fields": missing_info["missing_fields"],
            "clarification_question": missing_info["question"],
        }

    except (json.JSONDecodeError, ValidationError, ValueError) as e:
        print(f"Parsing error: {e}")
        print("Original response:", raw)
        return None