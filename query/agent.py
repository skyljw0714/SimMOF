import json
import re
from typing import Any, List, Optional, Set, Tuple
from langchain.schema import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError
from config import LLM_DEFAULT, AGENT_LLM_MAP
from core.llm_logging import log_llm_decision, set_llm_context
from core.databases import list_databases, resolve_db
from core.simulation_contracts import (
    PROPERTY_AGENT,
    canonical_property,
    explicit_properties,
)
from analysis.agent import AnalysisAgent
from rag.agent import RagAgent


CLARIFIED_QUERY_REWRITE_SYSTEM = """
You rewrite a clarified MOF-simulation request into one clean, self-contained
final user query.

Rules:
- Preserve every output, comparison, and analysis explicitly requested in the
  original query.
- Treat the user's clarification reply as authoritative for missing or
  ambiguous conditions and for any property that the reply explicitly narrows.
- If the reply narrows a broad phrase such as "compare adsorption" to a
  specific property such as mixture selectivity, replace the broad ambiguous
  phrase with that specific property. Do not retain both as separate tasks.
- Include only information explicitly present in the original query or reply.
- Do not add supporting simulations, evidence calculations, hypotheses,
  methods, defaults, or scientific assumptions.
- Remove clarification dialogue and phrases such as "additional conditions".
- Return only the rewritten query as plain text, with no label or explanation.
""".strip()


def _clean_clarified_query_response(value: Any) -> str:
    text = str(value or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and parsed.get("final_query"):
                text = str(parsed["final_query"]).strip()
        except json.JSONDecodeError:
            pass
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    return text


def rewrite_clarified_query(
    original_query: str,
    clarification_question: str,
    user_reply: str,
    llm=None,
) -> str:
    user = f"""
Original query:
{original_query.strip()}

Agent clarification question:
{clarification_question.strip()}

User reply:
{user_reply.strip()}

Rewrite these into one clean final query.
""".strip()
    rewrite_llm = llm or AGENT_LLM_MAP.get("QueryAgent", LLM_DEFAULT)
    set_llm_context("QueryAgent", "clarification_query_rewrite")
    response = rewrite_llm.invoke(
        [
            SystemMessage(content=CLARIFIED_QUERY_REWRITE_SYSTEM),
            HumanMessage(content=user),
        ]
    )
    rewritten = _clean_clarified_query_response(response.content)
    if not rewritten:
        raise RuntimeError("Final-query rewrite returned empty text.")
    try:
        log_llm_decision(
            "QueryAgent",
            "clarification_query_rewrite",
            {
                "original_query": original_query,
                "clarification_question": clarification_question,
                "user_reply": user_reply,
                "final_query": rewritten,
            },
        )
    except Exception:
        pass
    return rewritten


AGENT_DESCRIPTIONS = {
    "VASPAgent": "VASP is a DFT-based simulation package for electronic structure, binding energy, Bader charge, and atom/orbital-projected DOS.",
    "RASPAAgent": "RASPA is a Monte Carlo package for gas adsorption (uptake) and henry constants in porous materials.",
    "ZeoppAgent": "zeo++ (zeopp) calculates pore diameters, pore size distributions, surface areas, and accessible pore volumes.",
    "LAMMPSAgent": "LAMMPS computes diffusivity, mean squared displacement (MSD), thermal expansion, and stress-strain properties such as Young's modulus.",
    "ScreeningAgent": "ScreeningAgent performs fast, low-cost pre-screening of large material datasets using simple structural checks, heuristic rules, or approximate property estimators to select suitable candidates for subsequent detailed simulations."
}


class QueryInformation(BaseModel):
    Name: str
    Agent: str
    Property: str
    MOF: str
    Guest: Optional[str] = None
    CIFPath: Optional[str] = None
    CIFDir: Optional[str] = None
    HMOFParams: Optional[dict] = None
    QueryText: Optional[str] = None
    MetalFilter: Optional[List[str]] = None

class SimulationInputSnippet(BaseModel):
    software: str  
    text: str

class SimulationInputPayload(BaseModel):
    present: bool = False
    snippets: List[SimulationInputSnippet] = Field(default_factory=list)


def _model_dump(value: BaseModel) -> dict:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value.dict()


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
{{
  "status": "ok" or "needs_user_confirmation",
  "message": "string"
}}

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
Apply the execution checklist independently to every parsed query before taking
the union of missing fields. A field is present only when the original user
query explicitly supplies it; do not trust an inferred value in a generated
name.
Return ONLY valid JSON. No markdown. No extra keys."""

MISSING_INFO_USER = """Given the original user query and the parsed MOF simulation queries, decide whether any required information is missing.

Your task is NOT to reinterpret the query creatively.
Your task is to decide whether the request is EXECUTABLE as written.

A request may be semantically understandable but still require clarification before execution.

General required information by property:

- surface_area:
  required = [mof]

- pore_volume:
  required = [mof]

- lcd:
  required = [mof]

- pld:
  required = [mof]

- uptake:
  required = [mof, guest, temperature, pressure]

- isotherm:
  required = [mof, guest, temperature, pressure_range]

- henry_coefficient:
  required = [mof, guest, temperature]

- heat_of_adsorption:
  required = [mof, guest, temperature]

- diffusivity:
  required = [mof, guest, temperature]
  and at least one of [loading, pressure]

- mean_squared_displacement:
  required = [mof, guest, temperature]
  and at least one of [loading, pressure]

- binding_energy:
  required = [mof, guest]

- projected_dos:
  required = [mof, guest]

- selectivity:
  required = [mof, composition, temperature, pressure]

- working_capacity:
  required = [mof, guest, temperature, pressure]

- thermal_expansion:
  required = [mof, temperature_range, pressure, cte_type]

Special rule for database-wide requests:
- If any parsed query has MOF="database" but CIFDir is null, the concrete
  database or CIF directory is missing.
- In that case, set needs_clarification=true and include "database" in
  missing_fields.
- Ask which named MOF database or CIF directory should be used.
- Do not silently select a default database.

Special rule for broad adsorption requests:
- If the original user query says only "adsorption" without specifying the adsorption property subtype
  (for example uptake, isotherm, Henry coefficient, heat of adsorption, mixture adsorption, selectivity),
  then you MUST set needs_clarification=true.
- In that case, include "property" in missing_fields.

Hard rules:
1) Never invent missing guest, temperature, pressure, loading, pressure_range, composition, or database.
2) If execution-critical fields are missing, set needs_clarification=true.
3) If the parsed query contains Property="uptake" but guest is null, you MUST set needs_clarification=true.
4) If the parsed query contains Property="uptake" but the original user query does not specify temperature or pressure, you MUST set needs_clarification=true.
5) If the user query is broad/underspecified, prefer clarification over silent assumptions.
6) Ask for all missing required information in one short user-facing question.
7) Evaluate every parsed query. Conditions stated for one query apply to another
   query only when the user explicitly says that they are shared.
8) For an either/or requirement, include both canonical alternatives in
   missing_fields when neither is present, and phrase the question as a choice.

missing_fields must use only canonical names from:
- property
- mof
- guest
- temperature
- pressure
- pressure_range
- loading
- composition
- temperature_range
- cte_type
- database

Return JSON with exactly this schema:
{{
  "needs_clarification": true or false,
  "missing_fields": ["field1", "field2"],
  "question": "string"
}}

Original user query:
<<<{user_input}>>>

Parsed queries:
{queries_json}
"""


_CONDITION_FIELDS = {
    "temperature",
    "pressure",
    "pressure_range",
    "loading",
    "composition",
    "temperature_range",
    "cte_type",
}
_ALLOWED_MISSING_FIELDS = _CONDITION_FIELDS | {
    "property",
    "mof",
    "guest",
    "database",
}
_CONTRACT_PROPERTIES = {
    "surface_area",
    "pore_volume",
    "lcd",
    "pld",
    "uptake",
    "isotherm",
    "henry_coefficient",
    "heat_of_adsorption",
    "diffusivity",
    "mean_squared_displacement",
    "binding_energy",
    "projected_dos",
    "selectivity",
    "working_capacity",
    "thermal_expansion",
}


def _has_temperature(text: str) -> bool:
    return bool(
        re.search(
            r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?\s*(?:K\b|kelvin\b|°\s*C\b|celsius\b)",
            text,
            flags=re.IGNORECASE,
        )
    )


def _has_pressure(text: str) -> bool:
    return bool(
        re.search(
            r"(?<![A-Za-z])\d+(?:\.\d+)?\s*"
            r"(?:bar\b|atm\b|Pa\b|kPa\b|MPa\b|GPa\b|torr\b)",
            text,
            flags=re.IGNORECASE,
        )
    )


def _has_pressure_range(text: str) -> bool:
    return bool(
        re.search(
            r"(?:from\s+)?\d+(?:\.\d+)?\s*(?:-|–|—|to)\s*"
            r"\d+(?:\.\d+)?\s*(?:bar\b|atm\b|Pa\b|kPa\b|MPa\b|torr\b)",
            text,
            flags=re.IGNORECASE,
        )
    )


def _has_loading(text: str) -> bool:
    return bool(
        re.search(
            r"\bloading\b|"
            r"\b(?:one|single|\d+(?:\.\d+)?)\s+"
            r"(?:(?:guest|[A-Za-z][A-Za-z0-9+-]*)\s+)?molecules?\s+per\s+"
            r"(?:unit\s+cell|cell|uc)\b|"
            r"\b\d+(?:\.\d+)?\s*(?:mol|mmol)\s*/\s*(?:kg|g)\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _has_composition(text: str) -> bool:
    return bool(
        re.search(
            r"\bequimolar\b|"
            r"\b\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?\b|"
            r"\by[_\s-]?[A-Za-z0-9]+\s*=\s*\d|"
            r"\b(?:feed|mixture)\s+composition\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _has_temperature_range(text: str) -> bool:
    return bool(
        re.search(
            r"(?:from\s+)?\d+(?:\.\d+)?\s*(?:K\s*)?"
            r"(?:-|–|—|to)\s*\d+(?:\.\d+)?\s*(?:K\b|kelvin\b)",
            text,
            flags=re.IGNORECASE,
        )
    )


def _has_cte_type(text: str) -> bool:
    return bool(re.search(r"\b(?:volumetric|linear)\b", text, flags=re.IGNORECASE))


def _deterministic_condition_check(
    user_input: str,
    queries_list: list,
) -> Tuple[List[str], Set[str]]:
    properties = {
        canonical_property(query.get("Property"))
        for query in queries_list
        if isinstance(query, dict)
    }
    missing: List[str] = []
    covered: Set[str] = set()
    if properties & _CONTRACT_PROPERTIES:
        covered.update(_CONDITION_FIELDS)

    def require(field: str, present: bool) -> None:
        covered.add(field)
        if not present and field not in missing:
            missing.append(field)

    if properties & {"uptake", "working_capacity"}:
        require("temperature", _has_temperature(user_input))
        require("pressure", _has_pressure(user_input))
    if "isotherm" in properties:
        require("temperature", _has_temperature(user_input))
        require("pressure_range", _has_pressure_range(user_input))
    if properties & {"henry_coefficient", "heat_of_adsorption"}:
        require("temperature", _has_temperature(user_input))
    if properties & {"diffusivity", "mean_squared_displacement"}:
        require("temperature", _has_temperature(user_input))
        covered.update({"pressure", "loading"})
        if not (_has_pressure(user_input) or _has_loading(user_input)):
            missing.extend(
                field for field in ("pressure", "loading") if field not in missing
            )
    if "selectivity" in properties:
        require("composition", _has_composition(user_input))
        require("temperature", _has_temperature(user_input))
        require("pressure", _has_pressure(user_input))
    if "thermal_expansion" in properties:
        require("temperature_range", _has_temperature_range(user_input))
        require("pressure", _has_pressure(user_input))
        require("cte_type", _has_cte_type(user_input))

    return missing, covered


def _clarification_question(missing_fields: List[str]) -> str:
    labels = {
        "property": "the exact simulation property",
        "mof": "the MOF",
        "guest": "the guest species",
        "temperature": "temperature",
        "pressure": "pressure",
        "pressure_range": "pressure range",
        "loading": "loading",
        "composition": "mixture composition",
        "temperature_range": "temperature range",
        "cte_type": "linear or volumetric expansion coefficient",
        "database": "the named MOF database or CIF directory",
    }
    fields = list(dict.fromkeys(missing_fields))
    if "pressure" in fields and "loading" in fields:
        fields = [field for field in fields if field not in {"pressure", "loading"}]
        details = [labels.get(field, field) for field in fields]
        details.append("either pressure or loading")
    else:
        details = [labels.get(field, field) for field in fields]
    if not details:
        return ""
    if len(details) == 1:
        joined = details[0]
    else:
        joined = ", ".join(details[:-1]) + f", and {details[-1]}"
    return f"Please provide {joined} so I can build an executable workflow."


def _canonicalize_query_scope(
    user_input: str,
    queries: List[QueryInformation],
    *,
    analysis_enabled: bool,
) -> List[QueryInformation]:
    requested = explicit_properties(user_input)
    compact_input = re.sub(r"[^a-z0-9]", "", user_input.lower())
    named_database = None
    for database_key in list_databases():
        database = resolve_db(database_key) or {}
        names = [
            database_key,
            database.get("display_name"),
            *(database.get("aliases") or []),
        ]
        if any(
            re.sub(r"[^a-z0-9]", "", str(name).lower()) in compact_input
            for name in names
            if name
        ):
            named_database = database_key
            break

    normalized: List[QueryInformation] = []
    for query in queries:
        property_name = canonical_property(query.Property)
        if property_name == "uptake" and "isotherm" in requested:
            property_name = "isotherm"
        query.Property = property_name

        expected_agent = PROPERTY_AGENT.get(property_name)
        if expected_agent and query.Agent not in {"ScreeningAgent", expected_agent}:
            query.Agent = expected_agent
        if (
            str(query.MOF or "").strip().lower() == "database"
            and not query.CIFDir
            and named_database
        ):
            query.CIFDir = named_database

        if analysis_enabled and requested and property_name not in requested:
            continue
        normalized.append(query)
    return normalized

def extract_simulation_input(user_input: str, llm) -> dict:
    set_llm_context("QueryAgent", "simulation_input_extraction")
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
            "snippets": [_model_dump(s) for s in snippets],
        }
    except Exception:
        return {"present": False, "snippets": []}


def needs_analysis(user_input: str, llm) -> bool:
    set_llm_context("QueryAgent", "needs_analysis_check")
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
        set_llm_context("QueryAgent", "simulation_input_review")
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

def check_missing_info(
    user_input: str,
    queries_list: list,
    llm,
    semantic_guardrails: bool = False,
) -> dict:
    try:
        set_llm_context("QueryAgent", "missing_info_check")
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

        if not semantic_guardrails:
            missing_fields = list(dict.fromkeys(parsed.missing_fields))
            needs_clarification = bool(parsed.needs_clarification)
            question = parsed.question.strip()
            if needs_clarification and not question:
                missing = (
                    ", ".join(missing_fields)
                    if missing_fields
                    else "the missing simulation information"
                )
                question = f"Please provide {missing}."
            return {
                "needs_clarification": needs_clarification,
                "missing_fields": missing_fields,
                "question": question,
            }

        deterministic_missing, covered_conditions = _deterministic_condition_check(
            user_input,
            queries_list,
        )
        missing_fields = [
            field
            for field in dict.fromkeys(parsed.missing_fields)
            if (
                field in _ALLOWED_MISSING_FIELDS
                and field not in covered_conditions
                and field != "database"
            )
        ]
        missing_fields.extend(
            field for field in deterministic_missing if field not in missing_fields
        )
        generic_database = any(
            isinstance(query, dict)
            and str(query.get("MOF") or "").strip().lower() == "database"
            and not str(query.get("CIFDir") or "").strip()
            for query in queries_list
        )
        if generic_database and "database" not in missing_fields:
            missing_fields.append("database")

        needs_clarification = bool(missing_fields)
        question = _clarification_question(missing_fields)

        return {
            "needs_clarification": needs_clarification,
            "missing_fields": missing_fields,
            "question": question,
        }

    except Exception:
        if not semantic_guardrails:
            return {
                "needs_clarification": False,
                "missing_fields": [],
                "question": "",
            }
        deterministic_missing, _ = _deterministic_condition_check(
            user_input,
            queries_list,
        )
        generic_database = any(
            isinstance(query, dict)
            and str(query.get("MOF") or "").strip().lower() == "database"
            and not str(query.get("CIFDir") or "").strip()
            for query in queries_list
        )
        if generic_database and "database" not in deterministic_missing:
            deterministic_missing.append("database")
        return {
            "needs_clarification": bool(deterministic_missing),
            "missing_fields": deterministic_missing,
            "question": _clarification_question(deterministic_missing),
        }

def analyze_mof_query(
    user_input: str,
    llm=None,
    semantic_guardrails: bool = False,
):
    analysis_enabled = False
    analysis_recommendation_json = ""
    analysis_recommendation = {}

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
        if hasattr(plan_obj, "model_dump"):
            analysis_recommendation = _model_dump(plan_obj)
        elif hasattr(plan_obj, "dict"):
            analysis_recommendation = plan_obj.dict()
        elif isinstance(plan_obj, dict):
            analysis_recommendation = plan_obj
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
        },
        {
            "input": "Build a custom hMOF with tbo topology, N10 and N409 nodes, and calculate surface area",
            "output": [
                {
                "Name": "hmof-tbo-surface_area",
                "Agent": "ZeoppAgent",
                "Property": "surface_area",
                "MOF": "hmof",
                "Guest": None,
                "CIFPath": None,
                "CIFDir": None,
                "HMOFParams": {
                    "type": "custom",
                    "n_mofs": 1,
                    "topology": "tbo",
                    "nodes": {"0": "N10", "1": "N409"},
                    "edge_bbs": None,
                    "optimize": False
                }
                }
            ]
        },
        {
            "input": "Generate 5 random hMOFs with max 1000 atoms and calculate LCD for each",
            "output": [
                {
                "Name": "hmof-random-lcd",
                "Agent": "ZeoppAgent",
                "Property": "lcd",
                "MOF": "hmof",
                "Guest": None,
                "CIFPath": None,
                "CIFDir": None,
                "HMOFParams": {
                    "type": "random",
                    "n_mofs": 5,
                    "max_atoms": 1000,
                    "min_cell": 4.5,
                    "max_cell": 60.0,
                    "random_seed": None,
                    "optimize": False
                }
                }
            ]
        }
    ]

    examples_text = ""
    for i, example in enumerate(examples, 1):
        examples_text += f"Example {i}:\n"
        examples_text += f"Input: {example['input']}\n"
        examples_text += f"Output: {json.dumps(example['output'], indent=2)}\n\n"

    database_catalog = []
    for database_key in list_databases():
        database = resolve_db(database_key) or {}
        database_catalog.append(
            {
                "key": database_key,
                "display_name": database.get("display_name"),
                "aliases": database.get("aliases") or [],
            }
        )
    database_catalog_text = json.dumps(
        database_catalog,
        ensure_ascii=False,
        indent=2,
    )


    structured_prompt = f"""
{tool_desc_text}
You are a MOF simulation expert.
You may be provided with an analysis recommendation produced by AnalysisAgent.
If analysis_recommendation_json is provided:
- Treat AnalysisAgent's calculation_requests as its explicit tool selections.
- Create one simulation query for every calculation_request, reusing an equivalent
  user-requested simulation instead of duplicating it.
- Do not drop a calculation_request or add a different supporting calculation.
- Respect explicit exclusions in the user question; AnalysisAgent should already
  have omitted excluded tools from calculation_requests.
- Do not emit analysis_plan methods as simulation properties.
- WorkingAgent will execute the selected calculation requests and use
  analysis_plan after they finish.

Analysis recommendation (JSON):
{analysis_recommendation_json}

Hard rule:
- NEVER output Agent="AnalysisAgent" or Agent="ResponseAgent".
- Output only simulation/tool agents (VASPAgent, RASPAAgent, ZeoppAgent, LAMMPSAgent, ...).
- "identify/choose/rank/find the most stable/best" is NOT a separate simulation property. It must be handled in final_response using results from computed energies. Do NOT output a separate query for it.

QUERY SCOPE AND CARDINALITY CONTRACT:

- Every output element must correspond to one simulation explicitly requested
  by the user, except for the analysis-only fallback described above.
- Never add a property merely because it could help explain another result.
- Do not replace an explicitly requested property with a related property.
- Canonical property names include:
  surface_area, pore_volume, lcd, pld, uptake, isotherm,
  henry_coefficient, heat_of_adsorption, selectivity, working_capacity,
  diffusivity, mean_squared_displacement, thermal_expansion,
  binding_energy, adsorption_energy, bader_charge, projected_dos, band_gap.
- An adsorption isotherm or continuous pressure range is ONE query with
  Property="isotherm"; it is not a single-pressure uptake query.
- A thermal-expansion temperature range is ONE thermal_expansion query.
- Multiple discrete conditions for the same property require one query per
  condition. Multiple MOFs require one query per MOF and property.
- A request for selectivity does not implicitly request pure-component uptake,
  Henry coefficients, or heats of adsorption.
- A request for interpretation, explanation, correlation, or trend changes
  analysis_enabled; it does not expand the simulation-query set.

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

CIF PATH RULE:
- If the user explicitly provides a file path to a CIF file (e.g. "/path/to/mof.cif", "~/data/my.cif"),
  extract it as CIFPath. Otherwise set CIFPath to null.
- If CIFPath is provided, set MOF to the filename stem (without extension) unless the user names the MOF explicitly.

DATABASE RULE:
- If the user refers to a named MOF database or asks to run on "all MOFs" / "the database" / "entire database",
  set MOF to "database".
- If the user names a database in the registry catalog below, set CIFDir to
  its registry key. Otherwise set CIFDir to null.
- If the user provides an explicit directory path containing CIF files (e.g. "/path/to/cifs/", "~/data/mofs/"),
  set CIFDir to that path directly and set MOF to "database".
- Database registry catalog:
{database_catalog_text}
- For screening on a database: use Agent="ScreeningAgent".
- For batch computation on a database (e.g. "calculate LCD for all MOFs"): use the appropriate simulation agent (ZeoppAgent, RASPAAgent, etc.) with MOF="database".

METAL FILTER RULE:
- If the user restricts the database or directory to MOFs containing specific metals
  (e.g. "only Zr MOFs", "Zr-based MOFs", "Cu and Zn MOFs", "MOFs with Zr nodes"),
  extract the metal element symbols as a list in MetalFilter (e.g. ["Zr"] or ["Cu", "Zn"]).
- MetalFilter applies only when MOF="database" (named DB or directory).
- If no metal restriction is mentioned, set MetalFilter to null.

HMOF RULE:
- If the user asks to generate or build a hypothetical MOF (hMOF) and run a simulation, set MOF to "hmof".
- Set HMOFParams as a JSON object with the following fields:
  - "type": "custom" if the user specifies topology/building blocks, "random" if they want randomly generated hMOFs.
  - "n_mofs": number of hMOFs to generate (default 1 for custom, 1 for random unless specified).
  - For type="custom": "topology" (e.g. "tbo", "pcu"), "nodes" (dict of node index to BB name, e.g. {{"0": "N10", "1": "N409"}}), "edge_bbs" (optional dict, e.g. {{"0,1": "E41"}}).
  - For type="random": "max_atoms" (default 1500), "min_cell" (default 4.5), "max_cell" (default 60.0), "random_seed" (null unless specified).
  - "optimize": true if user requests LAMMPS energy minimization after building (default false).
- If n_mofs > 1 or type="random" with n_mofs > 1, a batch simulation is run on all generated CIFs.

Return your answer *strictly* as a JSON array (not an object).
Each element in the list must follow this schema:

{{
  "Name": "string",
  "Agent": "string",
  "Property": "string",
  "MOF": "string",
  "Guest": "string or null",
  "CIFPath": "string or null",
  "HMOFParams": "object or null",
  "MetalFilter": "list of element symbols or null",
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


    set_llm_context("QueryAgent", "query_parsing")
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
        try:
            data = json.loads(text)
        except json.JSONDecodeError as first_error:
            set_llm_context("QueryAgent", "query_json_contract_repair")
            syntax_repair = llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "Repair JSON syntax and schema formatting only. "
                            "Preserve every intended query and value. Return only "
                            "one valid JSON array with no markdown."
                        )
                    ),
                    HumanMessage(
                        content=f"""
The QueryAgent output was not valid JSON.

Parser error:
{first_error}

Malformed output:
<<<{raw}>>>

Return the same intended QueryInformation array with valid JSON syntax.
""".strip()
                    ),
                ]
            )
            repaired_text = syntax_repair.content.strip()
            if repaired_text.startswith("```"):
                repaired_text = "\n".join(
                    repaired_text.splitlines()[1:-1]
                ).strip()
            data = json.loads(repaired_text)
        if not isinstance(data, list):
            raise ValueError("Output is not a list.")
        initial_queries = [QueryInformation(**item) for item in data]
        queries = (
            _canonicalize_query_scope(
                user_input,
                initial_queries,
                analysis_enabled=analysis_enabled,
            )
            if semantic_guardrails
            else initial_queries
        )
        if semantic_guardrails and not queries:
            requested_properties = sorted(explicit_properties(user_input))
            set_llm_context("QueryAgent", "query_scope_contract_repair")
            repair_response = llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "You repair a MOF query parse to match the user's "
                            "explicit simulation scope. Return only a JSON array "
                            "of QueryInformation objects."
                        )
                    ),
                    HumanMessage(
                        content=f"""
The previous parse contained only calculations outside the explicitly named
simulation scope and was rejected.

Original user query:
<<<{user_input}>>>

Explicitly named canonical properties:
{json.dumps(requested_properties)}

Rejected parse:
{json.dumps(data, ensure_ascii=False, indent=2)}

Repair rules:
- The explicit properties are authoritative.
- Preserve the requested MOFs and guest species from the original query.
- Combine mixture components into one guest expression for a mixture property.
- Do not add supporting calculations for explanation or analysis.
- Follow the same QueryInformation JSON schema as the rejected parse.
- Return ONLY the corrected JSON array.
""".strip()
                    ),
                ]
            )
            repair_text = repair_response.content.strip()
            if repair_text.startswith("```"):
                repair_text = "\n".join(repair_text.splitlines()[1:-1]).strip()
            repaired_data = json.loads(repair_text)
            if not isinstance(repaired_data, list):
                raise ValueError("Scope-contract repair output is not a list.")
            queries = _canonicalize_query_scope(
                user_input,
                [QueryInformation(**item) for item in repaired_data],
                analysis_enabled=analysis_enabled,
            )
            if not queries:
                raise ValueError(
                    "No executable query remained after scope-contract repair."
                )

        for q in queries:
            q.QueryText = user_input

        print("=== Parsed Queries ===")
        for q in queries:
            print(f"- {q.Name}: {q.Agent} → {q.Property} ({q.MOF}, guest={q.Guest})")

        queries_list = [_model_dump(q) for q in queries]
        try:
            log_llm_decision("QueryAgent", "query_parsing", queries_list)
        except Exception:
            pass

        missing_info = check_missing_info(
            user_input=user_input,
            queries_list=queries_list,
            llm=llm,
            semantic_guardrails=semantic_guardrails,
        )

        return {
            "queries": queries_list,
            "analysis_enabled": analysis_enabled,
            "analysis_recommendation": analysis_recommendation,
            "simulation_input": simulation_input,
            "needs_clarification": missing_info["needs_clarification"],
            "missing_fields": missing_info["missing_fields"],
            "clarification_question": missing_info["question"],
        }

    except (json.JSONDecodeError, ValidationError, ValueError) as e:
        print(f"Parsing error: {e}")
        print("Original response:", raw)
        return None
