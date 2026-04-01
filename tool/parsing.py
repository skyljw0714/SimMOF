
import json
from typing import Optional, Dict, Any
from langchain.schema import HumanMessage

def _safe_json_loads(text: str) -> dict:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return json.loads(t)

def extract_conditions_with_llm(step, goal: Optional[str] = None, llm=None) -> Dict[str, Any]:
    if llm is None:
        from config import LLM_DEFAULT
        llm = LLM_DEFAULT

    
    if isinstance(step, dict):
        tool_raw = (step.get("tool") or "").strip()
        condition_text = (step.get("condition") or "").strip()
    else:
        tool_raw = (getattr(step, "tool", "") or "").strip()
        condition_text = (getattr(step, "condition", "") or "").strip()

    tool = tool_raw.lower()

    base_rules = """
You must convert ONE screening step into a SINGLE-condition JSON object.
Return JSON ONLY. No markdown. No commentary.

Schema (include all keys; use null when not applicable):

{
  "tool": "zeo++|ASE_atom_count|RASPA_henry|MLIP_geo|MLIP_be|ASE_atom_type|MOFChecker",
  "mode": "filter|rank",
  "property": "string",
  "op": ">=|<=|>|<|==|!=|TOP",
  "value": number,

  "top_n": number or null,

  "molecule": "string or null",
  "temperature_K": number or null,

  "zeopp_command": "string or null",
  "probe_radius": number or null,
  "num_samples": number or null,

  "max_atoms": number or null,
  "atom_type": ["El1","El2"] or null
}

Important:
- Each step must refer to EXACTLY ONE property (property field).
- If ranking (e.g., "Top-100"), set mode="rank", op="TOP", top_n=<N>, value=<N>.
- If filtering, set mode="filter" and use numeric op/value.
- The JSON must reflect ONLY the Step condition text, not the overall user goal.
- The 'User goal' is context only (e.g., molecule/temperature) and MUST NOT override numbers in the Step condition.
- If the Step condition contains an explicit number (e.g., Top-30, >2000, >=2.89), you MUST use that exact number.
- Never replace Top-N in the Step with a different N from the goal.
"""

    if tool in ["zeo++", "zeopp", "zeo"]:
        allowed_props = [
            "included_sphere",
            "free_sphere",
            "included_sphere_along_free_path",
            "ASA_m2_g",
            "AV_cm3_g",
        ]
        prompt = f"""
User goal: {goal or "N/A"}

Tool: zeo++
Step condition text: "{condition_text}"

Interpretation constraints:
- Choose exactly one property from: {allowed_props}
- Choose zeo++ command:
  * if property in ["included_sphere","free_sphere","included_sphere_along_free_path"] -> zeopp_command="-ha -res"
  * if property == "ASA_m2_g" -> zeopp_command="-ha -sa"
  * if property == "AV_cm3_g" -> zeopp_command="-ha -vol"
- For "-ha -sa" and "-ha -vol": probe_radius=1.2 (unless mentioned), num_samples=50000
- For "-ha -res": probe_radius=null, num_samples=null

{base_rules}
Output JSON now.
"""

    elif tool in ["ase_atom_count", "atom_count"]:
        prompt = f"""
User goal: {goal or "N/A"}

Tool: ASE_atom_count
Step condition text: "{condition_text}"

Constraints:
- property must be "atom_count"
- mode="filter"
- Example: ">2000 atoms" -> op=">", value=2000, max_atoms=2000

{base_rules}
Output JSON now.
"""

    elif tool in ["raspa_henry", "raspa henry", "henry"]:
        prompt = f"""
User goal: {goal or "N/A"}

Tool: RASPA_henry
Step condition text: "{condition_text}"

Constraints:
- property="henry_coefficient"
- ranking => mode="rank", op="TOP", top_n=N, value=N
- Extract molecule if present else null
- Extract temperature_K if present else null

{base_rules}
Output JSON now.
"""

    elif tool in ["mofchecker"]:
        
        prompt = f"""
User goal: {goal or "N/A"}

Tool: MOFChecker
Step condition text: "{condition_text}"

Constraints:
- mode="filter"
- property="structure_valid"
- op="=="
- value=1

{base_rules}
Output JSON now.
"""

    else:
        prompt = f"""
User goal: {goal or "N/A"}

Tool: {tool_raw}
Step condition text: "{condition_text}"

{base_rules}
Output JSON now.
"""

    resp = llm.invoke([HumanMessage(content=prompt)])
    data = _safe_json_loads(resp.content)

    defaults = {
        "tool": tool_raw if tool_raw else None,
        "mode": None,
        "property": None,
        "op": None,
        "value": None,
        "top_n": None,
        "molecule": None,
        "temperature_K": None,
        "zeopp_command": None,
        "probe_radius": None,
        "num_samples": None,
        "max_atoms": None,
        "atom_type": None,
    }
    for k, v in defaults.items():
        data.setdefault(k, v)

    if str(data.get("op", "")).upper() == "TOP":
        n = data.get("top_n", data.get("value"))
        try:
            n_int = int(n)
        except Exception:
            n_int = None
        data["mode"] = "rank"
        data["op"] = "TOP"
        data["top_n"] = n_int
        data["value"] = n_int

    return data
