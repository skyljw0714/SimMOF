from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


SCIENTIFIC_PLAN_FIELDS = {
    "target_observable",
    "physical_estimator",
    "system_scope",
    "sampling_design",
    "command_graph",
    "output_contract",
    "initial_state_checks",
    "numerical_hazards",
    "baseline_gaps",
    "evidence_limitations",
}


def unknown_scientific_plan(reason: str) -> Dict[str, Any]:
    return {
        "target_observable": "",
        "physical_estimator": "",
        "system_scope": "",
        "sampling_design": "",
        "command_graph": [],
        "output_contract": [],
        "initial_state_checks": [],
        "numerical_hazards": [],
        "baseline_gaps": [],
        "evidence_limitations": [reason],
        "planner_status": "fallback_unknown",
    }


def build_scientific_plan_messages(
    *,
    simulation_description: str,
    property_name: str,
    intent_spec: Dict[str, Any],
    capabilities: Dict[str, Any],
    baseline_script: str,
    official_command_hints: str,
    dependency_graph: Dict[str, Any],
) -> List[Any]:
    from langchain.schema import HumanMessage, SystemMessage

    system = """
You are a scientific simulation-method planner. Derive a LAMMPS calculation
design from the physical objective and official evidence. Do not retrieve a
memorized property template and do not assume that a retrieved command must be
used.

Reason in this order:
1. Identify the primary physical observable and the estimator that exposes it.
2. Separate state evolution, observable production, sampling/aggregation, and
   output.
3. Connect every producer and consumer with compatible scalar, vector, array,
   per-atom, per-chunk, or local data.
4. Check that the protocol matches the requested physics: dynamics versus
   minimization, fixed versus flexible cell, and the intended atom group.
   Activate only the cell degrees of freedom required by the observable.
   A triclinic-capable box does not by itself justify sampling shear or tilt.
5. Check initial-state and numerical domains. An expression must be defined
   when first evaluated; initialization and momentum removal must leave usable
   degrees of freedom; a stress definition must match dynamic or mechanical
   sampling.
6. Prefer a directly sampled primary observable over fragile on-the-fly
   post-processing when the derived property requires a fit, slope, integral,
   or multiple samples.
7. Specify what numerical output demonstrates that the calculation actually
   produced the requested property.

Use command names only when they are supported by the supplied official
evidence or already present in the baseline. Return strict JSON only.
Keep the plan concise: at most eight command-graph entries, short field values,
and no restatement of entire documentation passages.
""".strip()
    human = "\n".join(
        [
            "REQUESTED OBJECTIVE",
            simulation_description,
            "",
            "PROPERTY LABEL",
            property_name,
            "",
            "PHYSICAL INTENT",
            json.dumps(intent_spec or {}, ensure_ascii=True, indent=2),
            "",
            "SYSTEM CAPABILITIES",
            json.dumps(capabilities or {}, ensure_ascii=True, indent=2),
            "",
            "EVIDENCE-FREE BASELINE",
            baseline_script or "(none)",
            "",
            "OFFICIAL COMMAND EVIDENCE",
            official_command_hints or "(none)",
            "",
            "ADVISORY DEPENDENCY GRAPH",
            json.dumps(dependency_graph or {}, ensure_ascii=True, indent=2),
            "",
            "Return exactly one JSON object with this schema:",
            "{",
            '  "target_observable": "physical quantity that must be sampled",',
            '  "physical_estimator": "how the requested property follows from sampled data",',
            '  "system_scope": "atoms or groups that evolve and are observed",',
            '  "sampling_design": "state preparation, production sampling, and aggregation",',
            '  "command_graph": [',
            "    {",
            '      "command_name": "official command name",',
            '      "role": "state_evolution|observable_producer|consumer|output",',
            '      "why_needed": "scientific role in this calculation",',
            '      "required_inputs": ["dependency or reference form"],',
            '      "produced_data": "shape and meaning",',
            '      "critical_restrictions": ["relevant official restriction"]',
            "    }",
            "  ],",
            '  "output_contract": [',
            '    {"artifact": "requested or suitable filename", "must_contain": "numerical signal"}',
            "  ],",
            '  "initial_state_checks": ["condition to verify before the first run or minimization"],',
            '  "numerical_hazards": ["undefined expression, incompatible shape, or unstable protocol risk"],',
            '  "baseline_gaps": ["specific scientific or executable gap in the baseline"],',
            '  "evidence_limitations": ["missing or ambiguous evidence that should not be guessed"]',
            "}",
            "",
            "Do not return LAMMPS code.",
        ]
    )
    return [SystemMessage(content=system), HumanMessage(content=human)]


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    value = (text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", value, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def normalize_scientific_plan(raw: Dict[str, Any]) -> Dict[str, Any]:
    plan = unknown_scientific_plan("")
    for key in (
        "target_observable",
        "physical_estimator",
        "system_scope",
        "sampling_design",
    ):
        plan[key] = str(raw.get(key) or "").strip()
    plan["command_graph"] = [
        item
        for item in (raw.get("command_graph") or [])
        if isinstance(item, dict) and str(item.get("command_name") or "").strip()
    ]
    plan["output_contract"] = [
        item
        for item in (raw.get("output_contract") or [])
        if isinstance(item, dict)
    ]
    for key in (
        "initial_state_checks",
        "numerical_hazards",
        "baseline_gaps",
        "evidence_limitations",
    ):
        plan[key] = _string_list(raw.get(key))
    plan["planner_status"] = "ok"
    return {
        key: value
        for key, value in plan.items()
        if key in SCIENTIFIC_PLAN_FIELDS or key == "planner_status"
    }


def infer_lammps_scientific_plan(
    llm: Any,
    *,
    simulation_description: str,
    property_name: str,
    intent_spec: Dict[str, Any],
    capabilities: Dict[str, Any],
    baseline_script: str,
    official_command_hints: str,
    dependency_graph: Dict[str, Any],
) -> Dict[str, Any]:
    messages = build_scientific_plan_messages(
        simulation_description=simulation_description,
        property_name=property_name,
        intent_spec=intent_spec,
        capabilities=capabilities,
        baseline_script=baseline_script,
        official_command_hints=official_command_hints,
        dependency_graph=dependency_graph,
    )
    try:
        response = llm.invoke(messages)
    except Exception as exc:
        return unknown_scientific_plan(
            f"scientific planner failed with {type(exc).__name__}"
        )
    parsed = _extract_json_object(str(getattr(response, "content", "")))
    if parsed is None:
        return unknown_scientific_plan(
            "scientific planner did not return valid JSON"
        )
    return normalize_scientific_plan(parsed)


__all__ = [
    "build_scientific_plan_messages",
    "infer_lammps_scientific_plan",
    "normalize_scientific_plan",
    "unknown_scientific_plan",
]
