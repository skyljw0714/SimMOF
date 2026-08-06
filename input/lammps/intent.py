from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


INTENT_FIELDS = {
    "state_change_required",
    "dynamic_sampling_required",
    "minimization_required",
    "deformation_required",
    "cell_flexibility_required",
    "trajectory_required",
    "unwrapped_coordinates_required",
    "output_required",
    "ensemble",
    "temperature_schedule_K",
    "observables",
    "requested_output_files",
    "system_scope",
    "rationale",
    "normalization_warnings",
}


def unknown_lammps_intent() -> Dict[str, Any]:
    return {
        "state_change_required": None,
        "dynamic_sampling_required": None,
        "minimization_required": None,
        "deformation_required": None,
        "cell_flexibility_required": None,
        "trajectory_required": None,
        "unwrapped_coordinates_required": None,
        "output_required": None,
        "ensemble": "unspecified",
        "temperature_schedule_K": [],
        "observables": [],
        "requested_output_files": [],
        "system_scope": "unspecified",
        "rationale": "intent planner did not return valid structured output",
        "normalization_warnings": [],
        "planner_status": "fallback_unknown",
    }


def build_lammps_intent_messages(
    *,
    simulation_description: str,
    property_name: str,
    capabilities: Dict[str, Any],
) -> List[Any]:
    from langchain.schema import HumanMessage, SystemMessage

    capability_summary = {
        key: capabilities.get(key)
        for key in (
            "atom_style",
            "triclinic_box",
            "has_molecule_ids",
            "declared_groups",
        )
    }
    system = (
        "You infer the physical calculation intent of a requested LAMMPS Run "
        "Section. Describe what the calculation must accomplish without "
        "selecting or naming any LAMMPS command, compute, fix, or dump style. "
        "Return strict JSON only."
    )
    human = "\n".join(
        [
            "REQUEST:",
            simulation_description,
            "",
            "PROPERTY LABEL:",
            property_name,
            "",
            "SYSTEM CAPABILITIES:",
            json.dumps(capability_summary, ensure_ascii=True, sort_keys=True),
            "",
            "Return exactly one JSON object with these fields:",
            "{",
            '  "state_change_required": true | false | null,',
            '  "dynamic_sampling_required": true | false | null,',
            '  "minimization_required": true | false | null,',
            '  "deformation_required": true | false | null,',
            '  "cell_flexibility_required": true | false | null,',
            '  "trajectory_required": true | false | null,',
            '  "unwrapped_coordinates_required": true | false | null,',
            '  "output_required": true | false | null,',
            '  "ensemble": "NVE|NVT|NPT|NPH|unspecified",',
            '  "temperature_schedule_K": [number, ...],',
            '  "observables": ["physical quantity", ...],',
            '  "requested_output_files": ["filename", ...],',
            '  "system_scope": "all|framework|guest|host_guest|unspecified",',
            '  "rationale": "one concise sentence"',
            "}",
            "",
            "Field semantics:",
            "- state_change_required is true whenever the requested calculation "
            "needs a positive-length time evolution, velocity initialization, "
            "energy minimization, or coordinate/cell modification. It does not "
            "mean a phase transition.",
            "- dynamic_sampling_required is true when time evolution must "
            "produce samples for the result.",
            "- trajectory_required is true only when atom or molecule "
            "coordinates must be written as a trajectory.",
            "- unwrapped_coordinates_required can be true only when "
            "trajectory_required is also true and boundary-crossing "
            "displacements are needed.",
            "",
            "Do not include LAMMPS command names in any field.",
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


def _nullable_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    lowered = str(value).strip().lower()
    if lowered in {"true", "yes", "1"}:
        return True
    if lowered in {"false", "no", "0"}:
        return False
    return None


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [
        str(item).strip()
        for item in value
        if str(item).strip()
    ]


def normalize_lammps_intent(raw: Dict[str, Any]) -> Dict[str, Any]:
    intent = unknown_lammps_intent()
    warnings: List[str] = []
    for key in (
        "state_change_required",
        "dynamic_sampling_required",
        "minimization_required",
        "deformation_required",
        "cell_flexibility_required",
        "trajectory_required",
        "unwrapped_coordinates_required",
        "output_required",
    ):
        intent[key] = _nullable_bool(raw.get(key))

    state_changing_requirements = (
        "dynamic_sampling_required",
        "minimization_required",
        "deformation_required",
        "cell_flexibility_required",
    )
    if any(intent.get(key) is True for key in state_changing_requirements):
        if intent.get("state_change_required") is not True:
            warnings.append(
                "state_change_required promoted because the calculation "
                "requires dynamics, minimization, deformation, or cell motion"
            )
        intent["state_change_required"] = True
    if (
        intent.get("trajectory_required") is False
        and intent.get("unwrapped_coordinates_required") is True
    ):
        intent["unwrapped_coordinates_required"] = False
        warnings.append(
            "unwrapped_coordinates_required cleared because no trajectory "
            "is required"
        )

    ensemble = str(raw.get("ensemble") or "unspecified").strip().upper()
    intent["ensemble"] = (
        ensemble if ensemble in {"NVE", "NVT", "NPT", "NPH"} else "unspecified"
    )
    temperatures = raw.get("temperature_schedule_K")
    intent["temperature_schedule_K"] = []
    if isinstance(temperatures, list):
        for value in temperatures:
            try:
                intent["temperature_schedule_K"].append(float(value))
            except (TypeError, ValueError):
                continue
    intent["observables"] = _string_list(raw.get("observables"))
    intent["requested_output_files"] = _string_list(
        raw.get("requested_output_files")
    )
    scope = str(raw.get("system_scope") or "unspecified").strip().lower()
    intent["system_scope"] = (
        scope
        if scope in {"all", "framework", "guest", "host_guest"}
        else "unspecified"
    )
    intent["rationale"] = str(raw.get("rationale") or "").strip()
    intent["normalization_warnings"] = warnings
    intent["planner_status"] = "ok"
    return {
        key: value
        for key, value in intent.items()
        if key in INTENT_FIELDS or key == "planner_status"
    }


def infer_lammps_intent(
    llm: Any,
    *,
    simulation_description: str,
    property_name: str,
    capabilities: Dict[str, Any],
) -> Dict[str, Any]:
    messages = build_lammps_intent_messages(
        simulation_description=simulation_description,
        property_name=property_name,
        capabilities=capabilities,
    )
    try:
        response = llm.invoke(messages)
    except Exception as exc:
        intent = unknown_lammps_intent()
        intent["planner_status"] = "fallback_error"
        intent["rationale"] = (
            f"intent planner failed with {type(exc).__name__}"
        )
        return intent
    parsed = _extract_json_object(str(getattr(response, "content", "")))
    if parsed is None:
        return unknown_lammps_intent()
    return normalize_lammps_intent(parsed)


__all__ = [
    "build_lammps_intent_messages",
    "infer_lammps_intent",
    "normalize_lammps_intent",
    "unknown_lammps_intent",
]
