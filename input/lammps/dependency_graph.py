from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


COMMAND_CONTRACTS: Dict[str, Dict[str, Any]] = {
    "compute heat/flux": {
        "produces": "global_vector",
        "requires_compute_styles": [
            "ke/atom",
            "pe/atom",
            "stress/atom",
        ],
    },
    "compute gyration/chunk": {
        "produces": "global_vector",
        "requires_compute_styles": ["chunk/atom"],
    },
    "compute gyration/shape/chunk": {
        "produces": "global_array",
        "requires_compute_styles": ["gyration/chunk"],
    },
    "compute msd/chunk": {
        "produces": "global_array",
        "requires_compute_styles": ["chunk/atom"],
    },
    "compute vacf/chunk": {
        "produces": "global_array",
        "requires_compute_styles": ["chunk/atom"],
    },
    "compute rdf": {
        "produces": "global_array",
        "requires_compute_styles": [],
    },
    "compute stress/atom": {
        "produces": "per_atom_array",
        "requires_compute_styles": [],
        "requires_temperature_argument": True,
    },
    "fix ave/chunk": {
        "accepts": ["per_atom_vector", "per_atom_array"],
        "requires_compute_styles": ["chunk/atom"],
    },
    "fix ave/correlate": {
        "accepts": ["global_scalar", "global_vector"],
        "requires_compute_styles": [],
    },
    "fix ave/time": {
        "accepts": ["global_scalar", "global_vector", "global_array"],
        "requires_compute_styles": [],
    },
}


def build_advisory_dependency_graph(
    evidence_plan: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    plan = evidence_plan or {}
    candidates = []
    seen = set()
    for section in ("core", "dependencies"):
        for item in plan.get(section) or []:
            name = str(item.get("command_name") or "").strip().lower()
            if not name or name in seen:
                continue
            seen.add(name)
            contract = COMMAND_CONTRACTS.get(name, {})
            candidates.append(
                {
                    "command_name": name,
                    "role": item.get("role") or "guidance",
                    "source_chunk_id": item.get("source_chunk_id") or "",
                    "reason": item.get("reason") or "",
                    "required": False,
                    "produces": item.get("produces")
                    or ([contract["produces"]] if contract.get("produces") else []),
                    "accepts": item.get("accepts")
                    or contract.get("accepts")
                    or [],
                }
            )

    edges = []
    candidate_names = {item["command_name"] for item in candidates}
    for name in list(candidate_names):
        contract = COMMAND_CONTRACTS.get(name) or {}
        for style in contract.get("requires_compute_styles") or []:
            dependency = f"compute {style}"
            edges.append(
                {
                    "consumer": name,
                    "producer": dependency,
                    "relation": "requires_compute_style",
                    "required_style": style,
                }
            )
    return {
        "candidate_commands": candidates,
        "typed_edges": edges,
        "policy": (
            "Candidates are advisory. A command may be selected only when its "
            "typed dependencies can be satisfied by the generated script."
        ),
    }


def _logical_lines(script: str) -> List[str]:
    lines: List[str] = []
    pending = ""
    for raw in (script or "").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        pending = f"{pending} {line}".strip() if pending else line
        if pending.endswith("&"):
            pending = pending[:-1].rstrip()
            continue
        lines.append(pending)
        pending = ""
    if pending:
        lines.append(pending)
    return lines


def _definitions(
    script: str,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], set]:
    computes: Dict[str, Dict[str, Any]] = {}
    fixes: Dict[str, Dict[str, Any]] = {}
    variables = set()
    for line in _logical_lines(script):
        tokens = line.split()
        if len(tokens) >= 4 and tokens[0].lower() == "compute":
            computes[tokens[1]] = {
                "group": tokens[2],
                "style": tokens[3].lower(),
                "args": tokens[4:],
                "line": line,
            }
        elif len(tokens) >= 4 and tokens[0].lower() == "fix":
            fixes[tokens[1]] = {
                "group": tokens[2],
                "style": tokens[3].lower(),
                "args": tokens[4:],
                "line": line,
            }
        elif len(tokens) >= 3 and tokens[0].lower() == "variable":
            variables.add(tokens[1])
    return computes, fixes, variables


def _require_compute_style(
    *,
    errors: List[str],
    owner: str,
    compute_id: Optional[str],
    required_style: str,
    computes: Dict[str, Dict[str, Any]],
) -> None:
    if not compute_id:
        errors.append(f"{owner} requires a compute ID with style {required_style}")
        return
    definition = computes.get(compute_id)
    if not definition:
        errors.append(
            f"{owner} references undefined compute ID {compute_id}"
        )
        return
    if definition["style"] != required_style:
        errors.append(
            f"{owner} requires compute style {required_style}, but "
            f"{compute_id} uses {definition['style']}"
        )


def validate_lammps_command_dependencies(
    script: str,
    *,
    capabilities: Optional[Dict[str, Any]] = None,
) -> List[str]:
    errors: List[str] = []
    computes, fixes, variables = _definitions(script)
    triclinic = bool((capabilities or {}).get("triclinic_box"))

    for compute_id, definition in computes.items():
        style = definition["style"]
        args = definition["args"]
        owner = f"compute {compute_id} ({style})"
        if style in {"gyration/chunk", "msd/chunk", "vacf/chunk"}:
            _require_compute_style(
                errors=errors,
                owner=owner,
                compute_id=args[0] if args else None,
                required_style="chunk/atom",
                computes=computes,
            )
        elif style == "gyration/shape/chunk":
            _require_compute_style(
                errors=errors,
                owner=owner,
                compute_id=args[0] if args else None,
                required_style="gyration/chunk",
                computes=computes,
            )
        elif style == "heat/flux":
            required = ("ke/atom", "pe/atom", "stress/atom")
            for index, required_style in enumerate(required):
                _require_compute_style(
                    errors=errors,
                    owner=owner,
                    compute_id=args[index] if len(args) > index else None,
                    required_style=required_style,
                    computes=computes,
                )
        elif style == "stress/atom":
            if not args:
                errors.append(
                    f"{owner} requires a temperature compute ID or NULL"
                )
            elif args[0].upper() != "NULL":
                _require_compute_style(
                    errors=errors,
                    owner=owner,
                    compute_id=args[0],
                    required_style="temp",
                    computes=computes,
                )
        elif style == "pressure":
            temperature_id = args[0].upper() if args else ""
            contributions = {
                token.lower()
                for token in args[1:]
            }
            if temperature_id == "NULL" and (
                not contributions or "ke" in contributions
            ):
                errors.append(
                    f"{owner} uses NULL temperature but includes the kinetic "
                    "pressure contribution; specify a temperature compute or "
                    "request virial contributions only"
                )
        elif style == "rdf":
            pair_tokens = args[1:]
            if "cutoff" in [token.lower() for token in pair_tokens]:
                cutoff_index = [
                    token.lower() for token in pair_tokens
                ].index("cutoff")
                pair_tokens = pair_tokens[:cutoff_index]
            if len(pair_tokens) % 2:
                errors.append(
                    f"{owner} requires complete itype/jtype pairs; "
                    f"found {len(pair_tokens)} type-list tokens"
                )
        elif (
            style == "chunk/atom"
            and triclinic
            and any(arg.startswith("bin/") for arg in args)
            and not (
                "units" in [arg.lower() for arg in args]
                and "reduced" in [arg.lower() for arg in args]
            )
        ):
            errors.append(
                f"{owner} uses spatial bins in a triclinic box and requires "
                "units reduced"
            )

    for fix_id, definition in fixes.items():
        style = definition["style"]
        args = definition["args"]
        owner = f"fix {fix_id} ({style})"
        if style == "ave/chunk":
            chunk_id = args[3] if len(args) > 3 else None
            _require_compute_style(
                errors=errors,
                owner=owner,
                compute_id=chunk_id,
                required_style="chunk/atom",
                computes=computes,
            )
        if style in {"nvt", "npt", "nph"}:
            group_counts = (
                (capabilities or {}).get("declared_group_atom_counts") or {}
            )
            group_count = group_counts.get(definition["group"])
            if group_count is not None and int(group_count) <= 1:
                errors.append(
                    f"{owner} thermostats or barostats group "
                    f"{definition['group']} with only {group_count} atom; "
                    "its temperature degrees of freedom are not usable"
                )

    for line in _logical_lines(script):
        for prefix, identifier in re.findall(
            r"(?<![A-Za-z0-9_])([cfv])_([A-Za-z0-9_.]+)",
            line,
        ):
            if prefix == "c" and identifier not in computes:
                errors.append(f"reference c_{identifier} uses an undefined compute")
            elif prefix == "f" and identifier not in fixes:
                errors.append(f"reference f_{identifier} uses an undefined fix")
            elif prefix == "v" and identifier not in variables:
                errors.append(f"reference v_{identifier} uses an undefined variable")
    return list(dict.fromkeys(errors))


def _has_positive_run(script: str) -> bool:
    for line in _logical_lines(script):
        match = re.match(r"run\s+(\S+)", line, flags=re.IGNORECASE)
        if not match:
            continue
        value = match.group(1)
        try:
            if float(value) > 0:
                return True
        except ValueError:
            return True
    return False


def validate_lammps_intent_coverage(
    script: str,
    intent: Optional[Dict[str, Any]],
) -> List[str]:
    if not intent or intent.get("planner_status") != "ok":
        return []
    lower = script.lower()
    errors: List[str] = []
    has_minimize = bool(re.search(r"(?im)^\s*minimize\b", script))
    has_positive_run = _has_positive_run(script)

    if intent.get("state_change_required") is True and not (
        has_positive_run or has_minimize
    ):
        errors.append("intent requires state change but no positive run or minimization exists")
    if intent.get("state_change_required") is False and (
        has_positive_run
        or has_minimize
        or bool(
            re.search(
                r"(?im)^\s*(?:change_box|fix\s+\S+\s+\S+\s+deform)\b",
                script,
            )
        )
    ):
        errors.append("intent forbids physical state changes")
    if intent.get("dynamic_sampling_required") is True and not has_positive_run:
        errors.append("intent requires dynamic sampling but no positive run exists")
    if intent.get("minimization_required") is True and not has_minimize:
        errors.append("intent requires minimization")
    if intent.get("deformation_required") is True and not re.search(
        r"(?im)^\s*(?:change_box|fix\s+\S+\s+\S+\s+deform)\b",
        script,
    ):
        errors.append("intent requires cell or coordinate deformation")
    if intent.get("cell_flexibility_required") is True and not re.search(
        r"(?im)^\s*(?:fix\s+\S+\s+\S+\s+(?:npt|nph|box/relax)|change_box)\b",
        script,
    ):
        errors.append("intent requires flexible-cell or box-changing dynamics")

    ensemble = str(intent.get("ensemble") or "unspecified").lower()
    if ensemble in {"nve", "nvt", "npt", "nph"} and not re.search(
        rf"(?im)^\s*fix\s+\S+\s+\S+\s+{re.escape(ensemble)}\b",
        script,
    ):
        errors.append(f"intent requires {ensemble.upper()} dynamics")

    temperatures = intent.get("temperature_schedule_K") or []
    if len(temperatures) > 1:
        has_loop = bool(
            re.search(r"(?im)^\s*next\b", script)
            and re.search(r"(?im)^\s*jump\b", script)
        )
        explicit = sum(
            bool(
                re.search(
                    rf"(?<![0-9.]){value:g}(?:\.0+)?(?![0-9.])",
                    script,
                )
            )
            for value in temperatures
        )
        if not has_loop and explicit < len(temperatures):
            errors.append("intent requires a multi-temperature schedule")

    if intent.get("trajectory_required") is True and not re.search(
        r"(?im)^\s*dump\s+\S+\s+\S+\s+custom\b",
        script,
    ):
        errors.append("intent requires a trajectory output")
    if intent.get("unwrapped_coordinates_required") is True and not all(
        token in lower for token in ("xu", "yu", "zu")
    ):
        errors.append("intent requires unwrapped coordinates xu yu zu")
    if intent.get("output_required") is True and not re.search(
        (
            r"(?im)^\s*(?:print|dump|write_data|write_restart|thermo_style)\b"
            r"|^\s*fix\s+\S+\s+\S+\s+(?:ave/\S+|print)\b"
        ),
        script,
    ):
        errors.append("intent requires an exposed result or output")

    for filename in intent.get("requested_output_files") or []:
        if filename and filename.lower() not in lower:
            errors.append(f"intent requires output file {filename}")
    return list(dict.fromkeys(errors))


__all__ = [
    "COMMAND_CONTRACTS",
    "build_advisory_dependency_graph",
    "validate_lammps_command_dependencies",
    "validate_lammps_intent_coverage",
]
