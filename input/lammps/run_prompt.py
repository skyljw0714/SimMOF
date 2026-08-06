from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Union


MOLECULE_ATOM_STYLES = {
    "angle",
    "bond",
    "full",
    "molecular",
}
CHARGE_ATOM_STYLES = {
    "charge",
    "dipole",
    "electron",
    "full",
}

GENERAL_EASY_FEW_SHOT_EXAMPLES = """
GENERAL EASY FEW-SHOT EXAMPLES
These examples demonstrate command ordering, ID lifecycle, and executable
Run-Section structure only. They are not task templates. Do not copy a stage,
group, numeric value, or output command unless the requested objective requires
it and the capability contract permits it.

EASY FEW-SHOT EXAMPLE 1
Objective: Relax atomic coordinates at fixed cell dimensions.
Run Section:
min_style cg
minimize 1.0e-8 1.0e-10 1000 10000

EASY FEW-SHOT EXAMPLE 2
Objective: Equilibrate the existing system at 300 K with NVT dynamics and report
basic thermodynamic quantities.
Run Section:
velocity all create 300.0 4928459 mom yes rot yes dist gaussian
fix integrate all nvt temp 300.0 300.0 100.0
thermo 100
thermo_style custom step temp pe etotal
run 10000
unfix integrate

EASY FEW-SHOT EXAMPLE 3
Objective: Save an unwrapped atomic trajectory during an already requested
dynamics stage.
Run Section:
dump trajectory all custom 100 trajectory.lammpstrj id type xu yu zu
dump_modify trajectory sort id
run 10000
undump trajectory
""".strip()


def _read_if_present(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _first_atom_style(texts: Iterable[str]) -> Optional[str]:
    for text in texts:
        match = re.search(
            r"(?im)^\s*atom_style[ \t]+([^\s#]+(?:[ \t]+[^\n#]+)?)",
            text,
        )
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()
        match = re.search(r"(?im)^\s*Atoms\s*#\s*([A-Za-z0-9_/+-]+)", text)
        if match:
            return match.group(1).strip()
    return None


def _style_has_feature(
    atom_style: Optional[str],
    supported_styles: Set[str],
) -> Optional[bool]:
    if not atom_style:
        return None
    tokens = atom_style.lower().split()
    if tokens[0] == "hybrid":
        return any(token in supported_styles for token in tokens[1:])
    return tokens[0] in supported_styles


def _parse_groups(group_definitions: str) -> List[str]:
    groups: List[str] = []
    for line in (group_definitions or "").splitlines():
        match = re.match(r"^\s*group\s+(\S+)\s+", line)
        if match and match.group(1) not in groups:
            groups.append(match.group(1))
    return groups


def _atom_type_populations(
    data_text: str,
    atom_style: Optional[str],
) -> Dict[int, int]:
    style = str(atom_style or "").lower().split()
    base_style = style[0] if style else ""
    type_index = (
        2
        if base_style in {"angle", "bond", "full", "molecular"}
        else 1
    )
    populations: Dict[int, int] = {}
    in_atoms = False
    saw_atom = False
    for raw_line in data_text.splitlines():
        stripped = raw_line.split("#", 1)[0].strip()
        if not in_atoms:
            if re.match(r"^Atoms(?:\s|$)", raw_line.strip(), re.IGNORECASE):
                in_atoms = True
            continue
        if not stripped:
            continue
        tokens = stripped.split()
        if not tokens[0].lstrip("+-").isdigit():
            if saw_atom:
                break
            continue
        if len(tokens) <= type_index:
            continue
        try:
            atom_type = int(tokens[type_index])
        except ValueError:
            continue
        populations[atom_type] = populations.get(atom_type, 0) + 1
        saw_atom = True
    return populations


def _expand_type_tokens(
    tokens: Iterable[str],
    atom_type_count: Optional[int],
) -> Set[int]:
    selected: Set[int] = set()
    maximum = int(atom_type_count or 0)
    for token in tokens:
        value = token.strip()
        if not value:
            continue
        if value == "*" and maximum:
            selected.update(range(1, maximum + 1))
            continue
        if ":" in value:
            lower_text, upper_text = value.split(":", 1)
            try:
                lower = int(lower_text) if lower_text else 1
                upper = int(upper_text) if upper_text else maximum
            except ValueError:
                continue
            if upper >= lower:
                selected.update(range(lower, upper + 1))
            continue
        try:
            selected.add(int(value))
        except ValueError:
            continue
    return selected


def _declared_group_atom_counts(
    group_definitions: str,
    populations: Dict[int, int],
    atom_type_count: Optional[int],
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for raw_line in (group_definitions or "").splitlines():
        match = re.match(
            r"^\s*group\s+(\S+)\s+type\s+(.+?)(?:\s*#.*)?$",
            raw_line,
            flags=re.IGNORECASE,
        )
        if not match:
            continue
        atom_types = _expand_type_tokens(
            match.group(2).split(),
            atom_type_count,
        )
        counts[match.group(1)] = sum(
            populations.get(atom_type, 0)
            for atom_type in atom_types
        )
    return counts


def _triclinic_tilt_fractions(data_text: str) -> Dict[str, float]:
    bounds: Dict[str, float] = {}
    for axis in ("x", "y", "z"):
        match = re.search(
            rf"(?im)^\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+"
            rf"{axis}lo\s+{axis}hi\b",
            data_text,
        )
        if match:
            bounds[axis] = abs(float(match.group(2)) - float(match.group(1)))
    tilt_match = re.search(
        r"(?im)^\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)"
        r"\s+xy\s+xz\s+yz\b",
        data_text,
    )
    if not tilt_match:
        return {}
    tilts = {
        "xy": float(tilt_match.group(1)),
        "xz": float(tilt_match.group(2)),
        "yz": float(tilt_match.group(3)),
    }
    denominators = {
        "xy": bounds.get("x"),
        "xz": bounds.get("x"),
        "yz": bounds.get("y"),
    }
    return {
        key: round(abs(value) / denominators[key], 6)
        for key, value in tilts.items()
        if denominators.get(key)
    }


def extract_lammps_run_capabilities(
    output_file: Union[str, Path],
    group_definitions: str = "",
) -> Dict[str, Any]:
    output_path = Path(output_file)
    work_dir = output_path.parent
    system_text = _read_if_present(output_path)
    init_text = _read_if_present(work_dir / "system.in.init")
    data_text = _read_if_present(work_dir / "system.data")
    if not data_text:
        data_text = _read_if_present(work_dir / "data.lammps")
    if not data_text:
        read_data_match = re.search(
            r"(?im)^\s*read_data\s+[\"']?([^\"'\s]+)",
            system_text,
        )
        if read_data_match:
            data_text = _read_if_present(work_dir / read_data_match.group(1))

    atom_style = _first_atom_style((system_text, init_text, data_text))
    has_molecule_ids = _style_has_feature(atom_style, MOLECULE_ATOM_STYLES)
    has_charge_field = _style_has_feature(atom_style, CHARGE_ATOM_STYLES)

    atom_types_match = re.search(
        r"(?im)^\s*(\d+)\s+atom\s+types\b",
        data_text,
    )
    atom_type_count = (
        int(atom_types_match.group(1))
        if atom_types_match
        else None
    )
    if atom_type_count is None:
        create_box_match = re.search(
            r"(?im)^\s*create_box\s+(\d+)\b",
            system_text + "\n" + init_text,
        )
        if create_box_match:
            atom_type_count = int(create_box_match.group(1))

    if data_text:
        triclinic_box: Optional[bool] = bool(
            re.search(
                r"(?im)^\s*[-+0-9.eE]+\s+[-+0-9.eE]+\s+[-+0-9.eE]+"
                r"\s+xy\s+xz\s+yz\b",
                data_text,
            )
        )
    else:
        triclinic_box = None

    if triclinic_box is not True:
        combined = system_text + "\n" + init_text
        if re.search(r"(?im)^\s*(?:region\s+\S+\s+prism|create_box\b.*\btriclinic\b)", combined):
            triclinic_box = True
        elif re.search(r"(?im)^\s*region\s+\S+\s+block\b", combined):
            triclinic_box = False

    atom_type_populations = _atom_type_populations(data_text, atom_style)
    group_atom_counts = _declared_group_atom_counts(
        group_definitions,
        atom_type_populations,
        atom_type_count,
    )
    tilt_fractions = _triclinic_tilt_fractions(data_text)

    fields = ["id", "type", "x", "y", "z"]
    if has_molecule_ids is True:
        fields.append("mol")
    if has_charge_field is True:
        fields.append("q")

    return {
        "target_lammps_version": os.getenv(
            "SIMMOF_LAMMPS_TARGET_VERSION",
            "3 Mar 2020",
        ),
        "atom_style": atom_style or "unknown",
        "triclinic_box": triclinic_box,
        "has_molecule_ids": has_molecule_ids,
        "has_charge_field": has_charge_field,
        "available_per_atom_fields": fields,
        "atom_type_count": atom_type_count,
        "declared_groups": _parse_groups(group_definitions),
        "declared_group_atom_counts": group_atom_counts,
        "triclinic_tilt_fractions": tilt_fractions,
        "maximum_triclinic_tilt_fraction": (
            max(tilt_fractions.values()) if tilt_fractions else None
        ),
        "capability_policy": (
            "Treat null or unknown capabilities as unavailable unless official "
            "evidence and the existing input establish them."
        ),
    }


def build_minimal_runsection_prompt(
    *,
    simulation_description: str,
    property_name: str,
    group_definitions: str,
    official_command_hints: str,
    rag_summaries: str,
    capabilities: Dict[str, Any],
    intent_spec: Optional[Dict[str, Any]] = None,
) -> str:
    capability_json = json.dumps(
        capabilities,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    intent_json = json.dumps(
        intent_spec or {},
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    return f"""
You are an expert in writing LAMMPS input scripts.

Task:
Generate ONLY the Run Section commands to append to an existing `system.in`.
Earlier sections already define the simulation box, atoms, force-field
coefficients, `read_data`, `pair_style`, and `kspace_style` when required.

REQUESTED OBJECTIVE
{simulation_description}

REQUESTED PROPERTY LABEL
{property_name}

STRUCTURED CALCULATION INTENT
{intent_json}

SYSTEM CAPABILITY CONTRACT
{capability_json}

GROUP COMMANDS PREPENDED BY THE CALLER
{group_definitions or "(none)"}

These commands are inserted before your response. Do not emit or redefine them.

MINIMAL RUN-SECTION SCAFFOLD
Include only stages that are necessary for the requested objective:
1. Preparation required by the selected calculation.
2. Definitions of the requested observable.
3. Averaging or output required to expose that observable.
4. A run or minimization only when required to produce the requested result.
5. Cleanup only for IDs created by this Run Section.

Do not fill an optional stage merely because it appears in this scaffold.

{GENERAL_EASY_FEW_SHOT_EXAMPLES}

VALIDATION CONTRACT
- Output only executable LAMMPS commands, with no markdown or prose.
- Do not redefine the simulation box, atoms, force field, `pair_style`,
  `pair_coeff`, or `kspace_style`.
- Use only the prepended groups, or `all` when the requested operation genuinely
  applies to the whole system. Do not emit group commands or invent groups.
- For an observable that explicitly spans two or more declared groups, use a
  prepended union group when one exists; otherwise use `all` so neither side of
  the observable is excluded by the command's group-ID.
- Observable scope and state-change scope are independent. Using `all` for a
  pair observable does not authorize `velocity`, integration, thermostatting,
  or barostatting on `all`.
- For a host-guest observable, change the guest state only unless the objective
  explicitly requests framework motion or a flexible-framework ensemble.
- Preserve all numeric values explicitly supplied by the user.
- Do not assume a temperature, ensemble, thermostat, minimization, trajectory,
  or equilibration stage when the objective does not require one.
- Do not reference a per-atom field unless the capability contract confirms it.
- Define every compute, fix, dump, and variable ID before it is consumed.
- Keep every ID unique and remove it only after its final use.
- Treat retrieved commands as optional evidence candidates. Do not instantiate
  a candidate unless it is necessary for the structured intent and its typed
  dependencies are satisfied.
- Ensure producer and consumer data shapes are compatible: global, per-atom,
  local, scalar, vector, array, and per-chunk values are not interchangeable.
- Apply official restrictions for the detected box geometry and atom style.
- Use syntax valid for `target_lammps_version`; ignore evidence that requires a
  newer version.
- Prefer the mandatory positional syntax supported by `target_lammps_version`.
  Omit optional keyword clauses unless the objective requires them and the
  evidence confirms that they are available in the target version.
- The generated commands must create the property signal or output explicitly
  requested by the objective.
- Prefer the smallest command set that satisfies these requirements.

OFFICIAL LAMMPS EVIDENCE
{official_command_hints or "(none)"}

OPTIONAL INTERNAL OR LITERATURE NOTES
{rag_summaries or "(none)"}

EVIDENCE POLICY
- Use official evidence for command syntax, restrictions, defaults, output
  shape, and related-command requirements.
- Treat the `LAMMPS EVIDENCE PLAN` as an advisory candidate set, not an allowed
  list and not a required command list. The structured calculation intent is
  authoritative.
- Preserve a simpler standard command when it already satisfies the intent.
  Do not replace it with a specialized candidate merely because that candidate
  was retrieved.
- Connect producers and consumers using only reference forms documented by the
  consumer, such as `c_ID`, `c_ID[I]`, `f_ID`, or `v_name`. A bare thermo
  keyword is not interchangeable with one of these references.
- For `fix ave/time`, every sampled value must use a documented `c_`, `f_`, or
  `v_` reference form. Define equal-style variables for built-in thermo values
  instead of passing bare names such as `pe` or `vol`.
- When a producer returns an array or vector, select the documented component
  or wildcard reference and consumer mode required by that data shape. Do not
  pass an entire array where a scalar input is required.
- If a requested built-in thermo quantity is not directly accepted by the
  consumer syntax, expose it through a documented reference type, such as an
  equal-style variable, before consuming it.
- Let the structured intent, not retrieval, determine whether state changes are
  required. Evidence may clarify syntax and restrictions but cannot authorize a
  new physical protocol by itself.
- Apply state changes only to the minimum group and stage required by the
  objective. Do not default to `all` when the objective targets a declared
  subgroup.
- Do not copy an example's system-specific groups, atom types, filenames, or
  numeric parameters unless they match the request and capability contract.
- Never copy an example's velocity, ensemble, integration, or equilibration
  protocol merely because the example contains the selected observable.
- If evidence conflicts with the capability or validation contract, follow the
  capability and validation contract.

Return ONLY the Run Section commands.
""".strip()


def build_advisory_revision_prompt(
    *,
    baseline_prompt: str,
    baseline_script: str,
    intent_spec: Dict[str, Any],
    official_command_hints: str,
    rag_summaries: str,
    dependency_graph: Dict[str, Any],
    validation_errors: List[str],
) -> str:
    return "\n".join(
        [
            baseline_prompt,
            "",
            "BASELINE RUN SECTION",
            baseline_script,
            "",
            "ADVISORY OFFICIAL EVIDENCE",
            official_command_hints or "(none)",
            "",
            "OPTIONAL INTERNAL OR LITERATURE NOTES",
            rag_summaries or "(none)",
            "",
            "TYPED ADVISORY DEPENDENCY GRAPH",
            json.dumps(
                dependency_graph or {},
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            ),
            "",
            "STRUCTURED CALCULATION INTENT",
            json.dumps(
                intent_spec or {},
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            ),
            "",
            "VALIDATION ERRORS IN THE BASELINE",
            *[f"- {error}" for error in validation_errors],
            "",
            "Revise the baseline only where needed to fix the listed errors.",
            "Evidence commands are candidates, not mandatory commands.",
            "Keep an existing standard command unless a validated dependency or "
            "intent requirement makes replacement necessary.",
            "Return a complete replacement Run Section containing executable "
            "LAMMPS commands only.",
        ]
    )


def build_scientific_evidence_revision_prompt(
    *,
    baseline_prompt: str,
    baseline_script: str,
    scientific_plan: Dict[str, Any],
    official_command_hints: str,
    dependency_graph: Dict[str, Any],
    baseline_validation_errors: List[str],
) -> str:
    return "\n".join(
        [
            baseline_prompt,
            "",
            "EVIDENCE-FREE BASELINE RUN SECTION",
            baseline_script,
            "",
            "SCIENTIFIC CALCULATION PLAN",
            json.dumps(
                scientific_plan or {},
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            ),
            "",
            "OFFICIAL LAMMPS EVIDENCE",
            official_command_hints or "(none)",
            "",
            "TYPED ADVISORY DEPENDENCY GRAPH",
            json.dumps(
                dependency_graph or {},
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            ),
            "",
            "STATIC ISSUES FOUND IN THE BASELINE",
            *(
                [f"- {error}" for error in baseline_validation_errors]
                or ["- none detected; perform a scientific and executable audit"]
            ),
            "",
            "SCIENTIFIC REVISION POLICY",
            "- Treat the baseline as a candidate, not as ground truth.",
            "- Revise only when the physical estimator, command dependency "
            "graph, initial-state domain, sampling, or output contract exposes "
            "a concrete gap.",
            "- Preserve baseline parts that already satisfy the objective.",
            "- Use the selected official evidence as support, not as a mandatory "
            "command list.",
            "- Ensure every expression is defined at its first evaluation and "
            "every producer-consumer reference has the documented data shape.",
            "- Treat compute currency as part of the command graph. A zero-step "
            "run does not make an arbitrary compute current unless that compute "
            "is scheduled for evaluation during the run. When a variable or "
            "print consumes a compute between state-changing stages, schedule "
            "the compute in a documented consumer at the evaluation step.",
            "- Keep state evolution and observable scope physically consistent "
            "with the requested groups and system capabilities.",
            "- Before pressure-coupled or finite-temperature dynamics, check "
            "whether the supplied input establishes a relaxed, numerically "
            "stable initial state. If it does not, use a conservative generic "
            "stabilization stage and integration/neighbor settings justified by "
            "the force field before activating the requested production ensemble.",
            "- Activate only cell degrees of freedom needed by the objective. "
            "Do not enable shear or tilt fluctuations merely because the box "
            "can be represented as triclinic.",
            "- Prefer writing the primary sampled signal when an on-the-fly "
            "derived expression would require a fit, integral, or nonzero elapsed "
            "time.",
            "- Never invent a reduction, fitting, or array-index expression that "
            "is not supported by the official evidence. If the evidence exposes "
            "an array but does not document an in-script scalar reduction, write "
            "the documented array or components for post-processing.",
            "- The scientific plan is authoritative about whether a requested "
            "derived quantity belongs in post-processing. Do not create a "
            "nominal derived-output file using an unvalidated shortcut merely "
            "to match a requested filename.",
            "- Keep the Run Section compact. Repeated states may use a documented "
            "loop, but do not add in-script regression, differencing, or "
            "integration when the plan assigns that analysis to post-processing.",
            "- Return a complete replacement Run Section containing executable "
            "LAMMPS commands only. Do not return reasoning or markdown.",
        ]
    )


__all__ = [
    "build_advisory_revision_prompt",
    "build_scientific_evidence_revision_prompt",
    "build_minimal_runsection_prompt",
    "extract_lammps_run_capabilities",
]
