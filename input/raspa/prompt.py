import json
from typing import Optional

RASPA_GENERIC_TEMPLATE = """\
# SimulationType                ...
# NumberOfCycles                ...
# NumberOfInitializationCycles  ...
PrintEvery                    1000
PrintForcefieldToOutput       no

Forcefield                    {forcefield}
CutOffVDW                     12.8

{charge_block}

Framework 0
FrameworkName                 {framework_name}
UseChargesFromCIFFile         {use_charges_from_cif}
UnitCells                     {unitcell_x} {unitcell_y} {unitcell_z}
ExternalTemperature           {temperature}
# ExternalPressure              ...

{component_block}
"""


def render_raspa_format(params: dict) -> str:
    mol_name = params.get("molecule_name") or ""
    mol_def  = params.get("molecule_definition") or ""

    if mol_name:
        component_block = (
            f"Component 0 MoleculeName      {mol_name}\n"
            f"    MoleculeDefinition        {mol_def}\n"
            f"    # TranslationProbability    ...\n"
            f"    # ReinsertionProbability    ...\n"
            f"    # RotationProbability       ...\n"
            f"    # RegrowProbability         ...\n"
            f"    # SwapProbability           ...\n"
            f"    # WidomProbability          ...\n"
            f"    # CreateNumberOfMolecules   ..."
        )
    else:
        component_block = (
            "# Component 0 MoleculeName      ...\n"
            "#     MoleculeDefinition        ...\n"
            "#     TranslationProbability    ...\n"
            "#     ReinsertionProbability    ...\n"
            "#     RotationProbability       ...\n"
            "#     RegrowProbability         ...\n"
            "#     SwapProbability           ...\n"
            "#     WidomProbability          ...\n"
            "#     CreateNumberOfMolecules   ..."
        )

    return RASPA_GENERIC_TEMPLATE.format(
        forcefield=params.get("forcefield", "UFF"),
        charge_block=params.get("charge_block", "ChargeMethod                  None"),
        framework_name=params.get("framework_name", "MOF"),
        use_charges_from_cif=params.get("use_charges_from_cif", "no"),
        unitcell_x=params.get("unitcell_x", 1),
        unitcell_y=params.get("unitcell_y", 1),
        unitcell_z=params.get("unitcell_z", 1),
        temperature=params.get("temperature", 298.0),
        component_block=component_block,
    )


def create_raspa_input_prompt(
    query: dict,
    filled_template: str,
    params: dict,
    method_paragraph: Optional[str] = None,
    rag_hints: str = "",
    manual_hints: str = "",
) -> str:
    pressure_bar = params.get("pressure_bar")
    pressure_note = (
        f"ExternalPressure = {pressure_bar * 1e5:.0f}  # Pa  ({pressure_bar} bar)"
        if pressure_bar is not None else ""
    )

    prompt = f"""
You are a RASPA simulation.input file generation expert for MOF simulations.
Generate a complete RASPA simulation.input based on the generic template and simulation request below.

Generic RASPA template:
{filled_template}

Rules:
- Output ONLY the simulation.input content (no markdown fences, no explanation).
- Follow the provided generic template as closely as possible.
- Fill in the commented-out settings (SimulationType, NumberOfCycles, NumberOfInitializationCycles, ExternalPressure, move probabilities) with values appropriate for the requested calculation.
- Do NOT duplicate keywords.
- Use conservative, standard defaults unless the request explicitly requires otherwise.
- For GCMC / adsorption: set NumberOfCycles >= 10000, NumberOfInitializationCycles >= 2000, include ExternalPressure, set SwapProbability > 0.
- For Henry coefficient: set WidomProbability 1.0, all move probabilities 0.0, CreateNumberOfMolecules 0, omit ExternalPressure.
- For selectivity: include two Component blocks with appropriate MolFraction values summing to 1.0.
- Keep ChargeMethod and UseChargesFromCIFFile exactly as provided in the template (already computed).
"""

    if pressure_note:
        prompt += f"\nNote: {pressure_note}\n"

    if rag_hints and rag_hints.strip():
        prompt += f"""
LITERATURE_RAG_HINTS (optional; may be irrelevant. Use ONLY if clearly applicable):
{rag_hints.strip()}
"""

    if manual_hints and manual_hints.strip():
        prompt += f"""
RASPA_MANUAL_HINTS (official evidence; prefer exact RASPA keyword names when applicable):
{manual_hints.strip()}
"""

    if method_paragraph:
        prompt += f"""
Method paragraph (use ONLY explicit parameters from it):
{method_paragraph}
"""

    prompt += f"""
Simulation request:
{json.dumps(query, indent=2)}
"""
    return prompt


def get_raspa_system_message() -> str:
    return (
        "You are an expert in RASPA molecular simulation software for MOF gas adsorption. "
        "Generate valid RASPA simulation.input files following exact RASPA keyword syntax."
    )
