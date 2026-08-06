from __future__ import annotations

import re
from typing import Any, Dict, Set


PROPERTY_ALIASES: Dict[str, str] = {
    "accessible_pore_volume": "pore_volume",
    "adsorption_heat": "heat_of_adsorption",
    "adsorption_isotherm": "isotherm",
    "coefficient_of_thermal_expansion": "thermal_expansion",
    "cte": "thermal_expansion",
    "diffusion_coefficient": "diffusivity",
    "henry_constant": "henry_coefficient",
    "isosteric_heat": "heat_of_adsorption",
    "largest_cavity_diameter": "lcd",
    "mean_square_displacement": "mean_squared_displacement",
    "mean_squared_distance": "mean_squared_displacement",
    "msd": "mean_squared_displacement",
    "pore_limiting_diameter": "pld",
    "self_diffusion_coefficient": "diffusivity",
    "thermal_expansion_coefficient": "thermal_expansion",
    "volumetric_thermal_expansion_coefficient": "thermal_expansion",
}


PROPERTY_AGENT: Dict[str, str] = {
    "surface_area": "ZeoppAgent",
    "pore_volume": "ZeoppAgent",
    "lcd": "ZeoppAgent",
    "pld": "ZeoppAgent",
    "uptake": "RASPAAgent",
    "isotherm": "RASPAAgent",
    "henry_coefficient": "RASPAAgent",
    "heat_of_adsorption": "RASPAAgent",
    "selectivity": "RASPAAgent",
    "working_capacity": "RASPAAgent",
    "diffusivity": "LAMMPSAgent",
    "mean_squared_displacement": "LAMMPSAgent",
    "thermal_expansion": "LAMMPSAgent",
    "youngs_modulus": "LAMMPSAgent",
    "binding_energy": "VASPAgent",
    "adsorption_energy": "VASPAgent",
    "bader_charge": "VASPAgent",
    "projected_dos": "VASPAgent",
    "band_gap": "VASPAgent",
}


EXPLICIT_PROPERTY_PATTERNS: Dict[str, tuple[str, ...]] = {
    "surface_area": (r"\bsurface\s+area\b",),
    "pore_volume": (r"\bpore\s+volume\b",),
    "lcd": (r"\blcd\b", r"\blargest\s+cavity\s+diameter\b"),
    "pld": (r"\bpld\b", r"\bpore[\s-]+limiting\s+diameter\b"),
    "uptake": (r"\buptake\b",),
    "isotherm": (r"\bisotherms?\b",),
    "henry_coefficient": (
        r"\bhenry(?:'s)?\s+(?:coefficient|constant)\b",
        r"\badsorption\s+at\s+infinite\s+dilution\b",
    ),
    "heat_of_adsorption": (
        r"\bheat\s+of\s+adsorption\b",
        r"\bisosteric\s+heat\b",
    ),
    "selectivity": (r"\bselectivit(?:y|ies)\b",),
    "working_capacity": (r"\bworking\s+capacity\b",),
    "diffusivity": (
        r"\bdiffusivit(?:y|ies)\b",
        r"\bdiffusion\s+coefficients?\b",
        r"\bself[\s-]+diffusion\b",
    ),
    "mean_squared_displacement": (
        r"\bmean[\s-]+squared?\s+(?:displacement|distance)\b",
        r"\bmsd\b",
    ),
    "thermal_expansion": (
        r"\bthermal\s+expansion\b",
        r"\bcoefficient\s+of\s+thermal\s+expansion\b",
    ),
    "youngs_modulus": (
        r"\byoung(?:'s)?\s+modulus\b",
        r"\bstress[\s-]+strain\b",
    ),
    "binding_energy": (r"\bbinding\s+energ(?:y|ies)\b",),
    "adsorption_energy": (r"\badsorption\s+energ(?:y|ies)\b",),
    "bader_charge": (r"\bbader\s+charge\b", r"\bcharge\s+transfer\b"),
    "projected_dos": (
        r"\bprojected\s+(?:density\s+of\s+states|dos)\b",
        r"\bpdos\b",
    ),
    "band_gap": (r"\bband[\s-]+gaps?\b",),
}


def canonical_property(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    text = re.sub(r"_+", "_", text)
    return PROPERTY_ALIASES.get(text, text)


def explicit_properties(text: str) -> Set[str]:
    lowered = str(text or "").lower()
    found: Set[str] = set()
    for property_name, patterns in EXPLICIT_PROPERTY_PATTERNS.items():
        if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in patterns):
            found.add(property_name)
    return found
