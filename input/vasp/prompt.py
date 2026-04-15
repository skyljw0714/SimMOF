import json
from typing import Optional

VASP_RELAX_INCAR_DEFAULTS = {
    "GGA": "PE",
    "ISMEAR": "0",
    "SIGMA": "0.05",
    "ENCUT": "400",
    "EDIFF": "1E-6",
    "EDIFFG": "-0.01",
    "IBRION": "2",
    "ISIF": "3",
    "NSW": "500",
    "LREAL": "Auto",
    "KSPACING": "0.3",
    "KGAMMA": ".TRUE.",
    "NCORE": "4",
    "LPLANE": ".TRUE.",
    "IALGO": "38",
    "ISPIN": "2",
}

VASP_DOS_INCAR_DEFAULTS = {
    "GGA": "PE",
    "ENCUT": "520",
    "EDIFF": "1E-6",
    "ISMEAR": "0",
    "SIGMA": "0.05",
    "ALGO": "Normal",
    "NELM": "200",
    "ISYM": "0",
    "IBRION": "-1",
    "NSW": "0",
    "LORBIT": "11",
    "NEDOS": "2000",
    "PREC": "Accurate",
    "ADDGRID": ".TRUE.",
    "LASPH": ".TRUE.",
    "LREAL": "Auto",
    "KGAMMA": ".TRUE.",
    "LWAVE": ".FALSE.",
    "LCHARG": ".FALSE.",
}

VASP_BANDGAP_INCAR_DEFAULTS = {
    "GGA": "PE",
    "ENCUT": "520",
    "EDIFF": "1E-6",
    "ALGO": "Normal",
    "NELM": "200",
    "ISYM": "0",
    "IBRION": "-1",
    "NSW": "0",
    "ISMEAR": "0",
    "SIGMA": "0.05",
    "PREC": "Accurate",
    "ADDGRID": ".TRUE.",
    "LASPH": ".TRUE.",
    "LREAL": "Auto",
    "KGAMMA": ".TRUE.",
    "LWAVE": ".FALSE.",
    "LCHARG": ".FALSE.",
}

VASP_FORMAT = f'''
System = {{system}}

GGA = {VASP_RELAX_INCAR_DEFAULTS["GGA"]}
ISMEAR = {VASP_RELAX_INCAR_DEFAULTS["ISMEAR"]}
SIGMA = {VASP_RELAX_INCAR_DEFAULTS["SIGMA"]}
ENCUT = {VASP_RELAX_INCAR_DEFAULTS["ENCUT"]}
EDIFF = {VASP_RELAX_INCAR_DEFAULTS["EDIFF"]}
EDIFFG = {VASP_RELAX_INCAR_DEFAULTS["EDIFFG"]}
IBRION = {VASP_RELAX_INCAR_DEFAULTS["IBRION"]}
ISIF = {{ISIF}}
NSW = {VASP_RELAX_INCAR_DEFAULTS["NSW"]}
LREAL = {VASP_RELAX_INCAR_DEFAULTS["LREAL"]}
KSPACING = {VASP_RELAX_INCAR_DEFAULTS["KSPACING"]}
KGAMMA = {VASP_RELAX_INCAR_DEFAULTS["KGAMMA"]} #Never change

IVDW = # for dispersion correction

NCORE    =   {VASP_RELAX_INCAR_DEFAULTS["NCORE"]}
LPLANE   =   {VASP_RELAX_INCAR_DEFAULTS["LPLANE"]}
IALGO    =   {VASP_RELAX_INCAR_DEFAULTS["IALGO"]}
ISPIN   =   {VASP_RELAX_INCAR_DEFAULTS["ISPIN"]}
'''

VASP_DOS_FORMAT = f'''
System = {{system}}

# ---------- Electronic ----------
GGA    = {VASP_DOS_INCAR_DEFAULTS["GGA"]}
ENCUT  = {VASP_DOS_INCAR_DEFAULTS["ENCUT"]}
EDIFF  = {VASP_DOS_INCAR_DEFAULTS["EDIFF"]}
ISMEAR = {VASP_DOS_INCAR_DEFAULTS["ISMEAR"]}
SIGMA  = {VASP_DOS_INCAR_DEFAULTS["SIGMA"]}
ALGO   = {VASP_DOS_INCAR_DEFAULTS["ALGO"]}
NELM   = {VASP_DOS_INCAR_DEFAULTS["NELM"]}
ISYM   = {VASP_DOS_INCAR_DEFAULTS["ISYM"]}

# ---------- Static (no ionic relaxation) ----------
IBRION = {VASP_DOS_INCAR_DEFAULTS["IBRION"]}
NSW    = {VASP_DOS_INCAR_DEFAULTS["NSW"]}

# ---------- DOS / projections ----------
LORBIT = {VASP_DOS_INCAR_DEFAULTS["LORBIT"]}
NEDOS  = {VASP_DOS_INCAR_DEFAULTS["NEDOS"]}
# For DOS after a converged run:
# - If CHGCAR exists: ICHARG = 11
# - Else:            ICHARG = 2
ICHARG = {{ICHARG}}

# ---------- Grids / stability ----------
PREC    = {VASP_DOS_INCAR_DEFAULTS["PREC"]}
ADDGRID = {VASP_DOS_INCAR_DEFAULTS["ADDGRID"]}
LASPH   = {VASP_DOS_INCAR_DEFAULTS["LASPH"]}
LREAL   = {VASP_DOS_INCAR_DEFAULTS["LREAL"]}
KGAMMA  = {VASP_DOS_INCAR_DEFAULTS["KGAMMA"]}

# ---------- I/O ----------
LWAVE  = {VASP_DOS_INCAR_DEFAULTS["LWAVE"]}
LCHARG = {VASP_DOS_INCAR_DEFAULTS["LCHARG"]}
'''

VASP_BG_FORMAT = f"""
System = {{system}}

# ---------- Electronic ----------
GGA    = {VASP_BANDGAP_INCAR_DEFAULTS["GGA"]}
ENCUT  = {VASP_BANDGAP_INCAR_DEFAULTS["ENCUT"]}
EDIFF  = {VASP_BANDGAP_INCAR_DEFAULTS["EDIFF"]}
ALGO   = {VASP_BANDGAP_INCAR_DEFAULTS["ALGO"]}
NELM   = {VASP_BANDGAP_INCAR_DEFAULTS["NELM"]}
ISYM   = {VASP_BANDGAP_INCAR_DEFAULTS["ISYM"]}

# ---------- Static (no ionic relaxation) ----------
IBRION = {VASP_BANDGAP_INCAR_DEFAULTS["IBRION"]}
NSW    = {VASP_BANDGAP_INCAR_DEFAULTS["NSW"]}

# ---------- Smearing (band gap / eigenvalues) ----------
# If you use tetrahedron method (ISMEAR=-5), need enough k-points (NKPT >= 4).
# For Gamma-only large MOFs, keep ISMEAR=0 + small SIGMA.
ISMEAR = {VASP_BANDGAP_INCAR_DEFAULTS["ISMEAR"]}
SIGMA  = {VASP_BANDGAP_INCAR_DEFAULTS["SIGMA"]}

# ---------- Grids / stability ----------
PREC    = {VASP_BANDGAP_INCAR_DEFAULTS["PREC"]}
ADDGRID = {VASP_BANDGAP_INCAR_DEFAULTS["ADDGRID"]}
LASPH   = {VASP_BANDGAP_INCAR_DEFAULTS["LASPH"]}
LREAL   = {VASP_BANDGAP_INCAR_DEFAULTS["LREAL"]}
KGAMMA  = {VASP_BANDGAP_INCAR_DEFAULTS["KGAMMA"]}

# ---------- I/O ----------
LWAVE  = {VASP_BANDGAP_INCAR_DEFAULTS["LWAVE"]}
LCHARG = {VASP_BANDGAP_INCAR_DEFAULTS["LCHARG"]}
"""

def get_relax_isif(query: dict) -> str:
    stage = (query.get("vasp_stage") or "").lower()
    role = (query.get("vasp_role") or "").lower()
    job_id = (query.get("job_id") or "").lower()

    if stage == "guest" or role == "guest" or job_id.endswith("_guest"):
        return "2"

    return "3"

def render_vasp_format(query: dict) -> str:
    vasp_format = select_vasp_format(query)

    if vasp_format == VASP_FORMAT:
        return vasp_format.format(
            system="{system}",
            ISIF=get_relax_isif(query),
        )

    if vasp_format == VASP_DOS_FORMAT:
        return vasp_format.format(
            system="{system}",
            ICHARG="{ICHARG}",
        )

    if vasp_format == VASP_BG_FORMAT:
        return vasp_format.format(
            system="{system}",
        )

    return vasp_format
    
def select_vasp_format(query: dict) -> str:
    stage = (query.get("vasp_stage") or "").lower()
    calc = (query.get("vasp_calc_type") or "").lower()
    prop = (query.get("property") or "").lower()

    if stage == "dos" or calc == "dos" or prop in ["dos", "density_of_states", "electronic_density_of_states"]:
        return VASP_DOS_FORMAT

    if stage in ["bandgap", "band_gap"] or calc in ["bandgap", "band_gap"] or prop in ["band_gap", "bandgap", "electronic_band_gap"]:
        return VASP_BG_FORMAT

    return VASP_FORMAT


def create_vasp_incar_prompt(
    query: dict,
    vasp_format: str,
    method_paragraph: Optional[str] = None,
    rag_hints: str = "",
):
    prompt = f"""
You are a VASP input file generation expert for MOF simulations.
Below is the standard VASP INCAR template. Follow its style and produce a clean INCAR.

{vasp_format}

Rules:
- Output ONLY the INCAR content (no markdown fences, no explanation).
- Follow the provided INCAR template as closely as possible.
- Preserve the template’s existing tags, values, ordering, and overall style unless the simulation request explicitly requires a change.
- Do NOT duplicate tags (each key must appear at most once).
- Do not change template defaults unnecessarily.
- Use conservative, general-purpose defaults unless the request explicitly requires otherwise.
- If the request is DOS (static): ensure IBRION=-1 and NSW=0, and include DOS-related tags.
"""

    if rag_hints and rag_hints.strip():
        prompt += f"""
RAG_HINTS (optional; may be irrelevant. Use ONLY if clearly applicable; do not overfit):
{rag_hints.strip()}
"""

    if method_paragraph:
        prompt += f"""
Method paragraph (use ONLY explicit parameters from it; do not invent new ones):
{method_paragraph}
"""

    prompt += f"""
Simulation request:
{json.dumps(query, indent=2)}
"""
    return prompt


def get_vasp_system_message() -> str:
    return "You are a VASP input file expert. Output only the VASP INCAR file content."
