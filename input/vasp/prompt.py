import json
from typing import Optional

VASP_GENERIC_TEMPLATE = """System = {system}

# GGA      = ...
# ISMEAR   = ...
# SIGMA    = ...
# EDIFF    = ...
# LREAL    = ...
# KGAMMA   = ...

# ENCUT    = ...
# IBRION   = ...
# NSW      = ...
# EDIFFG   = ...
# ISIF     = ...
# KSPACING = ...
# NCORE    = ...
# LPLANE   = ...
# IALGO    = ...
# NELM     = ...
# ISYM     = ...
# ISPIN    = ...
# MAGMOM   = ...
# IVDW     = ...
"""


def render_vasp_format(query: dict) -> str:
    return VASP_GENERIC_TEMPLATE


def create_vasp_incar_prompt(
    query: dict,
    vasp_format: str,
    method_paragraph: Optional[str] = None,
    rag_hints: str = "",
    manual_hints: str = "",
):
    prompt = f"""
You are a VASP input file generation expert for MOF simulations.
Generate a complete VASP INCAR file based on the generic template and simulation request below.

Generic INCAR template:
{vasp_format}

Rules:
- Output ONLY the INCAR content (no markdown fences, no explanation).
- Do NOT duplicate tags (each key must appear at most once).
- Follow the provided INCAR template as closely as possible.
- Fill in the commented-out tags with appropriate values for the simulation type described in the request.
- Add any additional tags required for this specific calculation type (e.g., ALGO, PREC, LORBIT for DOS; ISIF, EDIFFG for relaxation).
- Use conservative, general-purpose defaults unless the request explicitly requires otherwise.
- Always specify GGA explicitly; do not leave the exchange-correlation functional undefined.
- Structurally relax all system components when the calculation involves ionic degrees of freedom (e.g., vasp_stage indicates mof_opt, guest, or complex optimization).
- ISYM=0 is required when ionic positions may break the initial crystal symmetry during structural optimization.
- Periodic framework systems (MOF, complex) require optimization of both atomic positions and lattice parameters to reach the true energy minimum. Isolated molecules in a vacuum box should be relaxed with fixed cell dimensions to avoid unphysical cell deformation.
- Van der Waals interactions are important in MOF–guest systems; choose the dispersion treatment based on the full property calculation and apply it consistently to all contributing MOF, guest, and complex energies, including isolated references.
- ISTART controls whether previously converged wavefunctions are reused; ISTART=0 initializes the wavefunctions from scratch.
- NELM sets the maximum number of electronic minimization steps. When wavefunctions are initialized from scratch or require further optimization, allow enough iterations to reach the EDIFF convergence criterion. Single-step electronic runs are appropriate only for compatible post-processing workflows that reuse already-converged wavefunctions.
- Reusing a charge density does not by itself guarantee that compatible converged wavefunctions are available. Allow sufficient electronic minimization unless the request explicitly confirms that a compatible WAVECAR will be reused without further orbital optimization.
- Choose the electronic-occupancy method based on the system's metallic character, the calculation purpose, and the actual k-point mesh. Brillouin-zone interpolation methods require a sufficiently dense regular mesh that supports the interpolation; for sparse sampling, use an occupation-broadening method that remains valid for individual k-points.
- Infer the k-point sampling from the cell dimensions and KSPACING; KGAMMA controls mesh centering and does not by itself imply Gamma-only sampling.
- Include non-spherical contributions to the gradient corrections when accurate total energies or electronic structures are required for systems containing localized d or f electrons.
- Apply the supplied `qmof_high_spin_initialization` whenever its `applicable` field is true: set ISPIN=2 and copy its MAGMOM string exactly. This is a QMOF-style high-spin initial guess in POSCAR atom order, with MAGMOM=5.0 on every QMOF d-block candidate atom (Sc-Cu, Y-Ag, Lu-Au, Lr-Rg; group 12 excluded), MAGMOM=7.0 on every QMOF f-block candidate atom (La-Yb and Ac-No), and MAGMOM=0.0 on all other atoms.
- `qmof_high_spin_initialization` is an initial magnetic guess, not a claim about the converged oxidation state, spin state, or magnetic ordering. Explicit user-supplied ISPIN/MAGMOM values or explicit low-spin, antiferromagnetic, non-collinear, spin-orbit, or nonmagnetic instructions take precedence.
- If `qmof_high_spin_initialization.applicable` is false, do not add spin polarization solely because this default policy exists.
- Always preserve explicit user constraints. If the simulation request contains a numeric or physical condition, explicitly map it to the corresponding INCAR tag when applicable.
- Do not omit explicit user constraints just because RAG evidence focuses on different tags. RAG evidence can add missing domain-specific tags, but it must not override or distract from the user's requested values.
"""

    if rag_hints and rag_hints.strip():
        prompt += f"""
LITERATURE_RAG_HINTS (optional; may be irrelevant. Use ONLY if clearly applicable; do not overfit):
{rag_hints.strip()}
"""

    if manual_hints and manual_hints.strip():
        prompt += f"""
VASP_MANUAL_RAG_HINTS (official/manual evidence; prefer exact INCAR tag names when applicable):
{manual_hints.strip()}
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
