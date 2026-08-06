import os
import re
import numpy as np
import math
import json

from pymatgen.core import Structure
from ase.io import read, write
from ase.geometry import cellpar_to_cell
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from config import LLM_DEFAULT
from core.llm_logging import log_llm_decision, set_llm_context
from input.interactive_review import maybe_interactive_review_input_file

def is_orthorhombic(structure, tol=1e-3):
    angles = structure.lattice.angles
    return all(abs(a - 90) < tol for a in angles)

def cif_to_xyz(cif_file, xyz_file=None):
    if xyz_file is None:
        xyz_file = cif_file.replace('.cif', '.xyz')
    atoms = read(cif_file)
    write(xyz_file, atoms)
    print(f"xyz file saved to: {xyz_file}")
    return xyz_file


def get_lammps_box_params_from_cif(cif_path):
    structure = Structure.from_file(cif_path)
    lattice = structure.lattice

    a, b, c = lattice.a, lattice.b, lattice.c
    alpha, beta, gamma = map(np.radians, [lattice.alpha, lattice.beta, lattice.gamma])

    xlo = 0.0
    xhi = a
    xy = b * np.cos(gamma)
    ylo = 0.0
    yhi = b * np.sin(gamma)
    xz = c * np.cos(beta)
    yz = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    zlo = 0.0
    zhi = np.sqrt(c**2 - xz**2 - yz**2)

    return {
        "xlo": xlo, "xhi": xhi,
        "ylo": ylo, "yhi": yhi,
        "zlo": zlo, "zhi": zhi,
        "xy": xy, "xz": xz, "yz": yz
    }


def compute_supercell_size(cif_file, cutoff=12.5):
    atoms = read(cif_file)
    A = atoms.cell.array  

    V = abs(np.linalg.det(A))
    h1 = V / np.linalg.norm(np.cross(A[1], A[2]))
    h2 = V / np.linalg.norm(np.cross(A[2], A[0]))
    h3 = V / np.linalg.norm(np.cross(A[0], A[1]))
    hmin = min(h1, h2, h3)

    required = 2 * cutoff
    nx = math.ceil(required / h1)
    ny = math.ceil(required / h2)
    nz = math.ceil(required / h3)

    print(f"Cell heights: h1={h1:.2f}, h2={h2:.2f}, h3={h3:.2f}")
    print(f"Required min height: {required:.2f}")
    print(f"Supercell needed: {nx} x {ny} x {nz}")

    return (nx, ny, nz)

def write_system_lt(
    cif_path,
    mof_lt_name,
    guest_lt_name=None,
    guest_count=0,
    output_file="system.lt",
    boundary_type="p p p",
):
    box = get_lammps_box_params_from_cif(cif_path)

    out = Path(output_file)
    if not out.is_absolute():
        out = Path(cif_path).parent / out
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        f.write(f'import "{mof_lt_name}.lt"\n')
        if guest_lt_name:
            f.write(f'import "{guest_lt_name}.lt"\n')
        f.write("\n")

        f.write('write_once("Data Boundary") {\n')
        f.write(f'{box["xlo"]:.4f} {box["xhi"]:.4f} xlo xhi\n')
        f.write(f'{box["ylo"]:.4f} {box["yhi"]:.4f} ylo yhi\n')
        f.write(f'{box["zlo"]:.4f} {box["zhi"]:.4f} zlo zhi\n')
        f.write(f'{box["xy"]:.4f} {box["xz"]:.4f} {box["yz"]:.4f} xy xz yz\n')
        f.write("}\n\n")

        f.write('write_once("In Init") {\n')
        f.write(f"  boundary {boundary_type}\n")
        f.write("}\n\n")

        f.write("mof = new mof[1]\n")
        if guest_lt_name and int(guest_count) > 0:
            f.write(f"guest = new guest[{int(guest_count)}]\n")

    print(f"system.lt written to {out}")
    return str(out)

def detect_charged_system(system_data_path: Union[str, Path], tol: float = 1e-12) -> bool:
    system_data_path = Path(system_data_path)
    if not system_data_path.exists():
        raise FileNotFoundError(f"system.data not found: {system_data_path}")

    lines = system_data_path.read_text().splitlines()

    
    atoms_start = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*Atoms\b", line):
            atoms_start = i
            break

    if atoms_start is None:
        raise RuntimeError(f"Cannot find 'Atoms' section in {system_data_path}")

    
    i = atoms_start + 1
    
    while i < len(lines) and (lines[i].strip() == "" or lines[i].lstrip().startswith("#")):
        i += 1

    
    
    section_title_re = re.compile(r"^\s*[A-Za-z][A-Za-z0-9_ ]*\s*$")

    started = False
    while i < len(lines):
        line = lines[i].strip()
        if line == "":
            if started:
                break
            i += 1
            continue

        
        
        if section_title_re.match(lines[i]) and not re.match(r"^\s*\d", lines[i]):
            if started:
                break

        if line.startswith("#"):
            i += 1
            continue

        
        
        parts = line.split()
        if len(parts) < 7:
            
            i += 1
            continue

        started = True
        try:
            q = float(parts[3])
            if abs(q) > tol:
                return True
        except ValueError:
            
            pass

        i += 1

    return False


def patch_pair_kspace_after_read_data(system_in_path, charged, cutoff=10.0, acc="1.0e-4"):
    p = Path(system_in_path)
    lines = p.read_text().splitlines()

    out = []
    for line in lines:
        out.append(line)
        if line.strip() in ['read_data "system.data"', "read_data system.data"]:
            if charged:
                out.append(f"pair_style lj/cut/coul/long {cutoff}")
                out.append(f"kspace_style pppm {acc}")
            else:
                out.append(f"pair_style lj/cut {cutoff}")

    p.write_text("\n".join(out) + "\n")


def deduplicate_system_in_init(input_file="system.in.init", output_file="system.in.init.cleaned"):
    from collections import defaultdict

    hybrid_keywords = ["angle_style", "bond_style", "dihedral_style", "improper_style"]
    drop_keys = {"pair_style", "kspace_style", "kspace_modify", "pair_modify"}

    grouped = defaultdict(list)

    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            keyword = parts[0]
            value = " ".join(parts[1:])
            grouped[keyword].append(value)

    with open(output_file, "w") as f:
        for keyword, values in grouped.items():
            if keyword in drop_keys:
                continue  

            deduped = []
            for val in values:
                if val not in deduped:
                    deduped.append(val)

            if keyword in hybrid_keywords:
                if len(deduped) > 1 or any(val.startswith("hybrid") for val in deduped):
                    style_set = []
                    for val in deduped:
                        styles = val.split()
                        if styles and styles[0] == "hybrid":
                            styles = styles[1:]
                        style_set.extend(styles)
                    unique_styles = list(dict.fromkeys(style_set))
                    combined = "hybrid " + " ".join(unique_styles)
                else:
                    combined = deduped[0]
            else:
                combined = deduped[0] if len(deduped) == 1 else " ".join(deduped)

            f.write(f"{keyword} {combined}\n")

    print(f"Cleaned file written to: {output_file}")


def clean_cif_with_ase(input_cif, output_cif):
    atoms = read(input_cif)
    write(output_cif, atoms, format='cif')
    print(f"Cleaned CIF written to: {output_cif}")

ALLOWED_FF = ("UFF", "UFF4MOF", "Dreiding", "BTW_FF", "Dubbeldam")
LAMMPS_FF_DESCRIPTIONS = """Implemented force-field descriptions:
- UFF: Universal Force Field, a broad-coverage generic force field for molecular and crystalline systems across much of the periodic table.
- UFF4MOF: MOF-oriented extension of UFF with atom typing and parameters adapted for metal-organic framework environments.
- Dreiding: generic force field based on simple atom typing and hybridization rules, commonly used as a transferable option for organic and framework atoms.
- BTW_FF: implemented lammps-interface option for MOF force-field assignment; use only when requested or supported by RAG evidence for the target system.
- Dubbeldam: lammps-interface option associated with published MOF-specific parameterizations developed to represent framework flexibility, adsorption, diffusion, or structural response. Do not infer applicability from a MOF name alone; use this option only when the user explicitly requests it or when RAG evidence identifies that the target belongs to a framework family with a transferable Dubbeldam-type parameterization for the intended task.
"""
def llm_option_from_query(query: str, rag_hints: str = "") -> str:
    prompt = f"""You are configuring 'lammps-interface' for a MOF simulation.

Return ONLY ONE line with the exact option string (no code block, no extra words):
- Must start with: -ff <FORCE_FIELD>
- Allowed FORCE_FIELD: {', '.join(ALLOWED_FF)}
- First use the user query and RAG hints. If these are not actionable, use the descriptions below as conservative internal-library guidance rather than literature evidence.
{LAMMPS_FF_DESCRIPTIONS}
- Optional flags (space-separated, any order, zero or more):
  --fix-metal        (constrain metal atoms during relaxation)
  --h-bonding        (enable explicit hydrogen-bond terms; only available with Dreiding)
  --dreid-bond-type morse  (use Morse potential for bonds instead of harmonic; only with Dreiding)
- Do NOT include CIF filename or any other flags.
- Output must be a single line, nothing else.

User query:
{query}

RAG hints from literature:
{rag_hints}
"""

    from langchain.schema import HumanMessage
    from config import ask_user_confirmation
    set_llm_context("LAMMPSInputAgent", "lammps_interface_option")
    resp = LLM_DEFAULT.invoke([HumanMessage(content=prompt)])
    option = resp.content.strip()
    print(f"[LAMMPSMofFF] LLM selected: {option}")

    def _reinvoke(instruction: str) -> str:
        revised_prompt = prompt + f"\n\nUser instruction: {instruction}\nRevise your selection accordingly."
        set_llm_context("LAMMPSInputAgent", "lammps_interface_option_revision")
        r = LLM_DEFAULT.invoke([HumanMessage(content=revised_prompt)])
        return r.content.strip()

    action, revised = ask_user_confirmation(
        "LAMMPSMofFF",
        f"Proposed MOF FF option: {option}",
        reinvoke_fn=_reinvoke,
        required=True,
    )
    if action == "apply" and revised != f"Proposed MOF FF option: {option}":
        print(f"[LAMMPSMofFF] Updated per user instruction: {revised}")
        option = revised

    try:
        log_llm_decision("LAMMPSInputAgent", "lammps_interface_option",
                         {"option": option, "query": query[:500]})
    except Exception:
        pass
    return option



def extract_hybrid_style_keys(init_path):
    hybrid_keywords = ["angle_style", "bond_style", "dihedral_style", "improper_style"]
    hybrid_keys = []

    with open(init_path, "r") as f:
        for line in f:
            stripped = line.strip()
            for key in hybrid_keywords:
                if stripped.startswith(key) and "hybrid" in stripped.split():
                    hybrid_keys.append(key)
    
    return hybrid_keys

def extract_styles_and_coeffs(lt_path, hybrid_keys):
    
    style_info = {
        "angle_style": {"block": "angle_style", "coeff": "angle_coeff"},
        "bond_style": {"block": "bond_style", "coeff": "bond_coeff"},
        "dihedral_style": {"block": "dihedral_style", "coeff": "dihedral_coeff"},
        "improper_style": {"block": "improper_style", "coeff": "improper_coeff"},
    }

    
    all_styles = {}
    with open(lt_path, "r") as f:
        lines = f.readlines()

    
    in_init = False
    style_lines = {}
    for line in lines:
        if 'write_once("In Init")' in line:
            in_init = True
        elif in_init and "}" in line:
            in_init = False
        if in_init:
            for key in hybrid_keys:
                block_key = style_info[key]["block"]
                if line.strip().startswith(block_key):
                    parts = line.split()[1:]
                    style_lines[key] = parts

    
    in_settings = False
    coeff_lines = defaultdict(list)
    for line in lines:
        if 'write_once("In Settings")' in line:
            in_settings = True
        elif in_settings and "}" in line:
            in_settings = False
        if in_settings:
            for key in hybrid_keys:
                coeff_key = style_info[key]["coeff"]
                if line.strip().startswith(coeff_key):
                    tokens = line.strip().split()
                    coeff_lines[key].append(" ".join(tokens[2:]))

    
    for key in hybrid_keys:
        all_styles[key] = {
            "style": style_lines.get(key, []),
            "coeff": coeff_lines.get(key, [])
        }
    return all_styles

def update_settings_with_style(lt_path, settings_path, hybrid_keys, output_path):
    style_dict = extract_styles_and_coeffs(lt_path, hybrid_keys)
    coeff_key_map = {
        "angle_style": "angle_coeff",
        "bond_style": "bond_coeff",
        "dihedral_style": "dihedral_coeff",
        "improper_style": "improper_coeff"
    }
    with open(settings_path, "r") as f:
        settings_lines = f.readlines()
    
    updated_lines = settings_lines[:]
    for style_key, val in style_dict.items():
        if "hybrid" in val["style"]:
            continue
        if len(val["style"]) != 1:
            continue
        style_name = val["style"][0]
        coeff_key = coeff_key_map[style_key]
        for coeff in val["coeff"]:
            coeff_data = coeff.split()
            for i, line in enumerate(settings_lines):
                tokens = line.strip().split()
                if tokens and tokens[0] == coeff_key:
                    if tokens[2:] == coeff_data:
                        tokens.insert(2, style_name)
                        updated_lines[i] = " ".join(tokens) + "\n"

    with open(output_path, "w") as f:
        f.writelines(updated_lines)



LEGACY_PROMPT_DIFFUSIVITY = """
You are an expert in writing LAMMPS input scripts.

Your task is to generate the **Run Section** of a `system.in` file
to calculate the **diffusivity** of guest molecules inside a MOF.

Use the following group definitions exactly as given:
{group_definitions}

Do not include any section headers or explanations.
Only output the commands inside the Run Section.

The Run Section must follow this structure:

1) Group and basic settings
- Re-declare the group definitions.
- Define neighbor and neigh_modify settings.
- Define a default timestep, e.g. `timestep 1.0`.
- Set thermo output:
  * e.g. `thermo 1000`
  * e.g. `thermo_style custom step time temp pe etotal`
  * include at least one energy column (`etotal` or `pe`) so energy autocorrelation can be post-processed.

2) Preparation before dynamics (ORDER IS CRITICAL)
- FIRST freeze the MOF atoms and zero its velocities — BEFORE minimize:
  * `fix freezeMOF MOF setforce 0.0 0.0 0.0`
  * `velocity MOF set 0.0 0.0 0.0`
- THEN perform energy minimization:
  * `min_style cg`
  * `minimize 1.0e-6 1.0e-8 1000 10000`
- Do NOT use `set group MOF image 0 0 0` — this corrupts image flags for bonds crossing periodic boundaries.
- IMPORTANT: Do NOT define or modify `kspace_style` in this Run Section.
  Assume long-range electrostatics (if any) are already configured elsewhere.

3) Equilibration with Langevin thermostat
- Initialize guest velocities. Consider whether removing net momentum/rotation from
  the initial velocity distribution is appropriate given the number of guest molecules.
- Choose a molecular constraint and thermostat scheme appropriate to the guest model:
  consider whether the guest has intramolecular degrees of freedom that need constraining,
  whether the chosen constraint algorithm is numerically stable for the molecule's
  equilibrium geometry, and whether thermostat coupling is compatible with the chosen
  constraint method.
- Equilibrate long enough for guest kinetic energy to reach target temperature.

4) Production run
- For diffusivity measurement, consider carefully which ensemble and thermostat scheme
  allow unbiased guest transport. Think about whether any coupling terms in the chosen
  dynamics introduce a systematic force that acts on molecular motion.
- Reset timestep before production.
- Define trajectory dump (REQUIRED for post-processing molecular COM MSD):
  * `dump traj guest custom 1000 traj.lammpstrj id mol type xu yu zu`
  * `dump_modify traj sort id`
- For in-LAMMPS molecular COM MSD, use chunk-based compute:
  * `compute guestChunk guest chunk/atom molecule nchunk once ids once compress yes`
  * `compute msd_guest guest msd/chunk guestChunk`
  * `fix avgmsd all ave/time 1000 1 1000 c_msd_guest[*] file msd_guest.dat mode vector`
- Run long production (multi-nanosecond), then clean up fixes/computes/dumps.

Do NOT include:
- NPT or NVT ensembles (no `fix npt`, `fix nvt`)
- `fix momentum` or similar momentum-removal fixes
- `set group MOF image 0 0 0` or `set atom * image 0 0 0`
- `compute msd ... com yes` — think about what this option removes from the measurement
- `compute msd/molecule` — does not exist in LAMMPS 3 Mar 2020; use `msd/chunk` instead
- Any `kspace_style` commands (assume they are defined in other sections)

CRITICAL DIFFUSIVITY RULES:
- The scientifically relevant diffusivity is based on the center of mass of each guest molecule.
- Before choosing any constraint algorithm for molecular bonds/angles, verify that the
  algorithm's iterative equations are well-defined for the molecule's equilibrium geometry.
- Before choosing a thermostat for the production run, consider whether that thermostat
  introduces any non-conservative forces that could bias molecular transport.
- MSD must be computed per molecule (center-of-mass), not per atom and not by subtracting
  the COM of the entire guest group. Use `compute msd/chunk` with `chunk/atom molecule`.
- `compute msd ... com yes` subtracts the drift of the entire group — reason carefully about
  whether this is appropriate when you want to measure that drift as diffusion.

IMPORTANT:
If simulation_description contains "JOB_NAME=..._<TEMP>K" (e.g., _200K, _300K, _400K),
you MUST use that <TEMP> as the ONLY temperature for both `velocity guest create` and Langevin.
Ignore any other temperatures mentioned elsewhere and do NOT create temperature loops.

Use the following simulation description to adapt the script:
{simulation_description}

--------------------------------
Optional RAG notes (may be irrelevant):
{rag_summaries}

Rules for using RAG notes:
- Use RAG only if it contains directly relevant LAMMPS run-section guidance.
- Do NOT follow RAG if it conflicts with the required structure and "Do NOT include" rules.
- Ignore experimental characterization content (XPS, FTIR, adsorption isotherms, etc.).
--------------------------------

Return only the Run Section code (no explanations, no markdown).
"""


LEGACY_PROMPT_THERMAL_EXPANSION = """
You are an expert in writing LAMMPS input scripts.

Your task is to generate the **Run Section only** (LAMMPS commands, no headers)
of a `system.in` file to simulate **thermal expansion** of a MOF framework
using NPT molecular dynamics and to output temperature-dependent averaged
cell properties.

STRICT OUTPUT RULES (MUST follow):
- Output ONLY valid LAMMPS commands.
- Do NOT include any section headers, comments, explanations, or prose.
- Do NOT define or modify any `group` commands in this Run Section.
- Apply all dynamics to `all` atoms.
- Do NOT define or modify any `kspace_style`.

----------------------------------------------------------------
CRITICAL LAMMPS SYNTAX RULES (NON-NEGOTIABLE):
----------------------------------------------------------------
1) You MUST define the following equal-style variables EXACTLY once:
   variable vVol equal vol
   variable vLx  equal lx
   variable vLy  equal ly
   variable vLz  equal lz

2) You MUST use ONLY these variables in fix ave/time:
   v_vVol v_vLx v_vLy v_vLz

3) NEVER write:
   fix ave/time ... vol lx ly lz ...
   (this is invalid and forbidden)

4) fix ave/time MUST generate MULTIPLE data lines per temperature:
   - Production run: 100000 steps
   - fix ave/time MUST be:
     fix favg all ave/time 100 100 10000 v_vVol v_vLx v_vLy v_vLz file thermal_avg_T${{T}}.dat
   - This guarantees at least 10 output lines per temperature.
   - Any other Nevery/Nrepeat/Nfreq choice is FORBIDDEN.

----------------------------------------------------------------
PHYSICAL / SIMULATION SETTINGS (IMPLEMENT AS COMMANDS):
----------------------------------------------------------------
- Temperatures (K): 200 250 300 350 400
- Pressure: 1.0 (project default units)
- Ensemble: NPT on all atoms
- Barostat: iso
- timestep: 1.0 fs
- Random seed: 12345

----------------------------------------------------------------
REQUIRED RUN LOGIC:
----------------------------------------------------------------
1) Initialization (before temperature loop):
   velocity all create 200.0 12345 mom yes rot yes dist gaussian
   neighbor 2.0 bin
   neigh_modify delay 0 every 1 check yes
   thermo_style custom step temp press vol lx ly lz etotal
   thermo 1000
   min_style cg
   minimize 1.0e-6 1.0e-8 1000 10000

2) Deterministic temperature loop:
   variable T index 200 250 300 350 400
   label loop_T

3) For each temperature ${{T}}:
   - Equilibration:
       fix npt_eq all npt temp ${{T}} ${{T}} 100.0 iso 1.0 1.0 1000.0
       run 50000
       unfix npt_eq

   - Production + averaging:
       reset_timestep 0
       fix npt_prod all npt temp ${{T}} ${{T}} 100.0 iso 1.0 1.0 1000.0
       fix favg all ave/time 100 100 10000 v_vVol v_vLx v_vLy v_vLz file thermal_avg_T${{T}}.dat
       run 100000

   - Extract averaged values:
       variable Vavg  equal f_favg[1]
       variable lxavg equal f_favg[2]
       variable lyavg equal f_favg[3]
       variable lzavg equal f_favg[4]

   - Append summary line:
       print "${{T}} ${{Vavg}} ${{lxavg}} ${{lyavg}} ${{lzavg}}" append thermal_expansion_summary.dat

   - Cleanup:
       unfix favg
       unfix npt_prod

   next T
   jump SELF loop_T

----------------------------------------------------------------
ABSOLUTELY DO NOT INCLUDE:
----------------------------------------------------------------
- Any group commands
- Guest-specific computes (MSD, diffusivity, etc.)
- SHAKE, rigid fixes, or freezing the framework
- Any alternative averaging strategy
- Any commentary or explanation

Use the following simulation description ONLY to adapt numeric parameters
if explicitly needed:
{simulation_description}

--------------------------------
Optional RAG notes (may be irrelevant):
{rag_summaries}

Rules for using RAG notes:
- Use RAG only if it contains directly relevant LAMMPS run-section guidance.
- Do NOT follow RAG if it conflicts with the required structure and "Do NOT include" rules.
- Ignore experimental characterization content (XPS, FTIR, adsorption isotherms, etc.).
--------------------------------

Return ONLY the LAMMPS Run Section commands.
"""

LEGACY_PROMPT_RDF_MOF_GUEST = """
You are an expert in writing LAMMPS input scripts.

Your task is to generate the **Run Section** of a `system.in` file
to compute the **MOF–guest RDF** directly in LAMMPS and write `rdf.dat`.

Use the following group definitions exactly as given:
{group_definitions}

Do not include any section headers or explanations.
Only output the commands inside the Run Section.

================================================================
REQUIREMENTS (MUST FOLLOW EXACTLY)
================================================================

1) Group and basic settings
- Re-declare the group definitions EXACTLY as given (copy-paste, no changes).
- Temperature rule:
  * Use 300 K unless simulation_description contains "JOB_NAME=..._<TEMP>K"
    (example: _200K, _300K, _400K). If present, use that TEMP as the ONLY temperature.
- Initialize velocities:
  velocity all create <T> 12345 mom yes rot yes dist gaussian
- Neighbor settings:
  neighbor 2.0 bin
  neigh_modify delay 0 every 1 check yes
- Timestep:
  timestep 1.0
- Thermo output:
  thermo 1000
  thermo_style custom step temp press etotal vol

2) Preparation before dynamics
- Energy minimization:
  min_style cg
  minimize 1.0e-6 1.0e-8 1000 10000
- Do NOT reset image flags (`set group MOF image 0 0 0` or `set atom * image 0 0 0`).
- IMPORTANT: Do NOT define or modify kspace_style or pair_style in this Run Section.

3) Equilibration with NVT (short)
- Apply NVT to all atoms:
  fix eq all nvt temp <T> <T> 100.0
- Run equilibration:
  run 50000
- Remove equilibration fix:
  unfix eq

4) RDF setup (MOF–guest)
- You MUST compute RDF using type-based pairs derived from the group definitions.
- Parse the atom type IDs from these two lines (they will appear in the group definitions):
  * group MOF type ...
  * group guest type ...
- Extract the integer type IDs from each line:
  * MOF types = [m1, m2, ..., mk]
  * guest types = [g1, g2, ..., gn]
- Construct the RDF pair list as a flat sequence of ALL guest×MOF combinations:
  g1 m1 g1 m2 ... g1 mk g2 m1 ... gn mk
- Define exactly ONE RDF compute with 200 bins:
  compute rdf_mg all rdf 200 <PAIR_LIST>
- Time-average ALL RDF outputs and write to rdf.dat:
  fix rdf_out all ave/time 1000 1 1000 c_rdf_mg[*] file rdf.dat mode vector
- IMPORTANT ordering:
  * compute rdf_mg must appear BEFORE fix rdf_out

5) Trajectory dump (for debugging/visualization; still required)
- Dump unwrapped coordinates including molecule id:
  dump d1 all custom 1000 traj_rdf.lammpstrj id mol type xu yu zu
  dump_modify d1 sort id

6) Production with NVT (long)
- Reset timestep counter:
  reset_timestep 0
- Apply production NVT:
  fix prod all nvt temp <T> <T> 100.0
- Long production run:
  run 200000
- Cleanup:
  unfix prod
  unfix rdf_out
  undump d1

================================================================
ABSOLUTELY DO NOT INCLUDE
================================================================
- Any kspace_style or pair_style commands
- Any additional group commands beyond re-declaring the provided group definitions
- NPT ensemble (no fix npt)
- SHAKE / rigid fixes
- Freezing the framework unless explicitly requested in simulation_description
- Post-processing steps (the goal is rdf.dat written by LAMMPS)

Simulation description:
{simulation_description}

--------------------------------
Optional RAG notes (may be irrelevant):
{rag_summaries}

Rules for using RAG notes:
- Use RAG only if it contains directly relevant LAMMPS run-section guidance.
- Do NOT follow RAG if it conflicts with the required structure and "Do NOT include" rules.
- Ignore experimental characterization content.
--------------------------------

Return only the Run Section code (no markdown).
"""

LEGACY_PROMPT_INTERACTION_ENERGY_MOF_GUEST = """
You are an expert in writing LAMMPS input scripts.

Your task is to generate the **Run Section** of a `system.in` file
to compute the **force-field interaction energy between MOF and guest**
(i.e., MOF–guest interaction energy) and write it to a data file.

Use the following group definitions exactly as given:
{group_definitions}

Do not include any section headers or explanations.
Only output the commands inside the Run Section.

================================================================
REQUIREMENTS (MUST FOLLOW EXACTLY)
================================================================

1) Group and basic settings
- Re-declare the group definitions EXACTLY as given (copy-paste, no changes).
- Temperature rule:
  * Use 300 K unless simulation_description contains "JOB_NAME=..._<TEMP>K"
    (example: _200K, _300K, _400K). If present, use that TEMP as the ONLY temperature.
- Initialize velocities:
  velocity all create <T> 12345 mom yes rot yes dist gaussian
- Neighbor settings:
  neighbor 2.0 bin
  neigh_modify delay 0 every 1 check yes
- Timestep:
  timestep 1.0
- Thermo output:
  thermo 1000
  thermo_style custom step temp press etotal vol

2) Preparation before dynamics
- Energy minimization:
  min_style cg
  minimize 1.0e-6 1.0e-8 1000 10000
- Do NOT reset image flags (`set group MOF image 0 0 0` or `set atom * image 0 0 0`).
- IMPORTANT: Do NOT define or modify kspace_style or pair_style in this Run Section.

3) Equilibration (simple NVT)
- Apply NVT to all atoms:
  fix eq all nvt temp <T> <T> 100.0
- Run equilibration:
  run 50000
- Remove equilibration fix:
  unfix eq

4) Interaction energy compute (MOF–guest)
- Define the interaction energy between MOF and guest using compute group/group:
  compute eint all group/group MOF guest pair yes kspace yes
- Time-average the interaction energy and write to file:
  fix eint_out all ave/time 1000 1 1000 c_eint file interaction_energy.dat

5) Production (collect statistics)
- Reset timestep counter:
  reset_timestep 0
- Production NVT:
  fix prod all nvt temp <T> <T> 100.0
- Run production long enough to average:
  run 200000
- Cleanup:
  unfix prod
  unfix eint_out

6) Optional trajectory dump (recommended for debugging; include molecule id)
- Dump unwrapped coordinates including molecule id:
  dump d1 all custom 1000 traj_intE.lammpstrj id mol type xu yu zu
  dump_modify d1 sort id
- After production:
  undump d1

================================================================
ABSOLUTELY DO NOT INCLUDE
================================================================
- Any kspace_style or pair_style commands
- Any additional group commands beyond re-declaring the provided group definitions
- NPT ensemble (no fix npt)
- SHAKE / rigid fixes
- Freezing the framework unless explicitly requested in simulation_description
- Post-processing steps (the goal is interaction_energy.dat written by LAMMPS)

Simulation description:
{simulation_description}

--------------------------------
Optional RAG notes (may be irrelevant):
{rag_summaries}

Rules for using RAG notes:
- Use RAG only if it contains directly relevant LAMMPS run-section guidance.
- Do NOT follow RAG if it conflicts with the required structure and "Do NOT include" rules.
--------------------------------

Return only the Run Section code (no markdown).
"""

LEGACY_PROMPT_YOUNGS_MODULUS = """
You are an expert in writing LAMMPS input scripts.

Your task is to generate the **Run Section only** (LAMMPS commands, no headers)
to compute the **Young's modulus of a MOF framework** (MOF-only).

STRICT OUTPUT:
- Output ONLY valid LAMMPS commands.
- Do NOT include any section headers, comments, or prose.
- Do NOT define or modify any kspace_style or pair_style commands.
- Do NOT define any group commands.

================================================================
REQUIREMENTS (MUST FOLLOW EXACTLY)
================================================================

1) Basic settings
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
thermo 1000
thermo_style custom step temp press pxx pyy pzz pe etotal vol lx ly lz

2) Preparation
min_style cg
fix initial_relax all box/relax aniso 0.0 vmax 0.001
minimize 1.0e-8 1.0e-10 5000 50000
unfix initial_relax

3) Quasi-static uniaxial strain (x direction)
- Define variables:
  variable emax equal 0.005
  variable nstep equal 10
  variable de equal v_emax/v_nstep
  variable Lx0 equal $(lx)

- Write header line once:
  print "i strain stress" file youngs_stress_strain.dat

- Loop:
  variable i loop ${{nstep}}
  label loop_strain

  variable scale equal 1.0+v_de
  change_box all x scale v_scale remap

  fix lateral_relax all box/relax y 0.0 z 0.0 vmax 0.001
  minimize 1.0e-8 1.0e-10 5000 50000
  unfix lateral_relax

  variable strain equal (lx - v_Lx0)/v_Lx0
  variable stress equal -pxx
  print "${{i}} ${{strain}} ${{stress}}" append youngs_stress_strain.dat

  next i
  jump SELF loop_strain

ABSOLUTELY DO NOT INCLUDE:
- fix nvt / fix npt
- any run command for MD dynamics
- any group commands
- any kspace_style / pair_style
- set atom * image 0 0 0 or set group * image 0 0 0

Simulation description:
{simulation_description}

Optional RAG notes:
{rag_summaries}

Return ONLY the LAMMPS Run Section commands.
"""


PROMPT_REPRODUCE_RUNSECTION = """
You are an expert in LAMMPS input scripts.

Task:
Generate ONLY the Run Section commands for a new system.in by REPRODUCING the user's provided example
as closely as possible, while injecting the correct group definitions for the NEW system.

You MUST:
1) Start by re-declaring the group definitions EXACTLY as given:
{group_definitions}

2) Then reproduce the user's example Run Section (or input snippet) as closely as possible.
- Keep fix/dump/thermo/run/compute structure and parameters as similar as possible.
- Do NOT add kspace_style or pair_style here.
- Do NOT add new physics or long explanations.

3) IMPORTANT (for reproducibility demonstration):
- Do NOT rename unknown group IDs from the example (e.g., solvent, water) proactively.
- If the example refers to groups that do not exist in the new group definitions, KEEP them as-is.
  (This may cause an error later and will be handled by an error-fixing agent.)

User provided example snippet:
<<<{example_text}>>>

Simulation description (context only):
{simulation_description}

Return ONLY the Run Section commands (no markdown, no headers, no explanation).
"""

PROMPT_UNIFIED_RUNSECTION = """
You are an expert in writing LAMMPS input scripts.

Task:
Generate ONLY the Run Section commands to append to an existing `system.in`.
Earlier sections already define atom styles, force-field coefficients, `read_data`,
`pair_style`, and `kspace_style` when needed.

================================================================
COMMON RULES (ALWAYS FOLLOW)
================================================================
- Output ONLY valid LAMMPS commands. No markdown, comments, headers, or prose.
- Do NOT define or modify `pair_style`, `pair_coeff`, `kspace_style`, or force-field coefficients.
- Re-declare the provided group definitions EXACTLY when they are non-empty.
- Do not invent new group names.
- Use 300 K unless simulation_description contains `JOB_NAME=..._<TEMP>K`
  such as `_200K`, `_300K`, or `_400K`; if present, use that temperature only.
- Use conservative defaults unless the user explicitly specifies numeric settings:
  neighbor 2.0 bin; neigh_modify delay 0 every 1 check yes; timestep 1.0; thermo 1000.
- Include `thermo_style custom` with fields relevant to the requested property.
- Do NOT use `set group MOF image 0 0 0` or `set atom * image 0 0 0` — resetting image flags corrupts bonds that cross periodic boundaries and causes incorrect rigid body geometry. Omit image flag resets entirely.
- Prefer `min_style cg` and a short `minimize` before production dynamics unless the task is pure reproduction.
- If trajectory output is useful, include molecule id and unwrapped coordinates:
  `dump ... custom ... id mol type xu yu zu`.
- Use RAG/official command evidence for exact command syntax.
- If RAG evidence conflicts with COMMON RULES, follow COMMON RULES.

================================================================
GROUP DEFINITIONS
================================================================
{group_definitions}

================================================================
SIMULATION DESCRIPTION
================================================================
{simulation_description}

================================================================
OFFICIAL LAMMPS COMMAND EVIDENCE
================================================================
{official_command_hints}

================================================================
OPTIONAL INTERNAL / LITERATURE RAG NOTES
================================================================
{rag_summaries}

Return ONLY the Run Section commands.
"""


def _format_lines(values):
    if not values:
        return "- none"
    return "\n".join(f"- {value}" for value in values)


def infer_lammps_run_task_profile(property: str):
    return {
        "intent": (
            f"Generate the requested LAMMPS Run Section for {property or 'the user objective'} "
            "using system capabilities and retrieved official evidence."
        ),
        "required_commands": [],
        "run_logic": [
            "Include only preparation required by the requested calculation.",
            "Define the requested observable before any command consumes it.",
            "Expose the observable through the requested output.",
            "Run or minimize only when required to produce that output.",
            "Clean up only IDs created by this Run Section.",
        ],
        "forbidden_commands": [
            "Do not redefine the simulation box, atoms, or force field.",
            "Do not use unavailable groups or per-atom fields.",
            "Do not add an ensemble, thermostat, minimization, or trajectory by default.",
        ],
        "expected_outputs": [
            "The property signal or file explicitly requested by the user.",
        ],
        "optional_postprocessing_analyses": [],
    }


def _legacy_infer_lammps_run_task_profile(property: str):
    prop = (property or "").strip().lower().replace("-", "_").replace(" ", "_")

    if (
        ("diffus" in prop)
        or (prop in {"msd", "mean_squared_displacement", "self_diffusion_coefficient"})
    ):
        return {
            "intent": "Compute guest diffusivity from molecular center-of-mass motion in a MOF.",
            "required_commands": [
                "velocity (no mom yes rot yes)",
                "neighbor",
                "neigh_modify",
                "timestep",
                "fix freezeMOF MOF setforce 0.0 0.0 0.0 (BEFORE minimize)",
                "velocity MOF set 0.0 0.0 0.0 (BEFORE minimize)",
                "minimize",
                "fix rigid/small molecule langevin (equilibration)",
                "unfix (remove Langevin fix before production)",
                "fix rigid/small molecule (NVE production, no thermostat)",
                "dump id mol type xu yu zu",
                "dump_modify",
                "compute chunk/atom molecule",
                "compute msd/chunk",
                "fix ave/time ... c_msd_guest[*] file msd_guest.dat mode vector",
                "run",
                "unfix",
                "uncompute",
                "undump",
            ],
            "run_logic": [
                "Use the provided MOF and guest groups.",
                "CRITICAL ORDER: framework freeze fix and velocity zeroing must be declared BEFORE minimize. Fixes declared after minimize have no effect on minimization.",
                "Do NOT use `set group MOF image 0 0 0` — image flags encode which periodic image each atom is in; resetting them without recomputing from bond topology corrupts bonded geometry.",
                "When initializing guest velocities, consider whether removing net momentum or angular momentum from the distribution is appropriate given the number of guest molecules present.",
                "Choose the molecular constraint method by reasoning about the guest model: does it have intramolecular DOF to constrain? Is the chosen constraint algorithm numerically stable for that molecule's equilibrium geometry? Is the algorithm compatible with the chosen integrator?",
                "Choose the production ensemble by reasoning about what forces act on guest molecules: does the chosen thermostat/barostat introduce any systematic coupling that could bias transport coefficients?",
                "Equilibration and production should use different setups if the equilibration thermostat introduces forces incompatible with unbiased diffusivity measurement.",
                "Dump `id mol type xu yu zu` so molecular COM MSD can be post-processed.",
                "For in-LAMMPS MSD: use `compute chunk/atom molecule` + `compute msd/chunk` to compute per-molecule COM MSD. Use `fix ave/time 1000 1 1000 c_msd_guest[*] file msd_guest.dat mode vector`.",
                "Include a thermo energy column such as `etotal` or `pe` so energy autocorrelation can be post-processed.",
                "Run a long production, e.g. `run 5000000`.",
            ],
            "forbidden_commands": [
                "Do not use `compute msd ... com yes` — reason carefully: this option subtracts the drift of the entire guest group, which is the quantity you are trying to measure.",
                "Do not use `compute msd/molecule` — does not exist in LAMMPS 3 Mar 2020; use `msd/chunk` instead.",
                "Do not use `set group MOF image 0 0 0` or `set atom * image 0 0 0`.",
                "Do not use `fix momentum` or global center-of-mass removal for diffusivity.",
                "Do not use `fix npt` or `fix nvt` for the production diffusivity run.",
                "Do not define `kspace_style` in the Run Section.",
            ],
            "expected_outputs": ["traj.lammpstrj", "log.lammps/thermo energy output", "msd_guest.dat"],
            "optional_postprocessing_analyses": [
                "anisotropic_diffusion",
                "van_hove_non_gaussian",
                "velocity_autocorrelation",
                "energy_autocorrelation",
                "rdf_guest_host_contact",
                "residence_hopping",
                "node_linker_contact",
                "pore_network_hopping_graph",
                "diffusion_activation_barrier for multi-temperature runs",
            ],
        }

    if ("thermal_expansion" in prop) or ("cte" in prop):
        return {
            "intent": "Compute temperature-dependent cell dimensions/volume for MOF thermal expansion.",
            "required_commands": [
                "variable",
                "velocity",
                "neighbor",
                "neigh_modify",
                "thermo",
                "thermo_style",
                "min_style",
                "minimize",
                "fix npt",
                "fix ave/time",
                "run",
                "print",
                "next",
                "jump",
                "unfix",
            ],
            "run_logic": [
                "Apply dynamics to `all` atoms; do not define groups.",
                "Define equal-style variables for vol/lx/ly/lz, then average `v_vVol v_vLx v_vLy v_vLz` with `fix ave/time`.",
                "Loop over temperatures 200 250 300 350 400 K unless the query provides a different scan.",
                "Use NPT with isotropic pressure coupling and write one averaged file per temperature.",
                "Append temperature and averaged cell properties to `thermal_expansion_summary.dat`.",
            ],
            "forbidden_commands": [
                "Do not write `fix ave/time ... vol lx ly lz`; use equal-style variables.",
                "Do not include guest-specific computes.",
                "Do not freeze the framework.",
                "Do not define `kspace_style` in the Run Section.",
            ],
            "expected_outputs": ["thermal_avg_T${T}.dat", "thermal_expansion_summary.dat"],
        }

    if ("rdf" in prop) or ("radial_distribution_function" in prop) or (prop == "gr"):
        return {
            "intent": "Compute MOF-guest radial distribution functions directly in LAMMPS.",
            "required_commands": [
                "velocity",
                "neighbor",
                "neigh_modify",
                "timestep",
                "thermo",
                "thermo_style",
                "min_style",
                "minimize",
                "fix nvt",
                "compute rdf",
                "fix ave/time",
                "dump",
                "dump_modify",
                "run",
                "unfix",
                "undump",
            ],
            "run_logic": [
                "Re-declare the provided group definitions exactly.",
                "Use type IDs from `group MOF type ...` and `group guest type ...` to build all guest x MOF RDF pairs.",
                "Define one RDF compute with 200 bins, e.g. `compute rdf_mg all rdf 200 <PAIR_LIST>`.",
                "Write all RDF outputs with `fix ave/time ... c_rdf_mg[*] file rdf.dat mode vector`.",
                "Use a short NVT equilibration and a longer NVT production.",
            ],
            "forbidden_commands": [
                "Do not define `pair_style` or `kspace_style` in the Run Section.",
                "Do not use NPT unless explicitly requested.",
                "Do not add post-processing instructions.",
            ],
            "expected_outputs": ["rdf.dat", "traj_rdf.lammpstrj"],
        }

    if (
        ("interaction_energy" in prop)
        or ("group_group_energy" in prop)
        or ("ff_interaction_energy" in prop)
        or ("binding_energy" in prop)
    ):
        return {
            "intent": "Compute force-field MOF-guest interaction energy.",
            "required_commands": [
                "velocity",
                "neighbor",
                "neigh_modify",
                "timestep",
                "thermo",
                "thermo_style",
                "min_style",
                "minimize",
                "fix nvt",
                "compute group/group",
                "fix ave/time",
                "dump",
                "dump_modify",
                "run",
                "unfix",
                "undump",
            ],
            "run_logic": [
                "Re-declare the provided group definitions exactly.",
                "Equilibrate with NVT, then define `compute eint all group/group MOF guest pair yes kspace yes`.",
                "Average `c_eint` to `interaction_energy.dat` with `fix ave/time`.",
                "Include a trajectory dump for debugging if useful.",
            ],
            "forbidden_commands": [
                "Do not define `pair_style` or `kspace_style` in the Run Section.",
                "Do not use NPT unless explicitly requested.",
                "Do not add post-processing instructions.",
            ],
            "expected_outputs": ["interaction_energy.dat", "traj_intE.lammpstrj"],
        }

    if ("young" in prop) or ("elastic" in prop) or ("modulus" in prop):
        return {
            "intent": "Compute Young's modulus of a MOF via quasi-static uniaxial strain and minimization.",
            "required_commands": [
                "neighbor",
                "neigh_modify",
                "thermo",
                "thermo_style",
                "min_style",
                "minimize",
                "fix box/relax",
                "unfix",
                "variable",
                "change_box",
                "print",
                "next",
                "jump",
            ],
            "run_logic": [
                "Do not run finite-temperature MD.",
                "Relax the initial cell at zero pressure with `fix box/relax aniso 0.0`, minimize, and unfix it before recording the reference length.",
                "Loop over small x-direction strain increments with `change_box all x scale ... remap`.",
                "Capture the initial x length once with immediate substitution: `variable Lx0 equal $(lx)`. Do not use `variable Lx0 equal lx`, which is re-evaluated after every box change.",
                "After each x strain increment, relax the lateral y/z dimensions at zero pressure with `fix box/relax y 0.0 z 0.0`, minimize, then unfix.",
                "Print strain and `-pxx` stress to `youngs_stress_strain.dat`.",
            ],
            "forbidden_commands": [
                "Do not use `fix nvt` or `fix npt`.",
                "Do not include `run` for MD dynamics.",
                "Do not define groups.",
                "Do not define `pair_style` or `kspace_style`.",
            ],
            "expected_outputs": ["youngs_stress_strain.dat"],
        }

    return {
        "intent": "Generate a conservative generic LAMMPS Run Section for the requested property.",
        "required_commands": [
            "velocity",
            "neighbor",
            "neigh_modify",
            "timestep",
            "thermo",
            "thermo_style",
            "min_style",
            "minimize",
            "fix nvt",
            "dump",
            "dump_modify",
            "run",
            "unfix",
            "undump",
        ],
        "run_logic": [
            "Re-declare provided group definitions exactly when present.",
            "Initialize velocities, minimize, equilibrate with NVT, write thermo and trajectory output, then run production.",
            "Use official command evidence to add property-specific computes or fixes only when directly relevant.",
        ],
        "forbidden_commands": [
            "Do not define `pair_style` or `kspace_style` in the Run Section.",
            "Do not invent force-field coefficients.",
        ],
        "expected_outputs": ["log/thermo output", "trajectory dump when useful"],
    }


def generate_system_in(simulation_description: str,
                       group_definition: str,
                       property: str,
                       output_file: str = "system.in",
                       mode: str = "standard",
                       example_text: str = "",
                       rag_summaries: str = "",
                       official_command_hints: str = "",
                       evidence_plan: dict = None,
                       evidence_candidates: list = None,
                       dependency_graph: dict = None,
                       intent_spec: dict = None,
                       evidence_provider: Optional[
                           Callable[[Dict[str, Any]], Dict[str, Any]]
                       ] = None,
                       context: dict = None):

    if mode == "reproduce":
        prompt_template = PROMPT_REPRODUCE_RUNSECTION
        prompt = prompt_template.format(
            simulation_description=simulation_description,
            group_definitions=group_definition,
            example_text=example_text,
        )
    else:
        from input.lammps.dependency_graph import (
            build_advisory_dependency_graph,
            validate_lammps_command_dependencies,
            validate_lammps_intent_coverage,
        )
        from input.lammps.intent import infer_lammps_intent
        from input.lammps.run_prompt import (
            build_advisory_revision_prompt,
            build_scientific_evidence_revision_prompt,
            build_minimal_runsection_prompt,
            extract_lammps_run_capabilities,
        )
        from input.lammps.scientific_reasoning import (
            infer_lammps_scientific_plan,
        )

        capabilities = extract_lammps_run_capabilities(
            output_file,
            group_definition,
        )
        if intent_spec is None:
            set_llm_context("LAMMPSInputAgent", "run_section_intent_planning")
            intent_spec = infer_lammps_intent(
                LLM_DEFAULT,
                simulation_description=simulation_description,
                property_name=property,
                capabilities=capabilities,
            )
        prompt = build_minimal_runsection_prompt(
            simulation_description=simulation_description,
            property_name=property,
            group_definitions=group_definition,
            official_command_hints="(none; baseline draft is evidence-free)",
            rag_summaries="",
            capabilities=capabilities,
            intent_spec=intent_spec,
        )
        if context is not None:
            results = context.setdefault("results", {})
            results["lammps_run_capabilities"] = capabilities
            results["lammps_intent_spec"] = intent_spec

    from langchain.schema import SystemMessage, HumanMessage
    set_llm_context(
        "LAMMPSInputAgent",
        "run_section_generation"
        if mode == "reproduce"
        else "run_section_baseline_generation",
    )
    response = LLM_DEFAULT.invoke([
        SystemMessage(content="You are an expert in LAMMPS simulation input generation."),
        HumanMessage(content=prompt),
    ])
    generated_code = response.content.strip()
    baseline_code = generated_code
    validation_errors = []
    repair_attempts = 0
    accepted_repairs = 0
    rejected_repairs = 0
    retrieval_error = ""
    scientific_plan = {}
    scientific_candidate = ""
    scientific_candidate_errors = []
    scientific_candidate_applied = False
    if mode != "reproduce":
        validation_errors = validate_lammps_intent_coverage(
            generated_code,
            intent_spec,
        )
        validation_errors.extend(
            validate_lammps_command_dependencies(
                generated_code,
                capabilities=capabilities,
            )
        )
        validation_errors = list(dict.fromkeys(validation_errors))
        baseline_validation_errors = list(validation_errors)

        if evidence_provider is not None:
            try:
                evidence = evidence_provider(intent_spec or {}) or {}
                official_command_hints = (
                    evidence.get("formatted_hints") or ""
                ).strip()
                evidence_plan = evidence.get("evidence_plan") or {}
                evidence_candidates = evidence.get("evidence_candidates") or []
                dependency_graph = evidence.get("dependency_graph") or {}
                rag_summaries = (
                    evidence.get("rag_summaries") or rag_summaries or ""
                ).strip()
            except Exception as exc:
                retrieval_error = str(exc)
                print(
                    "[RAG] official LAMMPS command hints disabled due to "
                    f"error: {exc}"
                )

        if not dependency_graph and evidence_plan:
            dependency_graph = build_advisory_dependency_graph(evidence_plan)
        if not evidence_candidates and dependency_graph:
            evidence_candidates = (
                dependency_graph.get("candidate_commands") or []
            )

        max_repair_attempts = max(
            0,
            int(os.getenv("SIMMOF_LAMMPS_EVIDENCE_REPAIR_ATTEMPTS", "2")),
        )
        has_advisory_evidence = bool(
            official_command_hints
            or rag_summaries
            or evidence_candidates
            or dependency_graph
        )
        scientific_rag_enabled = os.getenv(
            "SIMMOF_LAMMPS_SCIENTIFIC_RAG",
            "1",
        ).strip().lower() not in {"0", "false", "no", "off"}
        if (
            has_advisory_evidence
            and scientific_rag_enabled
            and repair_attempts < max_repair_attempts
        ):
            set_llm_context(
                "LAMMPSInputAgent",
                "run_section_scientific_planning",
            )
            scientific_plan = infer_lammps_scientific_plan(
                LLM_DEFAULT,
                simulation_description=simulation_description,
                property_name=property,
                intent_spec=intent_spec or {},
                capabilities=capabilities,
                baseline_script=generated_code,
                official_command_hints=official_command_hints,
                dependency_graph=dependency_graph or {},
            )
            scientific_prompt = build_scientific_evidence_revision_prompt(
                baseline_prompt=prompt,
                baseline_script=generated_code,
                scientific_plan=scientific_plan,
                official_command_hints=official_command_hints,
                dependency_graph=dependency_graph or {},
                baseline_validation_errors=validation_errors,
            )
            repair_attempts += 1
            set_llm_context(
                "LAMMPSInputAgent",
                "run_section_scientific_evidence_generation",
            )
            response = LLM_DEFAULT.invoke(
                [
                    SystemMessage(
                        content=(
                            "You generate an executable LAMMPS Run Section by "
                            "integrating a scientific calculation plan with "
                            "official command evidence."
                        )
                    ),
                    HumanMessage(content=scientific_prompt),
                ]
            )
            scientific_candidate = response.content.strip()
            scientific_candidate_errors = validate_lammps_intent_coverage(
                scientific_candidate,
                intent_spec,
            )
            scientific_candidate_errors.extend(
                validate_lammps_command_dependencies(
                    scientific_candidate,
                    capabilities=capabilities,
                )
            )
            scientific_candidate_errors = list(
                dict.fromkeys(scientific_candidate_errors)
            )
            if len(scientific_candidate_errors) <= len(validation_errors):
                generated_code = scientific_candidate
                validation_errors = scientific_candidate_errors
                accepted_repairs += 1
                scientific_candidate_applied = True
            else:
                rejected_repairs += 1

        while (
            validation_errors
            and has_advisory_evidence
            and repair_attempts < max_repair_attempts
        ):
            repair_attempts += 1
            repair_prompt = build_advisory_revision_prompt(
                baseline_prompt=prompt,
                baseline_script=generated_code,
                intent_spec=intent_spec or {},
                official_command_hints=official_command_hints,
                rag_summaries=rag_summaries,
                dependency_graph=dependency_graph or {},
                validation_errors=validation_errors,
            )
            set_llm_context(
                "LAMMPSInputAgent",
                "run_section_advisory_revision",
            )
            response = LLM_DEFAULT.invoke(
                [
                    SystemMessage(
                        content=(
                            "You revise a LAMMPS Run Section using optional "
                            "official evidence and typed dependency validation."
                        )
                    ),
                    HumanMessage(content=repair_prompt),
                ]
            )
            candidate_code = response.content.strip()
            candidate_errors = validate_lammps_intent_coverage(
                candidate_code,
                intent_spec,
            )
            candidate_errors.extend(
                validate_lammps_command_dependencies(
                    candidate_code,
                    capabilities=capabilities,
                )
            )
            candidate_errors = list(dict.fromkeys(candidate_errors))
            if len(candidate_errors) < len(validation_errors):
                generated_code = candidate_code
                validation_errors = candidate_errors
                accepted_repairs += 1
            else:
                rejected_repairs += 1
                break
    else:
        baseline_validation_errors = []

    if context is not None:
        results = context.setdefault("results", {})
        results["lammps_run_section_generator"] = (
            "reproduction_prompt"
            if mode == "reproduce"
            else (
                "intent_draft_then_scientific_evidence_candidate"
                if evidence_provider is not None
                else "intent_draft"
            )
        )
        results["lammps_baseline_run_section"] = baseline_code
        results["lammps_baseline_validation_errors"] = (
            baseline_validation_errors
        )
        results["lammps_evidence_repair_attempted"] = repair_attempts > 0
        results["lammps_evidence_repair_attempts"] = repair_attempts
        results["lammps_evidence_application_errors"] = validation_errors
        results["lammps_advisory_repairs_accepted"] = accepted_repairs
        results["lammps_advisory_repairs_rejected"] = rejected_repairs
        results["lammps_advisory_candidates"] = evidence_candidates or []
        results["lammps_advisory_dependency_graph"] = dependency_graph or {}
        results["lammps_advisory_rag_summaries"] = rag_summaries
        results["lammps_advisory_retrieval_error"] = retrieval_error
        results["lammps_scientific_plan"] = scientific_plan
        results["lammps_scientific_candidate"] = scientific_candidate
        results["lammps_scientific_candidate_errors"] = (
            scientific_candidate_errors
        )
        results["lammps_scientific_candidate_applied"] = (
            scientific_candidate_applied
        )

    with open(output_file, "a") as f:
        f.write("# ----------------- Run Section -----------------\n")
        if mode != "reproduce" and group_definition.strip():
            f.write(group_definition.strip() + "\n")
        f.write(generated_code + "\n")
    maybe_interactive_review_input_file(
        software="LAMMPS",
        path=output_file,
        context=context or {},
        llm=LLM_DEFAULT,
        label="LAMMPSInputAgent",
    )

    print(f"system.in generated at {output_file}")
