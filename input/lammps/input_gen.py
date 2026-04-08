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
from typing import Union

from config import OPENAI_MODEL_LAMMPS, get_openai_client

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
def llm_option_from_query(query: str) -> str:
    prompt = f"""You are configuring 'lammps-interface' for a MOF simulation.

Return ONLY ONE line with the exact option string (no code block, no extra words):
- Must start with: -ff <FORCE_FIELD>
- Allowed FORCE_FIELD: {', '.join(ALLOWED_FF)}
- Optional flags (space-separated, any order, zero or more):
  --fix-metal
  --h-bonding
  --dreid-bond-type morse
- Do NOT include CIF filename or any other flags.
- Output must be a single line, nothing else.

IMPORTANT force-field selection rules (do not print):
1) Dubbeldam force field is ONLY intended for IRMOF / MOF-5 family where the SBU/linker pattern matches.
   - Use '-ff Dubbeldam' ONLY if the user explicitly mentions IRMOF, IRMOF-1, MOF-5 (MOF5), or clearly indicates IRMOF family.
   - If the MOF is UiO-66 (or UiO series), HKUST-1, MIL-*, ZIF-*, NU-*, PCN-*, NOT IRMOF: do NOT choose Dubbeldam.

2) For thermal expansion / CTE / NPT temperature scan of general MOFs (e.g., UiO-66), default to:
   - '-ff UFF4MOF'
   Rationale: broad MOF coverage and avoids SBU-detection requirement.

3) Generic MOF adsorption/diffusion (when MOF family is not IRMOF):
   - Prefer '-ff UFF4MOF'

4) Hydrogen-bond networks on organics (if user explicitly mentions H-bonding):
   - Use '-ff Dreiding --h-bonding' (optionally add '--dreid-bond-type morse' if needed)

5) '--fix-metal' is only meaningful with UFF or Dreiding. Use it only if user requests metal constraints.

User query:
{query}
"""

    resp = get_openai_client().chat.completions.create(
        model=OPENAI_MODEL_LAMMPS,
        messages=[{"role": "user", "content": prompt}],
    )
    option = resp.choices[0].message.content.strip()

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



PROMPT_DIFFUSIVITY = """
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
- Initialize velocities for the guest group with:
  * `velocity guest create <temperature> <seed> mom yes rot yes dist gaussian`
- Define neighbor and neigh_modify settings.
  * Exclude MOF-MOF interactions with `neigh_modify exclude group MOF MOF`
- Define a default timestep, e.g. `timestep 1.0`.
- Define a dump for trajectory output using unwrapped coordinates:
  * e.g. `dump ... custom 1000 traj.lammpstrj id type xu yu zu`
- Set thermo output:
  * e.g. `thermo 1000`
  * e.g. `thermo_style custom step temp etotal press`

2) Preparation before dynamics
- Freeze the MOF atoms:
  * `fix freezeMOF MOF setforce 0.0 0.0 0.0`
- Reset image flags:
  * `set atom * image 0 0 0`
- Perform a short energy minimization:
  * `minimize 1.0e-4 1.0e-6 1000 10000`
- IMPORTANT: Do NOT define or modify `kspace_style` in this Run Section.
  Assume long-range electrostatics (if any) are already configured elsewhere.

3) Equilibration with Langevin (two stages)
- Apply `fix nve` to the guest group:
  * e.g. `fix nve_guest guest nve`
- Apply a Langevin thermostat to the guest group:
  * e.g. `fix langevin_guest guest langevin <Tstart> <Tstop> 100.0 <seed>`
- First equilibration run:
  * `run 5000`
- Second equilibration run with smaller timestep:
  * `timestep 0.5`
  * `run 5000`

4) Short NVE warm-up without thermostat
- Remove the Langevin thermostat:
  * `unfix langevin_guest`
- Restore the production timestep (e.g. `timestep 1.0`)
- Run a short NVE warm-up:
  * e.g. `run 10000`

5) MSD computation and long production run
- Define MSD for the guest group, using the center-of-mass option:
  * `compute msd_guest guest msd`
- Time-average the MSD and write to file:
  * e.g. `fix msd_out guest ave/time 1000 1 1000 c_msd_guest[4] file msd_guest.dat`
- Run a long NVE production to accumulate MSD data:
  * e.g. `run 1000000`

Do NOT include:
- NPT or NVT ensembles (no `fix npt`, `fix nvt`)
- SHAKE constraints
- `fix momentum` or similar momentum-removal fixes
- Any `kspace_style` commands (assume they are defined in other sections)

IMPORTANT:
If simulation_description contains "JOB_NAME=..._<TEMP>K" (e.g., _200K, _300K, _400K),
you MUST use that <TEMP> as the ONLY temperature for both `velocity guest create` and `fix langevin`.
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

PROMPT_GENERIC = """
    You are an expert in writing LAMMPS input scripts.

    Your task is to generate a `system.in` file for a LAMMPS simulation using the following input information.

    The system.in file should contain:
    - The **group definitions are already provided** and must be used as-is:
    {group_definitions}
    - Velocity creation
    - Time step definition
    - Dump output settings
    - Thermo output settings
    - Fix commands for freezing MOF atoms, shaking water molecules, and NPT integration
    - A `minimize` command
    - A `run` command

    Example format:
    group mof type 1:6
    group w subtract all mof

    velocity       all create 300  12345
    timestep       1.0
    dump my_dump all atom 1 dump.lammpstrj
    thermo 1
    # Constraint ##################################
    fix freeze     mof setforce 0.0 0.0 0.0
    minimize 1.0e-5 1.0e-7 1000 10000
    unfix freeze

    fix com        w momentum 100 linear 1 1 1
    fix rigid      w shake 1e-4 20 0 b 1 a 1
    fix   fxnpt all npt temp 300.0 300.0 100.0 iso 1.0 1.0 1000.0 drag 1.0
    run 1000

    Generate a valid LAMMPS system.in file based on this format.
    Return only the content of system.in (no explanation).

    Simulation description:
    {simulation_description}

    --------------------------------
    Optional RAG notes (may be irrelevant):
    {rag_summaries}

    Rules for using RAG notes:
    - Use RAG only if it contains directly relevant LAMMPS run-section guidance.
    - Do NOT follow RAG if it conflicts with the required structure and "Do NOT include" rules.
    - Ignore experimental characterization content (XPS, FTIR, adsorption isotherms, etc.).
    --------------------------------
    """

PROMPT_THERMAL_EXPANSION = """
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

PROMPT_RDF_MOF_GUEST = """
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
- Reset image flags:
  set atom * image 0 0 0
- Energy minimization:
  min_style cg
  minimize 1.0e-6 1.0e-8 1000 10000
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




























PROMPT_INTERACTION_ENERGY_MOF_GUEST = """
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
- Reset image flags:
  set atom * image 0 0 0
- Energy minimization:
  min_style cg
  minimize 1.0e-6 1.0e-8 1000 10000
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

PROMPT_YOUNGS_MODULUS = """
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
set atom * image 0 0 0
min_style cg
minimize 1.0e-8 1.0e-10 5000 50000

3) Quasi-static uniaxial strain (x direction)
- Define variables:
  variable emax equal 0.005
  variable nstep equal 10
  variable de equal v_emax/v_nstep
  variable Lx0 equal lx

- Write header line once:
  print "i strain stress" file youngs_stress_strain.dat

- Loop:
  variable i loop ${{nstep}}
  label loop_strain

  variable scale equal 1.0+v_de
  change_box all x scale v_scale remap

  minimize 1.0e-8 1.0e-10 5000 50000

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


def generate_system_in(simulation_description: str,
                       group_definition: str,
                       property: str,
                       output_file: str = "system.in",
                       mode: str = "standard",
                       example_text: str = "",
                       rag_summaries: str = ""):

    if mode == "reproduce":
        prompt_template = PROMPT_REPRODUCE_RUNSECTION
        prompt = prompt_template.format(
            simulation_description=simulation_description,
            group_definitions=group_definition,
            example_text=example_text,
        )
    else:
        if property in ("diffusivity", "mean_squared_displacement", "msd", "self_diffusion_coefficient"):
            prompt_template = PROMPT_DIFFUSIVITY
        elif property in ("thermal_expansion", "thermal expansion", "thermal_expansion_coefficient"):
            prompt_template = PROMPT_THERMAL_EXPANSION
        elif property in ("rdf", "mof_guest_rdf", "radial_distribution_function", "gr"):
            prompt_template = PROMPT_RDF_MOF_GUEST
        elif property in ("interaction_energy", "mof_guest_interaction_energy", "ff_interaction_energy", "group_group_energy"):
            prompt_template = PROMPT_INTERACTION_ENERGY_MOF_GUEST
        elif property in ("youngs_modulus", "young_modulus", "young", "elastic_modulus", "elastic"):
            prompt_template = PROMPT_YOUNGS_MODULUS
        else:
            prompt_template = PROMPT_GENERIC

        prompt = prompt_template.format(
            simulation_description=simulation_description,
            group_definitions=group_definition,
            rag_summaries=rag_summaries,
        )
        
    response = get_openai_client().chat.completions.create(
        model=OPENAI_MODEL_LAMMPS,
        messages=[
            {"role": "system", "content": "You are an expert in LAMMPS simulation input generation."},
            {"role": "user", "content": prompt}
        ],
    )

    generated_code = response.choices[0].message.content.strip()

    with open(output_file, "a") as f:
        f.write("# ----------------- Run Section -----------------\n")
        f.write(generated_code + "\n")

    print(f"system.in generated at {output_file}")
