import os
import re
import subprocess
import numpy as np
from pymatgen.core import Structure
from ase.io import read, write
from ase.geometry import cellpar_to_cell
from collections import defaultdict

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

def system_dict_from_cif(
    cif_file, 
    guest_list,
    tolerance=2.0,
    output_prefix="system",
    margin=2.0
):  
    mof_xyz = cif_file.replace('.cif', '.xyz')
    if not os.path.exists(mof_xyz):
        cif_to_xyz(cif_file, mof_xyz)

    structure = Structure.from_file(cif_file)
    lattice = structure.lattice
    a, b, c = lattice.a, lattice.b, lattice.c
    alpha, beta, gamma = lattice.angles

    molecules = [
        {"file": cif_file.replace(".cif", ".xyz"), "count": 1, "fixed": True, "fixed_coords": [0, 0, 0, 0, 0, 0]}
    ]
    for guest in guest_list:
        molecules.append({"file": guest["file"], "count": guest["count"]})

    if is_orthorhombic(structure):
        box = [0, 0, 0, lattice.a, lattice.b, lattice.c]
        system = {
            "cell_type": "orthorhombic",
            "box": [round(x, 6) for x in box],
            "molecules": molecules,
            "tolerance": tolerance,
            "output": f"{output_prefix}.xyz",
            "margin": margin
        }
    else:
        cellpar = [round(x, 6) for x in [a, b, c, alpha, beta, gamma]]
        system = {
            "cell_type": "triclinic",
            "cellpar": cellpar,
            "molecules": molecules,
            "tolerance": tolerance,
            "output": f"{output_prefix}.xyz",
            "margin": margin
        }
    return system

def write_packmol_input(system, input_filename="system.inp"):
    with open(input_filename, "w") as f:
        f.write(f"tolerance {system['tolerance']}\n")
        f.write(f"filetype xyz\n")
        f.write(f"output {system['output']}\n")
        for mol in system["molecules"]:
            f.write(f"\nstructure {mol['file']}\n")
            f.write(f"  number {mol['count']}\n")
            if mol.get("fixed"):
                coords = " ".join(str(x) for x in mol["fixed_coords"])
                f.write(f"  fixed {coords}\n")
            elif system["cell_type"] == "orthorhombic":
                box = " ".join(str(x) for x in system["box"])
                f.write(f"  inside box {box}\n")
            elif system["cell_type"] == "triclinic":
                cellpar = system["cellpar"]
                a, b, c, alpha, beta, gamma = map(str, cellpar)

                args = [
                    "python", "plane.py",
                    "-m", str(system.get("margin", 0.0)),
                    "-t", "cellpar",
                    a, b, c, alpha, beta, gamma
                ]

                result = subprocess.run(args, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"[ERROR] plane.py failed:\n{result.stderr}")

                for line in result.stdout.strip().splitlines():
                    if line.startswith("over") or line.startswith("below"):
                        f.write(f"  {line.strip()}\n")

            f.write("end structure\n")

def xyz_to_cif_from_system_dict(system_dict):
    xyz_file = system_dict["output"]
    cif_file = os.path.splitext(xyz_file)[0] + ".cif"
    if not os.path.exists(xyz_file):
        raise FileNotFoundError(f"{xyz_file} not found!")
    atoms = read(xyz_file)
    if system_dict["cell_type"] == "orthorhombic":
        xlo, ylo, zlo, xhi, yhi, zhi = system_dict["box"]
        cell = [xhi-xlo, yhi-ylo, zhi-zlo]
        atoms.set_cell(cell)
        atoms.set_pbc([True, True, True])
    elif system_dict["cell_type"] == "triclinic":
        cellpar = system_dict["cellpar"]
        cell = cellpar_to_cell(cellpar)
        atoms.set_cell(cell)
        atoms.set_pbc([True, True, True])

    write(cif_file, atoms)
    return cif_file


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

def write_system_lt(
    cif_path, mof_lt_name, guest_lt_name, guest_count,
    output_file="system.lt", boundary_type="p p p"
):
    box = get_lammps_box_params_from_cif(cif_path)

    with open(output_file, "w") as f:
        f.write(f'import "{mof_lt_name}.lt"\n')
        f.write(f'import "{guest_lt_name}.lt"\n\n')

        f.write('write_once("Data Boundary") {\n')
        f.write(f'{box["xlo"]:.4f} {box["xhi"]:.4f} xlo xhi\n')
        f.write(f'{box["ylo"]:.4f} {box["yhi"]:.4f} ylo yhi\n')
        f.write(f'{box["zlo"]:.4f} {box["zhi"]:.4f} zlo zhi\n')
        f.write(f'{box["xy"]:.4f} {box["xz"]:.4f} {box["yz"]:.4f} xy xz yz\n')
        f.write('}\n\n')

        f.write('write_once("In Init") {\n')
        f.write(f'  boundary {boundary_type}\n')
        f.write('}\n\n')

        f.write(f'mof = new mof[1]\n')
        f.write(f'guest = new guest[{guest_count}]\n')

    print(f"✅ system.lt written to {output_file}")

def deduplicate_system_in_init(input_file="system.in.init", output_file="system.in.init.cleaned"):
    from collections import defaultdict

    hybrid_keywords = ["angle_style", "bond_style", "dihedral_style", "improper_style"]
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
            deduped = []
            for val in values:
                if val not in deduped:
                    deduped.append(val)

            if keyword == "pair_style":
                combined = deduped[-1]

            elif keyword in hybrid_keywords:
                if len(deduped) > 1 or any(val.startswith("hybrid") for val in deduped):
                    style_set = []
                    for val in deduped:
                        styles = val.split()
                        if styles[0] == "hybrid":
                            styles = styles[1:]
                        style_set.extend(styles)
                    unique_styles = list(dict.fromkeys(style_set))
                    combined = "hybrid " + " ".join(unique_styles)
                else:
                    combined = deduped[0]
            else:
                combined = deduped[0] if len(deduped) == 1 else " ".join(deduped)
            
            f.write(f"{keyword} {combined}\n")

    print(f"✅ Cleaned file written to: {output_file}")


def clean_cif_with_ase(input_cif, output_cif):
    atoms = read(input_cif)
    write(output_cif, atoms, format='cif')
    print(f"✅ Cleaned CIF written to: {output_cif}")

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
    # style_name, style_line, coeff_line mapping
    style_info = {
        "angle_style": {"block": "angle_style", "coeff": "angle_coeff"},
        "bond_style": {"block": "bond_style", "coeff": "bond_coeff"},
        "dihedral_style": {"block": "dihedral_style", "coeff": "dihedral_coeff"},
        "improper_style": {"block": "improper_style", "coeff": "improper_coeff"},
    }

    # Result saving dictionary
    all_styles = {}
    with open(lt_path, "r") as f:
        lines = f.readlines()

    # 1. Extract each style type in "In Init" block
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

    # 2. Extract coeff in "In Settings" block
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

    # 3. Combine
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

    # with open(settings_path, "w") as f:
    #     f.writelines(updated_lines)
