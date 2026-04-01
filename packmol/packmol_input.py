import os
import re
import subprocess
import numpy as np
from pymatgen.core import Structure
from ase.io import read, write
from ase.geometry import cellpar_to_cell
from collections import defaultdict

def is_orthorhombic(structure, tol=1e-2):
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
    margin=0.0
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
    script_dir = os.path.dirname(__file__)
    plane_script = os.path.join(script_dir, "plane.py")

    with open(input_filename, "w") as f:
        f.write(f"tolerance {system['tolerance']}\n")
        f.write(f"filetype xyz\n")
        f.write(f"output {system['output']}\n")
        f.write(f"seed -1\n")
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
                    "python", plane_script,
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
