import sys
import subprocess
import os
import re

# === Utility functions for style guessing ===
def guess_pair_style(pair_coeffs):
    if pair_coeffs:
        return "lj/cut/coul/long 10.0"
    return None

def guess_bond_style(bond_coeffs):
    if bond_coeffs:
        return "harmonic"
    return None

def guess_angle_style(angle_coeffs):
    if angle_coeffs:
        return "harmonic"
    return None

def guess_dihedral_style(dihedral_coeffs):
    if dihedral_coeffs:
        return "opls"
    return None

def guess_improper_style(improper_coeffs):
    if improper_coeffs:
        return "harmonic"
    return None

def parse_coeff_section(lt_text):
    pattern = r'write_once\(["\']In Settings["\']\)\s*{([\s\S]*?)}'
    matches = re.findall(pattern, lt_text, re.MULTILINE)
    coeffs = {"pair": [], "bond": [], "angle": [], "dihedral": [], "improper": []}
    for section in matches:
        for line in section.split('\n'):
            line = line.strip()
            if line.startswith("pair_coeff"):
                coeffs["pair"].append(line.replace("pair_coeff","").strip())
            elif line.startswith("bond_coeff"):
                coeffs["bond"].append(line.replace("bond_coeff","").strip())
            elif line.startswith("angle_coeff"):
                coeffs["angle"].append(line.replace("angle_coeff","").strip())
            elif line.startswith("dihedral_coeff"):
                coeffs["dihedral"].append(line.replace("dihedral_coeff","").strip())
            elif line.startswith("improper_coeff"):
                coeffs["improper"].append(line.replace("improper_coeff","").strip())

    return coeffs


def generate_in_init_block(coeffs):
    lines = []
    lines.append("  atom_style full")
    if coeffs["pair"]:
        style = guess_pair_style(coeffs["pair"])
        if style:
            lines.append(f"  pair_style {style}")
            lines.append("  kspace_style pppm 1.0e-4")
    lines.append("  special_bonds lj/coul 0.0 0.0 0.5")
    if coeffs["bond"]:
        style = guess_bond_style(coeffs["bond"])
        if style:
            lines.append(f"  bond_style {style}")
    if coeffs["angle"]:
        style = guess_angle_style(coeffs["angle"])
        if style:
            lines.append(f"  angle_style {style}")
    if coeffs["dihedral"]:
        style = guess_dihedral_style(coeffs["dihedral"])
        if style:
            lines.append(f"  dihedral_style {style}")
    if coeffs["improper"]:
        style = guess_improper_style(coeffs["improper"])
        if style:
            lines.append(f"  improper_style {style}")
    if len(lines) > 1:
        result = ['write_once("In Init") {'] + lines + ['}']
        return "\n".join(result) + "\n"
    else:
        return ""

def insert_in_init(original_text, in_init_block):
    pattern = r'(write_once\(["\']In Settings["\']\)\s*{)'
    match = re.search(pattern, original_text)
    if match:
        idx = match.start()
        return original_text[:idx] + in_init_block + "\n" + original_text[idx:]
    else:
        return in_init_block + "\n" + original_text

# === MAIN SCRIPT ===
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python opls_aa.py molecule.xyz")
        sys.exit(1)
    xyzfile = sys.argv[1]
    name = os.path.splitext(os.path.basename(xyzfile))[0]

    # 1. Run LigParGen and remove all except the lmp file
    print(f"[*] Running LigParGen for {name}.xyz ...")
    try:
        subprocess.check_call(["ligpargen", "-i", xyzfile, "-n", name])
    except Exception as e:
        print("LigParGen failed:", e)
        sys.exit(2)

    # 2. Delete all files except {name}.lammps.lmp
    for f in os.listdir():
        if f.startswith(name) and not (
            f.endswith(".lammps.lmp") or f.endswith(".xyz")
        ):
            try:
                os.remove(f)
            except Exception:
                pass

    lmpfile = f"{name}.lammps.lmp"
    if not os.path.exists(lmpfile):
        print(f"Error: {lmpfile} not found.")
        sys.exit(2)

    # 3. Run ltemplify.py
    ltfile = f"{name}.lt"
    print(f"[*] Running ltemplify.py to generate {ltfile} ...")
    try:
        subprocess.check_call(f"ltemplify.py {lmpfile} > {ltfile}", shell=True)
    except Exception as e:
        print("ltemplify.py failed:", e)
        sys.exit(3)

    # 4. Parse .lt and inject In Init
    print(f"[*] Updating {ltfile} with write_once('In Init') ...")
    with open(ltfile) as f:
        lt_text = f.read()
    coeffs = parse_coeff_section(lt_text)
    in_init_block = generate_in_init_block(coeffs)
    if in_init_block:
        newtext = insert_in_init(lt_text, in_init_block)
    else:
        newtext = lt_text

    with open(ltfile, "w") as f:
        f.write(newtext)
    print(f"[Done] {ltfile} created with auto-detected write_once('In Init').")
