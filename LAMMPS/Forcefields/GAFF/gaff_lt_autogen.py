import os
import subprocess
import shutil
import re

def run(cmd, env=None):
    """Run a shell command and raise an error if it fails."""
    print(f"[CMD] {cmd}")
    ret = subprocess.run(cmd, shell=True, env=env)
    if ret.returncode != 0:
        raise RuntimeError(f"Failed: {cmd}")

def main(xyz_file):
    # === Step 0. Preparation ===
    name = os.path.splitext(os.path.basename(xyz_file))[0]
    workdir = f"{name}_GAFF_work"
    os.makedirs(workdir, exist_ok=True)
    xyz_path = os.path.abspath(xyz_file)
    os.chdir(workdir)
    shutil.copy(xyz_path, f"{name}.xyz")
    
    # 0.1 If input is xyz, convert it to mol2 using obabel first
    run(f"obabel -ixyz {name}.xyz -omol2 -O {name}.mol2")
    # 1. Now use antechamber with mol2 input
    run(f"antechamber -i {name}.mol2 -fi mol2 -o {name}.mol2 -fo mol2 -c bcc -s 2")

    # === Step 2. Generate frcmod with parmchk2 ===
    run(f"parmchk2 -i {name}.mol2 -f mol2 -o {name}.frcmod")
    
    # === Step 3. Generate Amber prmtop/inpcrd files using tleap ===
    leapin = f"""
source leaprc.gaff
mol = loadmol2 {name}.mol2
loadamberparams {name}.frcmod
saveamberparm mol {name}.prmtop {name}.inpcrd
quit
"""
    with open("leap.in", "w") as f:
        f.write(leapin)
    run("tleap -f leap.in")
    
    # === Step 4. Convert Amber to LAMMPS files with InterMol ===
    run(f"python -m intermol.convert --amb_in {name}.prmtop {name}.inpcrd --lammps --oname {name}_converted")
    lmp_input = f"{name}_converted.input"
    lmp_data = f"{name}_converted.lmp"
    if not (os.path.exists(lmp_input) and os.path.exists(lmp_data)):
        raise RuntimeError("LAMMPS input/data not generated.")
    
    # === Step 5. Convert LAMMPS data file to .lt format with ltemplify ===
    run(f"ltemplify.py {lmp_data} > {name}.lt")
    # Replace all integer type indices in coeffs with symbolic names
    
    # === Step 6. Add styles and coeffs from .input to .lt file (avoid duplicates) ===
    patch_lt_with_input(f"{name}_converted.input", f"{name}.lt", f"{name}_final.lt")
    symbolicize_lt_coeffs(f"{name}_final.lt")
    print(f"\n[Done] Final GAFF .lt file generated at: {workdir}/{name}_final.lt")

# --- Functions for symbolic coefficient replacement and patching .lt file ---

def parse_masses(lt_text):
    """Parse atom types from Data Masses section, map index to symbolic name."""
    atom_type_map = {}
    in_masses = False
    for line in lt_text.splitlines():
        if "Data Masses" in line:
            in_masses = True
            continue
        if in_masses:
            if line.strip().startswith("}"):
                break
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].replace("@atom:type", "").isdigit():
                idx = int(parts[0]) if parts[0].isdigit() else int(parts[0].replace("@atom:type", ""))
                atom_type_map[idx] = f"@atom:type{idx}"
    return atom_type_map

def patch_pair_coeffs(lt_text, atom_type_map):
    """Replace pair_coeff index numbers with symbolic atom type names."""
    def repl(match):
        i = int(match.group(1))
        j = int(match.group(2))
        rest = match.group(3)
        return f"pair_coeff {atom_type_map.get(i, i)} {atom_type_map.get(j, j)} {rest}"
    return re.sub(r"pair_coeff\s+(\d+)\s+(\d+)\s+(.*)", repl, lt_text)

def parse_bonds_angles(lt_text, kind):
    """Parse bond/angle/dihedral/improper types from corresponding Data section."""
    type_map = {}
    section = f"Data {kind.capitalize()}s"
    in_section = False
    for line in lt_text.splitlines():
        if section in line:
            in_section = True
            continue
        if in_section:
            if line.strip().startswith("}"):
                break
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].replace(f"@{kind}:type", "").isdigit():
                idx = int(parts[0]) if parts[0].isdigit() else int(parts[0].replace(f"@{kind}:type", ""))
                type_map[idx] = f"@{kind}:type{idx}"
    return type_map

def patch_coeffs(lt_text, kind, type_map):
    """Replace coeff index numbers with symbolic names for a given kind."""
    def repl(match):
        idx = int(match.group(1))
        rest = match.group(2)
        return f"{kind}_coeff {type_map.get(idx, idx)} {rest}"
    return re.sub(fr"{kind}_coeff\s+(\d+)\s+(.*)", repl, lt_text)

def symbolicize_lt_coeffs(lt_file):
    """Symbolically replace all coeff indices in .lt file with their proper names."""
    with open(lt_file, "r") as f:
        lt_text = f.read()
    atom_type_map = parse_masses(lt_text)
    lt_text = patch_pair_coeffs(lt_text, atom_type_map)
    for kind in ['bond', 'angle', 'dihedral', 'improper']:
        type_map = parse_bonds_angles(lt_text, kind)
        lt_text = patch_coeffs(lt_text, kind, type_map)
    with open(lt_file, "w") as f:
        f.write(lt_text)
    print(f"[Done] Symbolic coeff conversion completed for {lt_file}")

def patch_lt_with_input(input_file, lt_file, output_lt_file):
    """Patch .lt file with styles/coeffs from the .input file, avoiding duplicates."""
    init_keywords = [
        "atom_style", "bond_style", "angle_style", "dihedral_style", "improper_style",
        "pair_style", "kspace_style", "special_bonds"
    ]
    settings_keywords = [
        "pair_coeff", "bond_coeff", "angle_coeff", "dihedral_coeff", "improper_coeff", "pair_modify"
    ]
    init_lines = []
    settings_lines = []
    # Read input file and collect relevant lines
    with open(input_file, "r") as f:
        for line in f:
            stripped = line.strip()
            for key in init_keywords:
                if stripped.startswith(key):
                    init_lines.append(stripped)
            for key in settings_keywords:
                if stripped.startswith(key):
                    settings_lines.append(stripped)
    # Read the .lt file
    with open(lt_file, "r") as f:
        lt_lines = f.readlines()
    # Check which style lines already exist in the .lt Init section
    def get_existing_styles(lines, section_name):
        in_section = False
        styles = set()
        for line in lines:
            if section_name in line:
                in_section = True
                continue
            if in_section:
                if "}" in line:
                    break
                stripped = line.strip()
                if stripped:
                    key = stripped.split()[0]
                    styles.add(key)
        return styles
    existing_init_styles = get_existing_styles(lt_lines, 'write_once("In Init")')
    # Filter out duplicate style lines
    filtered_init_lines = []
    for line in init_lines:
        key = line.split()[0]
        if key not in existing_init_styles:
            filtered_init_lines.append(line)
    # Insert new style/coeff lines after the relevant lt sections
    def insert_after_section(lines, section_name, new_lines):
        out = []
        inserted = False
        for i, line in enumerate(lines):
            out.append(line)
            if section_name in line and not inserted:
                j = i+1
                while j < len(lines) and lines[j].strip() == "{":
                    out.append(lines[j])
                    j += 1
                out.extend([f"  {l}\n" for l in new_lines])
                inserted = True
        return out
    lt_lines = insert_after_section(lt_lines, 'write_once("In Init")', filtered_init_lines)
    lt_lines = insert_after_section(lt_lines, 'write_once("In Settings")', settings_lines)
    # Write the patched lt file
    with open(output_lt_file, "w") as f:
        f.writelines(lt_lines)
    print(f"[Done] {output_lt_file} created with no duplicate styles in Init section!")


# --- Script entry point ---
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python gaff_lt_autogen.py molecule.xyz")
        exit(1)
    main(sys.argv[1])
