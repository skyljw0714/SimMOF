import os
import subprocess
import shutil


def run(cmd, cwd=None):
    print(f"[CMD] {cmd}")
    ret = subprocess.run(cmd, shell=True, cwd=cwd)
    if ret.returncode != 0:
        raise RuntimeError(f"Failed: {cmd}")


def generate_lt(molecule: str, xyz_file: str, output_file: str, workdir: str = None):
    name = molecule
    xyz_path = os.path.abspath(xyz_file)
    out_path = os.path.abspath(output_file)

    if workdir is None:
        workdir = os.path.join(os.path.dirname(xyz_path), f"{name}_GAFF_work")
    os.makedirs(workdir, exist_ok=True)

    shutil.copy(xyz_path, os.path.join(workdir, f"{name}.xyz"))

    run(f"obabel -ixyz {name}.xyz -omol2 -O {name}_raw.mol2", cwd=workdir)

    run(
        f"antechamber -i {name}_raw.mol2 -fi mol2 "
        f"-o {name}.mol2 -fo mol2 -c bcc -s 2 -at gaff2",
        cwd=workdir,
    )

    run(f"parmchk2 -i {name}.mol2 -f mol2 -o {name}.frcmod -s gaff2", cwd=workdir)

    leapin = (
        f"source leaprc.gaff2\n"
        f"mol = loadmol2 {name}.mol2\n"
        f"loadamberparams {name}.frcmod\n"
        f"saveamberparm mol {name}.prmtop {name}.inpcrd\n"
        f"quit\n"
    )
    with open(os.path.join(workdir, "leap.in"), "w") as f:
        f.write(leapin)
    run("tleap -f leap.in", cwd=workdir)

    import parmed as pmd

    struct = pmd.load_file(
        os.path.join(workdir, f"{name}.prmtop"),
        os.path.join(workdir, f"{name}.inpcrd"),
    )
    lmp_data = os.path.join(workdir, f"{name}.lmp")
    struct.save(lmp_data, overwrite=True)

    raw_lt = os.path.join(workdir, f"{name}_raw.lt")
    run(f"ltemplify.py -name {name} {name}.lmp > {name}_raw.lt", cwd=workdir)

    _inject_init_block(raw_lt, out_path)
    print(f"[GAFF] .lt file written → {out_path}")
    return out_path


def _inject_init_block(src_lt: str, dst_lt: str):
    with open(src_lt) as f:
        text = f.read()

    init_block = (
        'write_once("In Init") {\n'
        "  atom_style full\n"
        "  pair_style lj/cut/coul/long 10.0\n"
        "  kspace_style pppm 1.0e-4\n"
        "  bond_style harmonic\n"
        "  angle_style harmonic\n"
        "  dihedral_style fourier\n"
        "  improper_style cvff\n"
        "  special_bonds amber\n"
        "}\n\n"
    )

    if 'write_once("In Init")' not in text:
        text = text.replace(
            'write_once("In Settings")',
            init_block + 'write_once("In Settings")',
            1,
        )

    with open(dst_lt, "w") as f:
        f.write(text)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python gaff_lt_autogen.py <molecule_name> <molecule.xyz> [output.lt]")
        sys.exit(1)
    mol = sys.argv[1]
    xyz = sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else f"{mol}_gaff.lt"
    generate_lt(mol, xyz, out)
