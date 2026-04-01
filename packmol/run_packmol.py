import subprocess
import os
import glob
import random
import math
from pathlib import Path

from ase.io import read

from .packmol_input import system_dict_from_cif, write_packmol_input, xyz_to_cif_from_system_dict
from config import PACKMOL_EXECUTABLE, PACKMOL_OUTPUT_DIR, PROJECT_ROOT

PACKMOL_DEFAULT_TOLERANCE = 2.0
PACKMOL_DEFAULT_MAX_ATTEMPTS = 10
PACKMOL_DEFAULT_TIMEOUT_SEC = 30
PACKMOL_MIN_MOF_GUEST_CUTOFF = 1.6
PACKMOL_MIN_GUEST_GUEST_CUTOFF = 2.0


def _read_xyz_atoms(xyz_path: str):
    lines = Path(xyz_path).read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"Invalid xyz file: {xyz_path}")

    n = int(lines[0].strip())
    atoms = []

    for i, line in enumerate(lines[2:2 + n], start=1):
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Bad xyz line in {xyz_path}: {line}")

        atoms.append({
            "id": i,
            "sym": parts[0],
            "x": float(parts[1]),
            "y": float(parts[2]),
            "z": float(parts[3]),
        })

    if len(atoms) != n:
        raise ValueError(
            f"XYZ atom count mismatch in {xyz_path}: header={n}, parsed={len(atoms)}"
        )

    return atoms


def _minimum_image(dx: float, boxlen: float) -> float:
    return dx - boxlen * round(dx / boxlen)


def _dist_pbc(a, b, box):
    xlo, xhi, ylo, yhi, zlo, zhi = box
    lx = xhi - xlo
    ly = yhi - ylo
    lz = zhi - zlo

    dx = _minimum_image(a["x"] - b["x"], lx)
    dy = _minimum_image(a["y"] - b["y"], ly)
    dz = _minimum_image(a["z"] - b["z"], lz)

    return math.sqrt(dx * dx + dy * dy + dz * dz)


def get_box_from_cif(cif_file: str):
    atoms = read(cif_file)
    a, b, c = map(float, atoms.cell.lengths())
    return (0.0, a, 0.0, b, 0.0, c)


def validate_packmol_xyz(
    packed_xyz: str,
    mof_atom_count: int,
    guest_template_xyz: str,
    number_of_guest: int,
    box,
    min_mof_guest_cutoff: float = 1.6,
    min_guest_guest_cutoff: float = 2.0,
):
    
    packed_atoms = _read_xyz_atoms(packed_xyz)
    guest_template_atoms = _read_xyz_atoms(guest_template_xyz)
    atoms_per_guest = len(guest_template_atoms)

    expected_total = mof_atom_count + number_of_guest * atoms_per_guest
    if len(packed_atoms) != expected_total:
        return {
            "ok": False,
            "reason": (
                f"Packed xyz atom count mismatch: got {len(packed_atoms)}, "
                f"expected {expected_total} = mof({mof_atom_count}) + "
                f"num_guest({number_of_guest})*atoms_per_guest({atoms_per_guest})"
            ),
        }

    mof_atoms = packed_atoms[:mof_atom_count]
    guest_atoms = packed_atoms[mof_atom_count:]

    if len(guest_atoms) % atoms_per_guest != 0:
        return {
            "ok": False,
            "reason": (
                f"Guest atom count {len(guest_atoms)} not divisible by atoms_per_guest {atoms_per_guest}"
            ),
        }

    guest_molecules = []
    for i in range(0, len(guest_atoms), atoms_per_guest):
        guest_molecules.append(guest_atoms[i:i + atoms_per_guest])

    min_mof_guest = 1.0e30
    min_mof_guest_pair = None

    for g in guest_atoms:
        for m in mof_atoms:
            d = _dist_pbc(g, m, box)
            if d < min_mof_guest:
                min_mof_guest = d
                min_mof_guest_pair = (g, m)

    min_guest_guest = 1.0e30
    min_guest_guest_pair = None

    if len(guest_molecules) >= 2:
        for i in range(len(guest_molecules)):
            for j in range(i + 1, len(guest_molecules)):
                for a in guest_molecules[i]:
                    for b in guest_molecules[j]:
                        d = _dist_pbc(a, b, box)
                        if d < min_guest_guest:
                            min_guest_guest = d
                            min_guest_guest_pair = (a, b, i, j)

    reasons = []
    ok = True

    if min_mof_guest < min_mof_guest_cutoff:
        ok = False
        g, m = min_mof_guest_pair
        reasons.append(
            f"min MOF-guest distance too small: {min_mof_guest:.4f} Å "
            f"(guest atom {g['id']} {g['sym']} - MOF atom {m['id']} {m['sym']})"
        )

    if len(guest_molecules) >= 2 and min_guest_guest < min_guest_guest_cutoff:
        ok = False
        a, b, imol, jmol = min_guest_guest_pair
        reasons.append(
            f"min guest-guest distance too small: {min_guest_guest:.4f} Å "
            f"(guest mol {imol+1} atom {a['id']} {a['sym']} - "
            f"guest mol {jmol+1} atom {b['id']} {b['sym']})"
        )

    result = {
        "ok": ok,
        "mof_atom_count": mof_atom_count,
        "atoms_per_guest": atoms_per_guest,
        "number_of_guest": number_of_guest,
        "min_mof_guest": min_mof_guest,
        "min_guest_guest": None if min_guest_guest == 1.0e30 else min_guest_guest,
        "reasons": reasons,
    }

    if min_mof_guest_pair is not None:
        g, m = min_mof_guest_pair
        result["min_mof_guest_pair"] = {
            "guest_atom_id": g["id"],
            "guest_atom_sym": g["sym"],
            "guest_x": g["x"],
            "guest_y": g["y"],
            "guest_z": g["z"],
            "mof_atom_id": m["id"],
            "mof_atom_sym": m["sym"],
            "mof_x": m["x"],
            "mof_y": m["y"],
            "mof_z": m["z"],
        }

    if min_guest_guest_pair is not None:
        a, b, imol, jmol = min_guest_guest_pair
        result["min_guest_guest_pair"] = {
            "guest_mol_i": imol + 1,
            "guest_mol_j": jmol + 1,
            "atom_i_id": a["id"],
            "atom_i_sym": a["sym"],
            "atom_j_id": b["id"],
            "atom_j_sym": b["sym"],
        }

    return result


def run_packmol_from_cif(
    cif_file: str,
    guest_xyz: str,
    number_of_guest: int,
    number_of_system: int = 1,
    tolerance: float = PACKMOL_DEFAULT_TOLERANCE,
    output_dir: str = str(PACKMOL_OUTPUT_DIR),
    packmol_exec: str = str(PACKMOL_EXECUTABLE),
    max_attempts: int = PACKMOL_DEFAULT_MAX_ATTEMPTS,
    timeout_sec: int = PACKMOL_DEFAULT_TIMEOUT_SEC,
    min_mof_guest_cutoff: float = PACKMOL_MIN_MOF_GUEST_CUTOFF,
    min_guest_guest_cutoff: float = PACKMOL_MIN_GUEST_GUEST_CUTOFF,
):
    
    os.makedirs(output_dir, exist_ok=True)

    cif_name = os.path.splitext(os.path.basename(cif_file))[0]
    guest_name = os.path.splitext(os.path.basename(guest_xyz))[0]

    output_subdir = os.path.join(output_dir, f"{cif_name}_{guest_name}")
    os.makedirs(output_subdir, exist_ok=True)

    mof_atom_count = len(read(cif_file))
    box = get_box_from_cif(cif_file)

    for i in range(number_of_system):
        output_prefix = os.path.join(output_subdir, f"{cif_name}_{guest_name}_{i+1}")
        output_inp = f"{output_prefix}.inp"
        output_xyz = f"{output_prefix}.xyz"

        guest_list = [{"file": guest_xyz, "count": number_of_guest}]

        success = False
        attempt = 0

        while not success and attempt < max_attempts:
            attempt += 1
            print(f"[{output_prefix}] Attempt {attempt}...")

            system_dict = system_dict_from_cif(
                cif_file,
                guest_list,
                tolerance=tolerance,
                output_prefix=output_prefix,
            )

            system_dict["seed"] = random.randint(1, 10**8)
            write_packmol_input(system_dict, output_inp)

            try:
                with open(output_inp, "r") as inp:
                    result = subprocess.run(
                        [packmol_exec],
                        stdin=inp,
                        capture_output=True,
                        text=True,
                        timeout=timeout_sec
                    )

                if result.returncode != 0:
                    print(f"[{output_prefix}] Failed (nonzero return code). Retrying...")
                    continue

                if not os.path.exists(output_xyz):
                    print(f"[{output_prefix}] XYZ not found after Packmol. Retrying...")
                    continue

                check = validate_packmol_xyz(
                    packed_xyz=output_xyz,
                    mof_atom_count=mof_atom_count,
                    guest_template_xyz=guest_xyz,
                    number_of_guest=number_of_guest,
                    box=box,
                    min_mof_guest_cutoff=min_mof_guest_cutoff,
                    min_guest_guest_cutoff=min_guest_guest_cutoff,
                )

                if not check["ok"]:
                    print(f"[{output_prefix}] Rejected by validator:")
                    if "reason" in check:
                        print("   -", check["reason"])
                    for r in check.get("reasons", []):
                        print("   -", r)
                    continue

                success = True
                cif_out = xyz_to_cif_from_system_dict(system_dict)
                print(f"[{output_prefix}] Accepted.")
                print(f"[{output_prefix}] min_mof_guest  = {check['min_mof_guest']:.4f} Å")
                if check["min_guest_guest"] is not None:
                    print(f"[{output_prefix}] min_guest_guest = {check['min_guest_guest']:.4f} Å")
                print(f"[{output_prefix}] CIF saved: {cif_out}")

            except subprocess.TimeoutExpired:
                print(f"[{output_prefix}] Timeout after {timeout_sec}s. Retrying with new seed...")

        if not success:
            raise RuntimeError(f"[{output_prefix}] Failed after {max_attempts} attempts.")


def run_packmol_batch(cif_dir, xyz_dir, number_of_guest, number_of_system, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    cif_files = glob.glob(os.path.join(cif_dir, "*.cif"))
    xyz_files = glob.glob(os.path.join(xyz_dir, "*.xyz"))

    for cif_file in cif_files:
        for xyz_file in xyz_files:
            print(f"\n=== Running Packmol for {os.path.basename(cif_file)} + {os.path.basename(xyz_file)} ===")

            run_packmol_from_cif(
                cif_file=cif_file,
                guest_xyz=xyz_file,
                number_of_guest=number_of_guest,
                number_of_system=number_of_system,
                output_dir=output_dir
            )


if __name__ == "__main__":
    run_packmol_from_cif(
        cif_file=str(PROJECT_ROOT / "HKUST-1.cif"),
        guest_xyz=str(PROJECT_ROOT / "CO2.xyz"),
        number_of_guest=1,
        number_of_system=1,
        output_dir=str(PACKMOL_OUTPUT_DIR),
        packmol_exec=str(PACKMOL_EXECUTABLE),
        max_attempts=PACKMOL_DEFAULT_MAX_ATTEMPTS,
        timeout_sec=PACKMOL_DEFAULT_TIMEOUT_SEC,
        min_mof_guest_cutoff=PACKMOL_MIN_MOF_GUEST_CUTOFF,
        min_guest_guest_cutoff=PACKMOL_MIN_GUEST_GUEST_CUTOFF,
    )
