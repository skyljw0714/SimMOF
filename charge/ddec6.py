
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import (
    CHARGEMOL_ATOMIC_DENSITIES_DIR,
    CHARGEMOL_BIN,
    WORKING_DIR,
)

_DDEC6_WORK_ROOT = WORKING_DIR / "ddec6"

_INCAR_TEMPLATE = """\
SYSTEM = DDEC6 charge calculation
ISTART = 0
ICHARG = 2
NSW    = 0
IBRION = -1
ISIF   = 0
EDIFF  = 1E-5
ENCUT  = 520
ISMEAR = 0
SIGMA  = 0.05
LREAL  = Auto
LAECHG = TRUE
LCHARG = TRUE
NCORE  = 4
"""

_KPOINTS_GAMMA = """\
Automatic
0
Gamma
 1 1 1
 0 0 0
"""


def _write_chargemol_job_control(vasp_dir: Path) -> None:
    job_control = (
        "<periodicity along A, B, and C vectors>\n"
        ".true.\n"
        ".true.\n"
        ".true.\n"
        "</periodicity along A, B, and C vectors>\n"
        "\n"
        "<atomic densities directory complete path>\n"
        f"{CHARGEMOL_ATOMIC_DENSITIES_DIR}/\n"
        "</atomic densities directory complete path>\n"
        "\n"
        "<charge type>\n"
        "DDEC6\n"
        "</charge type>\n"
    )
    (vasp_dir / "job_control.txt").write_text(job_control)


def _submit_vasp_job(vasp_dir: Path, label: str) -> Optional[str]:
    qsub_path = vasp_dir / f"{label}.qsub"
    if not qsub_path.exists():
        raise FileNotFoundError(f"qsub not found: {qsub_path}")
    proc = subprocess.run(
        ["qas", str(qsub_path)],
        cwd=str(vasp_dir),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qas submission failed: {proc.stderr}")
    stdout = proc.stdout.strip()
    return stdout.split()[0] if stdout else None


def _poll_vasp_done(vasp_dir: Path, check_interval: int = 60, max_hours: float = 24.0) -> bool:
    deadline = time.time() + max_hours * 3600
    while time.time() < deadline:
        if (vasp_dir / "DONE").exists():
            return True
        if (vasp_dir / "FAILED").exists():
            return False
        time.sleep(check_interval)
    print(f"[DDEC6] Timeout waiting for VASP: {vasp_dir}")
    return False


def _strip_potcar_sha256(vasp_dir: Path) -> None:
    potcar = vasp_dir / "POTCAR"
    lines = potcar.read_text(errors="replace").splitlines(keepends=True)
    skip_prefixes = ("   SHA256", "   COPYR")
    cleaned = [l for l in lines if not any(l.startswith(p) for p in skip_prefixes)]
    potcar.write_text("".join(cleaned))


def _run_chargemol(vasp_dir: Path) -> bool:
    if not CHARGEMOL_BIN.exists():
        raise FileNotFoundError(f"CHARGEMOL binary not found: {CHARGEMOL_BIN}")
    _strip_potcar_sha256(vasp_dir)
    proc = subprocess.run(
        [str(CHARGEMOL_BIN)],
        cwd=str(vasp_dir),
        capture_output=True,
        text=True,
    )
    xyz = vasp_dir / "DDEC6_even_tempered_net_atomic_charges.xyz"
    if proc.returncode != 0 or not xyz.exists():
        print(f"[DDEC6] CHARGEMOL failed:\n{proc.stderr[:500]}")
        return False
    return True


def _parse_ddec6_charges(vasp_dir: Path) -> List[float]:
    xyz_path = vasp_dir / "DDEC6_even_tempered_net_atomic_charges.xyz"
    if not xyz_path.exists():
        raise FileNotFoundError(f"CHARGEMOL output not found: {xyz_path}")

    charges: List[float] = []
    lines = xyz_path.read_text().splitlines()
    if len(lines) < 3:
        raise ValueError("CHARGEMOL output too short")
    n_atoms = int(lines[0].strip())
    for line in lines[2: 2 + n_atoms]:
        parts = line.split()
        if len(parts) >= 5:
            charges.append(float(parts[4]))
    if len(charges) != n_atoms:
        raise ValueError(f"Expected {n_atoms} charges, got {len(charges)}")
    return charges


def _write_charges_to_cif(cif_path: Path, charges: List[float]) -> None:
    text = cif_path.read_text(errors="replace")

    loop_pattern = re.compile(
        r"(loop_\s*\n(?:\s*_atom_site_\S+\s*\n)+)((?:(?!\s*loop_|\s*_\w).*\n)*)",
        re.MULTILINE,
    )
    match = loop_pattern.search(text)
    if not match:
        raise ValueError("Cannot find _atom_site loop in CIF")

    header_block = match.group(1)
    data_block = match.group(2)

    if "_atom_site_charge" not in header_block:
        header_block = header_block.rstrip("\n") + "\n  _atom_site_charge\n"

    header_cols = re.findall(r"_atom_site_\S+", header_block)
    charge_col = (
        header_cols.index("_atom_site_charge")
        if "_atom_site_charge" in header_cols
        else len(header_cols) - 1
    )

    data_lines = [ln for ln in data_block.splitlines() if ln.strip()]
    if len(data_lines) != len(charges):
        raise ValueError(
            f"Atom count mismatch: CIF has {len(data_lines)}, charges has {len(charges)}"
        )

    charge_already_in_header = "_atom_site_charge" in re.findall(
        r"_atom_site_\S+", match.group(1)
    )
    new_data_lines = []
    for i, line in enumerate(data_lines):
        parts = line.split()
        if not charge_already_in_header:
            parts.append(f"{charges[i]:.6f}")
        else:
            while len(parts) <= charge_col:
                parts.append("0.000000")
            parts[charge_col] = f"{charges[i]:.6f}"
        new_data_lines.append("  " + "  ".join(parts))

    new_block = header_block + "\n".join(new_data_lines) + "\n"
    new_text = text[: match.start()] + new_block + text[match.end():]
    cif_path.write_text(new_text)


def _write_poscar_unsorted(cif_path: Path, job_dir: Path) -> None:
    import numpy as np
    import ase.io
    import ase.io.vasp
    from config import VASP_POTENTIAL_DIR_PATH
    from file.agent import VASPFileAgent

    atoms = ase.io.read(str(cif_path))
    order = np.argsort(atoms.get_chemical_symbols())
    atoms = atoms[order]
    ase.io.write(str(cif_path), atoms, format="cif")
    ase.io.vasp.write_vasp(str(job_dir / "POSCAR"), atoms, direct=True, sort=False, vasp5=True)
    pot_dir = str(VASP_POTENTIAL_DIR_PATH) + "/"
    VASPFileAgent.atoms_to_potcar(atoms, str(job_dir) + "/", pot_dir)


def run_ddec6_on_cif(
    cif_path: Path,
    context: Dict[str, Any],
    work_dir: Optional[Path] = None,
    check_interval: int = 60,
    max_hours: float = 24.0,
) -> bool:
    from core.resource_allocator import ResourceAllocator
    from file.agent import VASPFileAgent

    cif_path = Path(cif_path)
    label = cif_path.stem + "_ddec6"
    job_dir = (work_dir or _DDEC6_WORK_ROOT) / label
    job_dir.mkdir(parents=True, exist_ok=True)

    print(f"[DDEC6] Setting up VASP in {job_dir}")

    try:
        n_atoms = len(ase.io.read(str(cif_path)))
    except Exception:
        n_atoms = 0
    spec = context.get("resource_spec")
    if spec is None:
        spec = ResourceAllocator().recommend("VASP", "DDEC6 charge calculation", n_atoms, context)
        context["resource_spec"] = spec
    _write_poscar_unsorted(cif_path, job_dir)

    (job_dir / "INCAR").write_text(_INCAR_TEMPLATE)
    (job_dir / "KPOINTS").write_text(_KPOINTS_GAMMA)
    VASPFileAgent.make_qsub(str(job_dir) + "/", label, spec=spec)

    try:
        job_id = _submit_vasp_job(job_dir, label)
        print(f"[DDEC6] Submitted VASP job: {job_id}")
    except Exception as e:
        print(f"[DDEC6] VASP submission failed: {e}")
        return False

    print(f"[DDEC6] Waiting for VASP (max {max_hours}h)...")
    done = _poll_vasp_done(job_dir, check_interval=check_interval, max_hours=max_hours)
    if not done:
        print(f"[DDEC6] VASP failed or timed out: {job_dir}")
        return False
    print("[DDEC6] VASP finished. Running CHARGEMOL...")

    _write_chargemol_job_control(job_dir)
    try:
        chargemol_ok = _run_chargemol(job_dir)
    except FileNotFoundError as e:
        print(f"[DDEC6] {e}")
        return False
    if not chargemol_ok:
        return False

    try:
        charges = _parse_ddec6_charges(job_dir)
        _write_charges_to_cif(cif_path, charges)
        print(f"[DDEC6] DDEC6 charges written -> {cif_path.name} ({len(charges)} atoms)")
        return True
    except Exception as e:
        print(f"[DDEC6] Failed to write charges: {e}")
        return False
