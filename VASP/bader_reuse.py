import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


def is_valid_chgcar(chgcar_path: str, min_size_bytes: int = 1024 * 1024) -> Dict[str, Any]:
    p = Path(chgcar_path)
    if not p.exists():
        return {"ok": False, "reason": "file not found", "path": str(p)}
    size = p.stat().st_size
    if size < min_size_bytes:
        return {"ok": False, "reason": f"file too small ({size} bytes)", "size": size, "path": str(p)}
    
    try:
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            head = [next(f) for _ in range(10)]
    except Exception as e:
        return {"ok": False, "reason": f"read failed: {e}", "size": size, "path": str(p)}
    if sum(1 for x in head if x.strip()) < 6:
        return {"ok": False, "reason": "header seems empty/invalid", "size": size, "path": str(p)}
    return {"ok": True, "reason": "passed sanity checks", "size": size, "path": str(p)}


def patch_incar_for_charge(incar_path: Path) -> Dict[str, Any]:
    if not incar_path.exists():
        return {"ok": False, "reason": f"INCAR not found: {incar_path}"}

    lines = incar_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    def upsert(key: str, value: str):
        pat = re.compile(rf"^\s*{re.escape(key)}\s*=", re.IGNORECASE)
        for i, line in enumerate(lines):
            if pat.search(line):
                lines[i] = f"{key} = {value}"
                return
        lines.append(f"{key} = {value}")

    upsert("IBRION", "-1")
    upsert("NSW", "0")
    upsert("LCHARG", ".TRUE.")
    upsert("LAECHG", ".TRUE.")

    incar_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"ok": True, "reason": "INCAR patched for charge run", "path": str(incar_path)}


def run_bader(vasp_dir: Path) -> Dict[str, Any]:
    chgcar = vasp_dir / "CHGCAR"
    if not chgcar.exists():
        return {"status": "error", "reason": "CHGCAR missing", "CHGCAR": str(chgcar)}

    try:
        proc = subprocess.run(
            ["bader", "CHGCAR"],
            cwd=str(vasp_dir),
            capture_output=True,
            text=True,
        )
    except Exception as e:
        return {"status": "error", "reason": f"bader exec failed: {e}"}

    acf = vasp_dir / "ACF.dat"
    if proc.returncode != 0:
        return {
            "status": "error",
            "reason": "bader returned nonzero",
            "returncode": proc.returncode,
            "stdout_tail": (proc.stdout or "")[-2000:],
            "stderr_tail": (proc.stderr or "")[-2000:],
            "ACF_exists": acf.exists(),
        }
    if not acf.exists():
        return {
            "status": "error",
            "reason": "bader finished but ACF.dat not found",
            "stdout_tail": (proc.stdout or "")[-2000:],
            "stderr_tail": (proc.stderr or "")[-2000:],
        }

    return {"status": "ok", "ACF": str(acf)}


def parse_acf(acf_path: Path) -> Dict[int, float]:
    idx_to_val: Dict[int, float] = {}
    lines = acf_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    in_block = False
    for line in lines:
        if line.strip().startswith("----"):
            in_block = True
            continue
        if not in_block:
            continue
        s = line.strip()
        if not s:
            continue
        if s.lower().startswith("vacuum"):
            break
        parts = s.split()
        if len(parts) < 5:
            continue
        try:
            idx = int(parts[0])
            val = float(parts[4])
        except Exception:
            continue
        idx_to_val[idx] = val
    return idx_to_val


def make_charge_dir_from_source(
    source_vasp_dir: Path,
    charge_dir: Path,
    submit_label: str,
) -> Dict[str, Any]:
    charge_dir.mkdir(parents=True, exist_ok=True)

    
    for fn in ["DONE", "FAILED", "START"]:
        p = charge_dir / fn
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    src_pos = source_vasp_dir / "CONTCAR"
    if not src_pos.exists():
        src_pos = source_vasp_dir / "POSCAR"
    if not src_pos.exists():
        return {"ok": False, "reason": f"no POSCAR/CONTCAR in {source_vasp_dir}", "charge_dir": str(charge_dir)}

    shutil.copy2(src_pos, charge_dir / "POSCAR")

    for fn in ["POTCAR", "KPOINTS", "INCAR"]:
        src = source_vasp_dir / fn
        if src.exists():
            shutil.copy2(src, charge_dir / fn)

    incar_path = charge_dir / "INCAR"
    if not incar_path.exists():
        return {"ok": False, "reason": f"INCAR missing in charge_dir={charge_dir}", "charge_dir": str(charge_dir)}

    
    qsubs = sorted(source_vasp_dir.glob("*.qsub"))
    if not qsubs:
        return {"ok": False, "reason": f"no *.qsub found in source_vasp_dir={source_vasp_dir}", "charge_dir": str(charge_dir)}
    
    shutil.copy2(qsubs[0], charge_dir / f"{submit_label}.qsub")

    patch_res = patch_incar_for_charge(incar_path)
    if not patch_res.get("ok"):
        return {"ok": False, "reason": patch_res.get("reason", "INCAR patch failed"), "charge_dir": str(charge_dir)}

    return {"ok": True, "charge_dir": str(charge_dir), "incar_patch": patch_res, "qsub_src": str(qsubs[0])}
