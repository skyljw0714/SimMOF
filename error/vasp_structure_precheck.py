from __future__ import annotations

import math
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ase.io
from ase.neighborlist import neighbor_list


class VASPStructurePrecheckAgent:

    def __init__(
        self,
        *,
        min_distance_a: float = 0.55,
        suspicious_distance_a: float = 0.75,
        min_cell_length_a: float = 1.5,
        max_cell_length_a: float = 120.0,
        max_atoms: int = 5000,
        auto_wrap: bool = True,
    ):
        self.min_distance_a = min_distance_a
        self.suspicious_distance_a = suspicious_distance_a
        self.min_cell_length_a = min_cell_length_a
        self.max_cell_length_a = max_cell_length_a
        self.max_atoms = max_atoms
        self.auto_wrap = auto_wrap

    def _get_system_dir(self, context: Dict[str, Any]) -> Optional[Path]:
        sys_info = context.get("vasp_system")
        if isinstance(sys_info, dict) and sys_info.get("dir"):
            return Path(sys_info["dir"])
        if context.get("vasp_dir"):
            return Path(context["vasp_dir"])
        return None

    @staticmethod
    def _backup(path: Path, suffix: str) -> Optional[str]:
        if not path.exists():
            return None
        backup = path.with_name(f"{path.name}.simmof_{suffix}_{time.strftime('%Y%m%d_%H%M%S')}")
        shutil.copy2(path, backup)
        return str(backup)

    @staticmethod
    def _finite_positions(atoms) -> bool:
        for row in atoms.get_positions():
            for value in row:
                if not math.isfinite(float(value)):
                    return False
        return True

    def _cell_issues(self, atoms) -> List[str]:
        issues: List[str] = []
        lengths = atoms.cell.lengths()
        volume = atoms.get_volume()

        if any(not math.isfinite(float(x)) for x in lengths) or not math.isfinite(float(volume)):
            issues.append("cell has non-finite length or volume")
            return issues
        if volume <= 1e-6:
            issues.append(f"cell volume is non-positive or near zero: {volume:.6g}")
        for i, length in enumerate(lengths, start=1):
            if length < self.min_cell_length_a:
                issues.append(f"cell vector {i} is too short: {length:.3f} A")
            if length > self.max_cell_length_a:
                issues.append(f"cell vector {i} is unusually long: {length:.3f} A")
        return issues

    def _short_contacts(self, atoms) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        fatal: List[Dict[str, Any]] = []
        suspicious: List[Dict[str, Any]] = []
        n_atoms = len(atoms)
        if n_atoms < 2:
            return fatal, suspicious

        cutoffs = [self.suspicious_distance_a / 2.0] * n_atoms
        try:
            i_list, j_list, d_list = neighbor_list("ijd", atoms, cutoffs)
        except Exception:
            return fatal, suspicious

        seen = set()
        for i, j, dist in zip(i_list, j_list, d_list):
            if i == j:
                continue
            key = tuple(sorted((int(i), int(j))))
            if key in seen:
                continue
            seen.add(key)
            item = {
                "i": int(i),
                "j": int(j),
                "symbols": f"{atoms[int(i)].symbol}-{atoms[int(j)].symbol}",
                "distance_a": round(float(dist), 4),
            }
            if dist < self.min_distance_a:
                fatal.append(item)
            else:
                suspicious.append(item)

        fatal.sort(key=lambda x: x["distance_a"])
        suspicious.sort(key=lambda x: x["distance_a"])
        return fatal[:20], suspicious[:20]

    def _wrap_positions_if_needed(self, atoms, poscar: Path) -> Optional[str]:
        if not self.auto_wrap or not any(atoms.pbc):
            return None
        scaled = atoms.get_scaled_positions(wrap=False)
        needs_wrap = False
        for row in scaled:
            for value in row:
                if value < -1e-6 or value >= 1.0 + 1e-6:
                    needs_wrap = True
                    break
            if needs_wrap:
                break
        if not needs_wrap:
            return None
        backup = self._backup(poscar, "backup_before_wrap")
        atoms.wrap()
        ase.io.write(poscar, atoms, format="vasp", vasp5=True, direct=True)
        return backup

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if context.get("vasp_status") == "needs_structure_from_user":
            context["vasp_structure_precheck_status"] = (
                "blocked_missing_structure"
            )
            context.setdefault("results", {})[
                "vasp_structure_precheck"
            ] = {
                "status": "blocked_missing_structure",
                "issues": ["source CIF is unavailable"],
            }
            return context

        system_dir = self._get_system_dir(context)
        if system_dir is None:
            return context

        poscar = system_dir / "POSCAR"
        report: Dict[str, Any] = {
            "status": "ok",
            "poscar": str(poscar),
            "issues": [],
            "warnings": [],
            "actions": [],
        }

        if not poscar.is_file():
            report["status"] = "failed"
            report["issues"].append("POSCAR is missing")
            context.setdefault("results", {})["vasp_structure_precheck"] = report
            context["vasp_structure_precheck_status"] = "failed"
            return context

        try:
            atoms = ase.io.read(poscar)
        except Exception as exc:
            report["status"] = "failed"
            report["issues"].append(f"ASE could not read POSCAR: {exc}")
            context.setdefault("results", {})["vasp_structure_precheck"] = report
            context["vasp_structure_precheck_status"] = "failed"
            return context

        if len(atoms) == 0:
            report["issues"].append("structure has zero atoms")
        if len(atoms) > self.max_atoms:
            report["warnings"].append(f"large VASP structure: {len(atoms)} atoms")
        if not self._finite_positions(atoms):
            report["issues"].append("structure has NaN/inf atomic coordinates")

        report["issues"].extend(self._cell_issues(atoms))

        wrap_backup = self._wrap_positions_if_needed(atoms, poscar)
        if wrap_backup:
            report["actions"].append({"action": "wrap_positions", "backup": wrap_backup})
            atoms = ase.io.read(poscar)

        fatal_contacts, suspicious_contacts = self._short_contacts(atoms)
        if fatal_contacts:
            report["issues"].append(
                f"fatal short contacts below {self.min_distance_a:.2f} A detected"
            )
            report["fatal_short_contacts"] = fatal_contacts
        if suspicious_contacts:
            report["warnings"].append(
                f"suspicious short contacts below {self.suspicious_distance_a:.2f} A detected"
            )
            report["suspicious_short_contacts"] = suspicious_contacts

        if report["issues"]:
            report["status"] = "failed"

        context.setdefault("results", {})["vasp_structure_precheck"] = report
        context["vasp_structure_precheck_status"] = report["status"]
        if report["status"] == "failed":
            context["vasp_status"] = "structure_precheck_failed"
            print("[VASPStructurePrecheckAgent] failed:", report["issues"])
        elif report["warnings"]:
            print("[VASPStructurePrecheckAgent] warnings:", report["warnings"])
        return context
