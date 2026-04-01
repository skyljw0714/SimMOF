import os
from typing import Dict, Any
from config import working_dir


class ZeoppOutputAgent:

    @staticmethod
    def _read_res_file(mof: str, work_dir: str) -> dict:
        res_path = os.path.join(work_dir, f"{mof}.res")
        with open(res_path, "r") as f:
            line = f.readline().strip()
            parts = line.split()
            return {
                "included_sphere": float(parts[1]),
                "free_sphere": float(parts[2]),
                "included_sphere_along_free_path": float(parts[3]),
            }

    @staticmethod
    def _read_vol_file(mof: str, work_dir: str) -> dict:
        vol_path = os.path.join(work_dir, f"{mof}.vol")
        with open(vol_path, "r") as f:
            for line in f:
                if line.startswith("@"):
                    values = line.strip().split()
                    return {
                        "AV_A3": float(values[7]),
                        "AV_Volume_fraction": float(values[9]),
                        "AV_cm3_g": float(values[11]),
                    }
        return {}

    @staticmethod
    def _read_sa_file(mof: str, work_dir: str) -> dict:
        sa_path = os.path.join(work_dir, f"{mof}.sa")
        with open(sa_path, "r") as f:
            for line in f:
                if line.startswith("@"):
                    values = line.strip().split()
                    return {
                        "ASA_A2": float(values[7]),
                        "ASA_m2_cm3": float(values[9]),
                        "ASA_m2_g": float(values[11]),
                    }
        return {}

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        work_dir   = context.get("work_dir", working_dir)
        zeopp_info = context.get("zeopp_info", {})
        results    = context.setdefault("results", {})

        

        if results.get("zeopp_status") != "ok":
            print("[ZeoppOutputAgent] zeopp_status != ok -> skipping parsing")
            return context

        mof     = zeopp_info.get("MOF")
        command = zeopp_info.get("command", "")

        if not mof:
            print("[ZeoppOutputAgent] ERROR: MOF name missing in zeopp_info.")
            results["zeopp_status"] = "output_missing_mof"
        elif "-res" in command:
            parsed = self._read_res_file(mof, work_dir)
            prop_type = "pore_diameter"
        elif "-vol" in command:
            parsed = self._read_vol_file(mof, work_dir)
            prop_type = "accessible_volume"
        elif "-sa" in command:
            parsed = self._read_sa_file(mof, work_dir)
            prop_type = "surface_area"
        else:
            print("[ZeoppOutputAgent] WARNING: unknown command type, no parser matched.")
            parsed = {}
            prop_type = "unknown"

        results["zeopp"] = {
            "type": prop_type,
            "mof": mof,
            "command": command,
            "raw": parsed,
        }

        return context