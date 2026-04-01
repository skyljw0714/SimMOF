import os
import numpy as np

from typing import Dict, Any, List, Tuple
from pathlib import Path
from scipy.stats import linregress


class LAMMPSOutputAgent:
    def _parse_thermal_summary(self, path: str):
        T, V, lx, ly, lz = [], [], [], [], []

        with open(path, "r") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = s.split()
                if len(parts) < 2:
                    continue

                try:
                    t = float(parts[0])
                    v = float(parts[1])
                    T.append(t)
                    V.append(v)

                    
                    if len(parts) >= 5:
                        lx.append(float(parts[2]))
                        ly.append(float(parts[3]))
                        lz.append(float(parts[4]))
                    else:
                        lx.append(np.nan)
                        ly.append(np.nan)
                        lz.append(np.nan)
                except ValueError:
                    continue

        if len(T) < 2:
            raise RuntimeError(f"Not enough points in thermal expansion summary: {path}")

        return {
            "T_K": T,
            "V": V,
            "lx": lx,
            "ly": ly,
            "lz": lz,
        }

    def _fit_alpha_from_VT(self, T: np.ndarray, V: np.ndarray, T_ref: float = 300.0):
        
        mask = np.isfinite(T) & np.isfinite(V) & (V > 0.0)
        T2 = T[mask]; V2 = V[mask]
        if len(T2) < 3:
            raise RuntimeError(f"Not enough valid positive V(T) points (n={len(T2)}).")

        slope, intercept, r_value, p_value, std_err = linregress(T2, V2)

        
        idx = int(np.argmin(np.abs(T2 - T_ref)))
        Vref = float(V2[idx])

        
        if not np.isfinite(Vref) or abs(Vref) < 1e-12:
            
            nz = np.where(np.isfinite(V2) & (np.abs(V2) >= 1e-12))[0]
            if len(nz) == 0:
                raise RuntimeError("All V values are zero/invalid; cannot compute alpha.")
            
            idx2 = int(nz[np.argmin(np.abs(T2[nz] - T_ref))])
            Vref = float(V2[idx2])
            idx = idx2

        alpha_V = float(slope / Vref)

        return {
            "dVdT": float(slope),
            "intercept": float(intercept),
            "r2": float(r_value**2),
            "p_value": float(p_value),
            "std_err_slope": float(std_err),
            "T_ref_K": float(T2[idx]),
            "V_ref": float(Vref),
            "alpha_V_per_K": alpha_V,
        }

    def _parse_msd_file(self, path: str) -> Tuple[List[int], List[float]]:
        steps = []
        msd = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    step_i = int(float(parts[0]))
                    msd_i = float(parts[1])
                except ValueError:
                    continue
                steps.append(step_i)
                msd.append(msd_i)
        return steps, msd

    def _summarize_msd(self, steps: np.ndarray, msd: np.ndarray) -> Dict[str, Any]:
        if len(steps) == 0:
            raise RuntimeError("Empty MSD series")

        
        msd0 = float(msd[0])
        msd_last = float(msd[-1])
        delta = msd_last - msd0

        
        n_up = int(np.sum(np.diff(msd) > 0))
        n_dn = int(np.sum(np.diff(msd) < 0))

        return {
            "n_points": int(len(msd)),
            "step_first": int(steps[0]),
            "step_last": int(steps[-1]),
            "msd_first_A2": msd0,
            "msd_last_A2": msd_last,
            "msd_delta_A2": float(delta),
            "n_increases": n_up,
            "n_decreases": n_dn,
        }

    def _compute_diffusivity(
        self,
        steps: np.ndarray,
        msd: np.ndarray,
        dt_fs: float = 1.0,
        n_skip: int = 0,
    ) -> Dict[str, float]:
        if len(steps) <= n_skip + 2:
            raise RuntimeError(
                f"Not enough MSD points for regression (len={len(steps)}, n_skip={n_skip})"
            )

        
        time_fs = steps * dt_fs

        x = time_fs[n_skip:]
        y = msd[n_skip:]

        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        
        
        D_m2_s = slope / 6.0 * 1e-5

        return {
            "slope_A2_per_fs": slope,
            "intercept_A2": intercept,
            "D_m2_per_s": D_m2_s,
            "r2": r_value**2,
            "p_value": p_value,
            "std_err_slope": std_err,
        }

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        work_dir = context.get("work_dir")
        if not work_dir:
            raise RuntimeError("LAMMPSOutputAgent.run: context['work_dir'] is missing.")

        print(f"\n=== LAMMPSOutputAgent: parsing outputs in {work_dir} ===")
        results = context.setdefault("results", {})

        
        if not context.get("lammps_success", False):
            print("[LAMMPSOutputAgent] lammps_success=False -> skipping output parsing")
            results["lammps_output_status"] = "failed"
            return context

        prop = (context.get("property") or context.get("simulation_property") or "").lower()

        
        if ("thermal_expansion" in prop) or ("thermal expansion" in prop) or ("cte" in prop):
            summary_path = os.path.join(work_dir, "thermal_expansion_summary.dat")

            if not os.path.exists(summary_path):
                print(f"[LAMMPSOutputAgent] WARNING: {summary_path} not found.")
                results["lammps_output_status"] = "no_thermal_summary"
                return context

            try:
                te = self._parse_thermal_summary(summary_path)
            except Exception as e:
                print(f"[LAMMPSOutputAgent] ERROR parsing thermal expansion summary: {e}")
                results["lammps_output_status"] = "thermal_parse_error"
                return context

            T = np.array(te["T_K"], dtype=float)
            V = np.array(te["V"], dtype=float)

            try:
                fit = self._fit_alpha_from_VT(T, V, T_ref=300.0)
            except Exception as e:
                print(f"[LAMMPSOutputAgent] ERROR fitting alpha_V: {e}")
                results["thermal_expansion"] = {"raw": te}
                results["lammps_output_status"] = "thermal_fit_error"
                return context

            results["thermal_expansion"] = {
                "summary_file": "thermal_expansion_summary.dat",
                "raw": te,
                "fit_VT": fit,
            }
            results["lammps_output_status"] = "ok"

            print(
                "[LAMMPSOutputAgent] thermal expansion: "
                f"alpha_V = {fit['alpha_V_per_K']:.4e} 1/K, "
                f"dV/dT = {fit['dVdT']:.4e}, R^2 = {fit['r2']:.3f}"
            )
            return context

        
        msd_path = os.path.join(work_dir, "msd_guest.dat")
        if not os.path.exists(msd_path):
            print(f"[LAMMPSOutputAgent] WARNING: {msd_path} not found.")
            results["lammps_output_status"] = "no_msd"
            return context

        
        steps_list, msd_list = self._parse_msd_file(msd_path)
        if not steps_list:
            print(f"[LAMMPSOutputAgent] WARNING: {msd_path} is empty or could not be parsed.")
            results["lammps_output_status"] = "empty_msd"
            return context

        steps = np.array(steps_list, dtype=float)
        msd = np.array(msd_list, dtype=float)

        results["lammps_output_status"] = "ok"
        results["msd"] = {
            "file": "msd_guest.dat",
            "steps": steps_list,
            "msd_A2": msd_list,
            "summary": self._summarize_msd(steps, msd),
        }

        
        if prop in ["diffusivity", "diffusion", "self_diffusivity", "self_diffusion_coefficient"]:
            dt_fs = context.get("dt_fs", 1.0)
            n_skip = context.get("msd_n_skip", 0)

            try:
                diff_res = self._compute_diffusivity(steps, msd, dt_fs=dt_fs, n_skip=n_skip)
            except Exception as e:
                print(f"[LAMMPSOutputAgent] ERROR during diffusivity computation: {e}")
                results["lammps_output_status"] = "analysis_error"
                return context

            results["lammps_output_status"] = "ok"
            results["diffusivity"] = diff_res

            print(
                "[LAMMPSOutputAgent] D = "
                f"{diff_res['D_m2_per_s']:.4e} m^2/s, "
                f"slope = {diff_res['slope_A2_per_fs']:.4e} Å^2/fs, "
                f"R^2 = {diff_res['r2']:.3f}"
            )
        else:
            
            summ = results["msd"]["summary"]
            print(
                "[LAMMPSOutputAgent] MSD-only mode: "
                f"MSD(first,last)=({summ['msd_first_A2']:.4e}, {summ['msd_last_A2']:.4e}) Å^2 "
                f"over steps {summ['step_first']}→{summ['step_last']} (Δ={summ['msd_delta_A2']:.4e} Å^2)"
            )

        return context
