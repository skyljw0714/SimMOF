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
    def _compute_diffusivity_from_traj(
        self,
        traj_path: str,
        guest_types,
        masses_by_type,
        production_start_step: int = 0,
        dt_fs_per_step: float = 1.0,
    ):
        import os
        import math
        import numpy as np
        from scipy.stats import linregress

        timesteps = []
        frames = []

        with open(traj_path, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break

                if not line.startswith("ITEM: TIMESTEP"):
                    continue

                ts = int(f.readline().strip())
                timesteps.append(ts)

                f.readline()
                natoms = int(f.readline().strip())

                f.readline()
                f.readline()
                f.readline()
                f.readline()

                line = f.readline()
                header = line.strip().split()[2:]
                col = {name: i for i, name in enumerate(header)}

                required = ["id", "mol", "type", "xu", "yu", "zu"]
                for key in required:
                    if key not in col:
                        raise RuntimeError(f"Trajectory missing column: {key}")

                frame = []
                for _ in range(natoms):
                    parts = f.readline().split()
                    frame.append({
                        "id": int(parts[col["id"]]),
                        "mol": int(parts[col["mol"]]),
                        "type": int(parts[col["type"]]),
                        "x": float(parts[col["xu"]]),
                        "y": float(parts[col["yu"]]),
                        "z": float(parts[col["zu"]]),
                    })
                frames.append(frame)

        sel = [i for i, ts in enumerate(timesteps) if ts >= production_start_step]
        if len(sel) < 2:
            raise RuntimeError("Not enough production frames in trajectory.")

        timesteps = [timesteps[i] for i in sel]
        frames = [frames[i] for i in sel]

        guest_types = set(guest_types)
        mol_map = {}

        for atom in frames[0]:
            if atom["type"] in guest_types:
                mol_id = atom["mol"]
                if mol_id <= 0:
                    mol_id = atom["id"]
                mol_map.setdefault(mol_id, []).append(atom["id"])

        if not mol_map:
            raise RuntimeError("No guest molecules found.")

        mol_ids = sorted(mol_map.keys())
        n_frames = len(frames)
        n_mol = len(mol_ids)

        needed_ids = set()
        for ids in mol_map.values():
            needed_ids.update(ids)

        com_traj = np.zeros((n_frames, n_mol, 3), dtype=float)

        for i, frame in enumerate(frames):
            atoms_by_id = {a["id"]: a for a in frame if a["id"] in needed_ids}

            for j, mol_id in enumerate(mol_ids):
                atom_ids = mol_map[mol_id]
                total_mass = 0.0
                weighted = np.zeros(3, dtype=float)

                for aid in atom_ids:
                    atom = atoms_by_id[aid]
                    atype = atom["type"]

                    if atype not in masses_by_type:
                        raise RuntimeError(f"Missing mass for atom type {atype}")

                    m = masses_by_type[atype]
                    total_mass += m
                    weighted += m * np.array([atom["x"], atom["y"], atom["z"]])

                com_traj[i, j] = weighted / total_mass

        max_lag = max(5, int((n_frames - 1) * 0.5))
        lags = np.arange(max_lag + 1)
        msd = np.zeros(max_lag + 1, dtype=float)

        for lag in lags:
            dr = com_traj[lag:] - com_traj[:n_frames - lag]
            sq = np.sum(dr * dr, axis=2)
            msd[lag] = np.mean(sq)

        dump_stride_steps = timesteps[1] - timesteps[0]
        time_fs = lags * dump_stride_steps * dt_fs_per_step
        time_ps = time_fs / 1000.0
        tmax_ps = float(time_ps[-1])

        start_fracs = [0.02, 0.05, 0.10, 0.20]
        end_fracs = [0.20, 0.30, 0.40, 0.50, 0.70]

        windows = []
        for s in start_fracs:
            for e in end_fracs:
                if e <= s:
                    continue

                w0 = s * tmax_ps
                w1 = e * tmax_ps

                min_width = max(20.0, 0.05 * tmax_ps)
                if (w1 - w0) < min_width:
                    continue

                windows.append((round(w0, 6), round(w1, 6)))

        windows = sorted(set(windows))
        results = []

        for w0, w1 in windows:
            mask = (time_ps >= w0) & (time_ps <= w1)
            npts = int(np.count_nonzero(mask))

            if npts < 30:
                continue

            x = time_fs[mask]
            y = msd[mask]

            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            if slope <= 0:
                continue

            r2 = float(r_value**2)
            D = float(slope / 6.0 * 1e-5)
            width = float(w1 - w0)
            score = r2 * math.log(width + 1.0)

            results.append({
                "fit_start_ps": float(w0),
                "fit_end_ps": float(w1),
                "n_points": npts,
                "slope": float(slope),
                "intercept": float(intercept),
                "r2": r2,
                "p_value": float(p_value),
                "std_err": float(std_err),
                "D": D,
                "score": score,
            })

        if not results:
            raise RuntimeError("No valid diffusivity window found.")

        best = max(results, key=lambda r: r["score"])
        fit_line = best["slope"] * time_fs + best["intercept"]

        out_dir = os.path.dirname(traj_path)

        np.savetxt(
            os.path.join(out_dir, "com_msd.dat"),
            np.column_stack([time_ps, msd]),
            header="time_ps msd_A2",
            comments="",
        )

        np.savetxt(
            os.path.join(out_dir, "com_msd_fit.dat"),
            np.column_stack([time_ps, fit_line]),
            header="time_ps fit_A2",
            comments="",
        )

        with open(os.path.join(out_dir, "window_scan.dat"), "w") as f:
            f.write("fit_start_ps fit_end_ps n_points D_m2_s R2 score\n")
            for r in sorted(results, key=lambda x: x["score"], reverse=True):
                f.write(
                    f"{r['fit_start_ps']:.3f} "
                    f"{r['fit_end_ps']:.3f} "
                    f"{r['n_points']} "
                    f"{r['D']:.8e} "
                    f"{r['r2']:.6f} "
                    f"{r['score']:.6f}\n"
                )

        return {
            "method": "adaptive_molecular_com_from_traj",
            "trajectory_file": os.path.basename(traj_path),
            "n_frames_used": int(n_frames),
            "n_molecules": int(n_mol),
            "molecule_ids": [int(x) for x in mol_ids],
            "lag_time_ps": time_ps.tolist(),
            "msd_A2": msd.tolist(),
            "fit_start_ps": best["fit_start_ps"],
            "fit_end_ps": best["fit_end_ps"],
            "n_fit_points": best["n_points"],
            "slope_A2_per_fs": best["slope"],
            "intercept_A2": best["intercept"],
            "D_m2_per_s": best["D"],
            "r2": best["r2"],
            "p_value": best["p_value"],
            "std_err_slope": best["std_err"],
            "score": best["score"],
            "dt_fs_per_step": float(dt_fs_per_step),
            "max_lag_ps": tmax_ps,
            "all_window_results": results,
            "com_msd_file": "com_msd.dat",
            "com_msd_fit_file": "com_msd_fit.dat",
            "window_scan_file": "window_scan.dat",
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

        if prop in ["diffusivity", "diffusion", "self_diffusivity", "self_diffusion_coefficient"]:
            traj_path = os.path.join(work_dir, "traj.lammpstrj")

            if not os.path.exists(traj_path):
                print(f"[LAMMPSOutputAgent] WARNING: {traj_path} not found.")
                results["lammps_output_status"] = "no_traj"
                return context

            try:
                guest_types = context.get("guest_types", [])
                masses_by_type = context.get("masses_by_type", {})
                production_start_step = context.get("production_start_step", 0)
                fit_start_ps = context.get("fit_start_ps", 0.0)
                fit_end_ps = context.get("fit_end_ps", None)
                dt_fs_per_step = context.get("dt_fs", 1.0)

                diff_res = self._compute_diffusivity_from_traj(
                    traj_path=traj_path,
                    guest_types=guest_types,
                    masses_by_type=masses_by_type,
                    production_start_step=production_start_step,
                    dt_fs_per_step=dt_fs_per_step,
                )

                results["lammps_output_status"] = "ok"
                results["diffusivity"] = diff_res

                print(
                    "[LAMMPSOutputAgent] D = "
                    f"{diff_res['D_m2_per_s']:.4e} m^2/s, "
                    f"slope = {diff_res['slope_A2_per_fs']:.4e} Å^2/fs, "
                    f"R^2 = {diff_res['r2']:.3f}, "
                    f"n_mol = {diff_res['n_molecules']}"
                )
                return context

            except Exception as e:
                print(f"[LAMMPSOutputAgent] ERROR: traj-based diffusivity failed: {e}")
                results["lammps_output_status"] = "traj_diffusivity_error"
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

        summ = results["msd"]["summary"]
        print(
            "[LAMMPSOutputAgent] MSD-only mode: "
            f"MSD(first,last)=({summ['msd_first_A2']:.4e}, {summ['msd_last_A2']:.4e}) Å^2 "
            f"over steps {summ['step_first']}→{summ['step_last']} (Δ={summ['msd_delta_A2']:.4e} Å^2)"
        )

        return context