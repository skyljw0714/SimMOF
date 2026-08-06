from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import linregress


METAL_ELEMENTS = {
    "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe",
    "Co", "Ni", "Cu", "Zn", "Ga", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru",
    "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Sm",
    "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re",
    "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
}


def _clean_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def _parse_guest_types(system_in: Path) -> List[int]:
    if not system_in.exists():
        return []
    text = system_in.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"^\s*group\s+guest\s+type\s+(.+)$", text, re.MULTILINE)
    if not m:
        return []
    out = []
    for token in m.group(1).split():
        try:
            out.append(int(token))
        except ValueError:
            pass
    return sorted(set(out))


def _parse_dt_fs(system_in: Path) -> float:
    if not system_in.exists():
        return 1.0
    dt = 1.0
    for raw in system_in.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.match(r"^\s*timestep\s+([-+0-9.eE]+)", raw)
        if m:
            parsed = _clean_float(m.group(1))
            if parsed is not None:
                dt = parsed
    return float(dt)


def _parse_masses_and_type_labels(system_data: Path) -> Tuple[Dict[int, float], Dict[int, str]]:
    masses: Dict[int, float] = {}
    labels: Dict[int, str] = {}
    if not system_data.exists():
        return masses, labels

    lines = system_data.read_text(encoding="utf-8", errors="ignore").splitlines()
    in_masses = False
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.lower() == "masses":
            in_masses = True
            continue
        if not in_masses:
            continue
        if re.match(r"^[A-Za-z]", line):
            break
        parts = line.split("#", 1)
        cols = parts[0].split()
        if len(cols) < 2:
            continue
        try:
            atom_type = int(cols[0])
            mass = float(cols[1])
        except ValueError:
            continue
        masses[atom_type] = mass
        labels[atom_type] = parts[1].strip() if len(parts) > 1 else str(atom_type)
    return masses, labels


def _count_frames(traj_path: Path) -> int:
    n = 0
    with traj_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("ITEM: TIMESTEP"):
                n += 1
    return n


def _triclinic_cell_from_bounds(bounds: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
    xlo_b, xhi_b, xy = bounds[0]
    ylo_b, yhi_b, xz = bounds[1]
    zlo_b, zhi_b, yz = bounds[2]

    xlo = xlo_b - min(0.0, xy, xz, xy + xz)
    xhi = xhi_b - max(0.0, xy, xz, xy + xz)
    ylo = ylo_b - min(0.0, yz)
    yhi = yhi_b - max(0.0, yz)
    zlo = zlo_b
    zhi = zhi_b

    cell = np.array(
        [
            [xhi - xlo, 0.0, 0.0],
            [xy, yhi - ylo, 0.0],
            [xz, yz, zhi - zlo],
        ],
        dtype=float,
    )
    origin = np.array([xlo, ylo, zlo], dtype=float)
    return cell, origin


def _iter_lammps_frames(
    traj_path: Path,
    frame_stride: int = 1,
    max_frames: Optional[int] = None,
):
    kept = 0
    seen = 0
    with traj_path.open("r", encoding="utf-8", errors="ignore") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue

            timestep = int(float(f.readline().strip()))
            f.readline()
            n_atoms = int(float(f.readline().strip()))
            box_header = f.readline().strip()
            bounds = []
            for _ in range(3):
                vals = [float(x) for x in f.readline().split()]
                if len(vals) == 2:
                    vals.append(0.0)
                bounds.append(vals[:3])
            atoms_header = f.readline().strip().split()[2:]

            take = (seen % max(1, frame_stride) == 0) and (max_frames is None or kept < max_frames)
            seen += 1
            if not take:
                for _ in range(n_atoms):
                    f.readline()
                continue

            col = {name: i for i, name in enumerate(atoms_header)}
            required = {"id", "mol", "type"}
            if not required <= set(col):
                raise RuntimeError(f"Trajectory is missing required columns: {sorted(required - set(col))}")

            coord_cols = None
            for names in (("xu", "yu", "zu"), ("x", "y", "z")):
                if set(names) <= set(col):
                    coord_cols = names
                    break
            if coord_cols is None:
                raise RuntimeError("Trajectory needs x/y/z or xu/yu/zu coordinates.")

            ids = np.empty(n_atoms, dtype=int)
            mols = np.empty(n_atoms, dtype=int)
            types = np.empty(n_atoms, dtype=int)
            pos = np.empty((n_atoms, 3), dtype=float)
            for i in range(n_atoms):
                parts = f.readline().split()
                ids[i] = int(float(parts[col["id"]]))
                mols[i] = int(float(parts[col["mol"]]))
                types[i] = int(float(parts[col["type"]]))
                pos[i, 0] = float(parts[col[coord_cols[0]]])
                pos[i, 1] = float(parts[col[coord_cols[1]]])
                pos[i, 2] = float(parts[col[coord_cols[2]]])

            cell, origin = _triclinic_cell_from_bounds(bounds)
            kept += 1
            yield {
                "timestep": timestep,
                "ids": ids,
                "mols": mols,
                "types": types,
                "positions": pos,
                "cell": cell,
                "origin": origin,
                "box_header": box_header,
            }


def _minimum_image_displacements(points: np.ndarray, refs: np.ndarray, cell: np.ndarray) -> np.ndarray:
    inv_cell = np.linalg.inv(cell)
    delta = points[:, None, :] - refs[None, :, :]
    frac = delta @ inv_cell
    frac -= np.round(frac)
    return frac @ cell


def _wrap_positions(pos: np.ndarray, cell: np.ndarray, origin: np.ndarray) -> np.ndarray:
    inv_cell = np.linalg.inv(cell)
    frac = (pos - origin) @ inv_cell
    frac -= np.floor(frac)
    return frac @ cell + origin


def _guest_com_for_frame(frame: Dict[str, Any], guest_types: Sequence[int], masses: Dict[int, float]) -> Tuple[np.ndarray, List[int]]:
    guest_types_set = set(int(x) for x in guest_types)
    mask = np.array([t in guest_types_set for t in frame["types"]], dtype=bool)
    if not np.any(mask):
        return np.empty((0, 3), dtype=float), []

    mol_ids = frame["mols"][mask].copy()
    atom_ids = frame["ids"][mask]
    mol_ids[mol_ids <= 0] = atom_ids[mol_ids <= 0]
    positions = frame["positions"][mask]
    types = frame["types"][mask]

    out = []
    out_ids = []
    for mol_id in sorted(set(int(x) for x in mol_ids)):
        idx = np.where(mol_ids == mol_id)[0]
        weights = np.array([masses.get(int(types[i]), 1.0) for i in idx], dtype=float)
        total = float(np.sum(weights))
        if total <= 0.0:
            weights[:] = 1.0
            total = float(len(weights))
        out.append(np.sum(positions[idx] * weights[:, None], axis=0) / total)
        out_ids.append(int(mol_id))
    return np.array(out, dtype=float), out_ids


def _fit_diffusion(time_fs: np.ndarray, msd: np.ndarray, dimensions: int) -> Dict[str, Any]:
    if len(time_fs) < 6:
        return {"status": "insufficient_points"}
    tmax = float(time_fs[-1])
    mask = (time_fs >= 0.10 * tmax) & (time_fs <= 0.50 * tmax)
    if int(np.count_nonzero(mask)) < 5:
        mask = np.arange(len(time_fs)) >= max(1, len(time_fs) // 5)
    x = time_fs[mask]
    y = msd[mask]
    if len(x) < 3:
        return {"status": "insufficient_fit_points"}
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    factor = 2.0 * float(dimensions)
    status = "ok" if slope > 0.0 else "non_diffusive_or_poor_fit"
    return {
        "status": status,
        "fit_start_ps": float(x[0] / 1000.0),
        "fit_end_ps": float(x[-1] / 1000.0),
        "n_fit_points": int(len(x)),
        "slope_A2_per_fs": float(slope),
        "intercept_A2": float(intercept),
        "D_m2_s": float(slope / factor * 1.0e-5) if slope > 0.0 else None,
        "r2": float(r_value**2),
        "p_value": float(p_value),
        "std_err_slope": float(std_err),
    }


def _component_msd_analysis(timesteps: List[int], com_traj: np.ndarray, dt_fs: float) -> Dict[str, Any]:
    n_frames = com_traj.shape[0]
    max_lag = max(2, int((n_frames - 1) * 0.5))
    lags = np.arange(max_lag + 1, dtype=int)
    dump_stride = float(np.median(np.diff(timesteps))) if len(timesteps) > 1 else 1.0
    time_fs = lags * dump_stride * dt_fs

    msd_components = np.zeros((len(lags), 3), dtype=float)
    for out_i, lag in enumerate(lags):
        dr = com_traj[lag:] - com_traj[: n_frames - lag]
        msd_components[out_i] = np.mean(dr * dr, axis=(0, 1))
    msd_total = np.sum(msd_components, axis=1)

    axes = {}
    for i, label in enumerate(("x", "y", "z")):
        axes[label] = _fit_diffusion(time_fs, msd_components[:, i], dimensions=1)
    total = _fit_diffusion(time_fs, msd_total, dimensions=3)
    if total.get("status") == "ok":
        dvals = [axes[a].get("D_m2_s") for a in ("x", "y", "z") if axes[a].get("status") == "ok"]
        if dvals:
            total["anisotropy_ratio_max_min"] = float(max(dvals) / min(dvals)) if min(dvals) > 0 else None
            total["D_axis_mean_m2_s"] = float(np.mean(dvals))

    return {
        "definition": "Directional diffusivity from molecular COM MSD components using unwrapped coordinates.",
        "time_ps": (time_fs / 1000.0).tolist(),
        "msd_x_A2": msd_components[:, 0].tolist(),
        "msd_y_A2": msd_components[:, 1].tolist(),
        "msd_z_A2": msd_components[:, 2].tolist(),
        "msd_total_A2": msd_total.tolist(),
        "fit_total": total,
        "fit_axes": axes,
    }


def _vector_autocorrelation(vectors: np.ndarray, max_lag: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    n_frames = vectors.shape[0]
    if n_frames < 3:
        return np.array([], dtype=int), np.array([], dtype=float)
    max_lag = min(max_lag if max_lag is not None else n_frames // 2, n_frames - 1)
    lags = np.arange(max_lag + 1, dtype=int)
    corr = np.zeros(len(lags), dtype=float)
    for out_i, lag in enumerate(lags):
        a = vectors[: n_frames - lag]
        b = vectors[lag:]
        corr[out_i] = float(np.mean(np.sum(a * b, axis=2)))
    if abs(corr[0]) > 1e-30:
        corr = corr / corr[0]
    return lags, corr


def _velocity_autocorrelation(timesteps: List[int], com_traj: np.ndarray, dt_fs: float) -> Dict[str, Any]:
    if com_traj.shape[0] < 4:
        return {"status": "insufficient_frames"}
    dump_stride = float(np.median(np.diff(timesteps))) if len(timesteps) > 1 else 1.0
    frame_dt_fs = dump_stride * dt_fs
    velocities = np.diff(com_traj, axis=0) / max(frame_dt_fs, 1e-30)
    lags, corr = _vector_autocorrelation(velocities, max_lag=max(1, min(200, velocities.shape[0] // 2)))
    time_ps = lags * frame_dt_fs / 1000.0
    first_zero = None
    for t, c in zip(time_ps[1:], corr[1:]):
        if c <= 0:
            first_zero = float(t)
            break
    return {
        "definition": "Velocity autocorrelation of guest molecular COM velocities estimated by finite differences from unwrapped trajectory coordinates.",
        "status": "ok" if corr.size else "insufficient_frames",
        "time_ps": time_ps.tolist(),
        "vacf_normalized": corr.tolist(),
        "first_zero_crossing_ps": first_zero,
        "frame_dt_ps": float(frame_dt_fs / 1000.0),
        "limitations": [
            "Finite-difference VACF needs a sufficiently small dump stride; coarse dumps smooth out fast collisions.",
            "Velocities are reconstructed from COM displacements, not read from native vx/vy/vz trajectory columns.",
        ],
    }


def _van_hove_non_gaussian(timesteps: List[int], com_traj: np.ndarray, dt_fs: float, bin_width_A: float = 0.2) -> Dict[str, Any]:
    n_frames = com_traj.shape[0]
    if n_frames < 4:
        return {"status": "insufficient_frames"}
    dump_stride = float(np.median(np.diff(timesteps))) if len(timesteps) > 1 else 1.0
    candidate_lags = sorted(set([1, 2, 5, 10, max(1, n_frames // 20), max(1, n_frames // 10), max(1, n_frames // 4)]))
    candidate_lags = [lag for lag in candidate_lags if lag < n_frames]

    lag_summaries = []
    max_r = 0.0
    dr_by_lag: Dict[int, np.ndarray] = {}
    for lag in candidate_lags:
        disp = com_traj[lag:] - com_traj[: n_frames - lag]
        r = np.linalg.norm(disp.reshape(-1, 3), axis=1)
        dr_by_lag[lag] = r
        if r.size:
            max_r = max(max_r, float(np.max(r)))
        r2 = float(np.mean(r**2)) if r.size else None
        r4 = float(np.mean(r**4)) if r.size else None
        alpha2 = (3.0 * r4 / (5.0 * r2 * r2) - 1.0) if r2 and r4 and r2 > 0 else None
        lag_summaries.append(
            {
                "lag_frames": int(lag),
                "time_ps": float(lag * dump_stride * dt_fs / 1000.0),
                "n_displacements": int(r.size),
                "mean_displacement_A": float(np.mean(r)) if r.size else None,
                "p50_displacement_A": float(np.quantile(r, 0.50)) if r.size else None,
                "p90_displacement_A": float(np.quantile(r, 0.90)) if r.size else None,
                "non_gaussian_parameter_alpha2": float(alpha2) if alpha2 is not None else None,
            }
        )

    bins = np.arange(0.0, max(max_r + bin_width_A, bin_width_A), bin_width_A)
    histograms = []
    for lag in candidate_lags[: min(5, len(candidate_lags))]:
        r = dr_by_lag[lag]
        counts, edges = np.histogram(r, bins=bins, density=False)
        prob = counts.astype(float) / max(float(np.sum(counts)), 1.0)
        histograms.append(
            {
                "lag_frames": int(lag),
                "time_ps": float(lag * dump_stride * dt_fs / 1000.0),
                "r_bin_center_A": ((edges[:-1] + edges[1:]) / 2.0).tolist(),
                "probability": prob.tolist(),
            }
        )

    peak = None
    valid = [x for x in lag_summaries if x["non_gaussian_parameter_alpha2"] is not None]
    if valid:
        peak = max(valid, key=lambda x: abs(float(x["non_gaussian_parameter_alpha2"])))
    return {
        "definition": "Self part of the van Hove displacement distribution Gs(r,t) plus non-Gaussian parameter alpha2 for guest COM motion.",
        "status": "ok",
        "lag_summaries": lag_summaries,
        "histograms": histograms,
        "peak_abs_alpha2": peak,
        "interpretation_hint": "alpha2 near 0 suggests roughly Brownian/Gaussian motion; positive peaks suggest heterogeneous hopping or intermittent caging.",
    }


def _extract_element(label: str) -> Optional[str]:
    m = re.match(r"\s*([A-Z][a-z]?)", label or "")
    return m.group(1) if m else None


def _load_chemistry_unit_map(chemistry_summary: Optional[Path]) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    if not chemistry_summary or not chemistry_summary.exists():
        return {}, {"status": "not_provided"}
    try:
        blob = json.loads(chemistry_summary.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, {"status": "read_failed", "error": str(exc), "path": str(chemistry_summary)}

    atom_map: Dict[int, Dict[str, Any]] = {}
    for structure in blob.get("structures", []) or []:
        for unit_key, kind in (("nodes", "node"), ("linkers", "linker")):
            for unit in structure.get(unit_key, []) or []:
                unit_id = f"{kind}_{unit.get('index')}"
                rec = {
                    "unit_type": kind,
                    "unit_id": unit_id,
                    "formula": unit.get("formula"),
                    "functional_tags": unit.get("functional_tags") or [],
                    "smiles": unit.get("smiles"),
                }
                for atom_id in unit.get("source_atom_indices", []) or []:
                    try:
                        atom_map[int(atom_id)] = rec
                    except Exception:
                        continue
    return atom_map, {
        "status": "ok" if atom_map else "no_source_atom_indices",
        "path": str(chemistry_summary),
        "n_mapped_atoms": int(len(atom_map)),
    }


def _fallback_unit_for_type(atom_type: int, type_labels: Dict[int, str]) -> Dict[str, Any]:
    label = type_labels.get(int(atom_type), str(atom_type))
    element = _extract_element(label)
    if element in METAL_ELEMENTS:
        return {
            "unit_type": "node",
            "unit_id": "fallback_metal_node",
            "formula": element,
            "functional_tags": ["metal node fallback"],
            "smiles": None,
        }
    return {
        "unit_type": "linker",
        "unit_id": "fallback_organic_linker",
        "formula": element or label,
        "functional_tags": ["organic/linker fallback"],
        "smiles": None,
    }


def _parse_lammps_thermo_table(log_path: Path) -> Dict[str, List[float]]:
    if not log_path.exists():
        return {}
    numeric_header = None
    rows: Dict[str, List[float]] = {}
    best_rows: Dict[str, List[float]] = {}

    def keep_best() -> None:
        nonlocal best_rows
        if rows and len(next(iter(rows.values()), [])) > len(next(iter(best_rows.values()), [])):
            best_rows = {key: list(vals) for key, vals in rows.items()}

    for raw in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        cols = line.split()
        if cols and cols[0].lower() == "step" and len(cols) > 1:
            keep_best()
            numeric_header = cols
            rows = {key: [] for key in numeric_header}
            continue
        if numeric_header is None:
            continue
        if len(cols) < len(numeric_header):
            if rows.get("Step"):
                keep_best()
                numeric_header = None
            continue
        vals = []
        ok = True
        for token in cols[: len(numeric_header)]:
            x = _clean_float(token)
            if x is None:
                ok = False
                break
            vals.append(x)
        if ok:
            for key, val in zip(numeric_header, vals):
                rows.setdefault(key, []).append(val)
    keep_best()
    return best_rows


def _scalar_autocorrelation(series: np.ndarray, max_lag: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    if series.size < 4:
        return np.array([], dtype=int), np.array([], dtype=float)
    x = series.astype(float) - float(np.mean(series))
    denom = float(np.mean(x * x))
    if denom <= 0.0:
        return np.array([], dtype=int), np.array([], dtype=float)
    max_lag = min(max_lag if max_lag is not None else series.size // 2, series.size - 1)
    lags = np.arange(max_lag + 1, dtype=int)
    corr = np.zeros(len(lags), dtype=float)
    for out_i, lag in enumerate(lags):
        corr[out_i] = float(np.mean(x[: series.size - lag] * x[lag:]) / denom)
    return lags, corr


def _energy_autocorrelation(work_dir: Path) -> Dict[str, Any]:
    log_path = work_dir / "log.lammps"
    thermo = _parse_lammps_thermo_table(log_path)
    if not thermo:
        return {"status": "no_thermo_table_found", "log_path": str(log_path)}
    key = None
    for candidate in ("TotEng", "etotal", "Etot", "PotEng", "pe", "E_pair", "evdwl", "ecoul"):
        for existing in thermo:
            if existing.lower() == candidate.lower() and len(thermo[existing]) >= 4:
                key = existing
                break
        if key:
            break
    if not key:
        return {
            "status": "no_energy_column_found",
            "available_columns": sorted(thermo.keys()),
            "log_path": str(log_path),
        }
    series = np.array(thermo[key], dtype=float)
    lags, corr = _scalar_autocorrelation(series, max_lag=max(1, min(500, series.size // 2)))
    step = np.array(thermo.get("Step", list(range(series.size))), dtype=float)
    step_stride = float(np.median(np.diff(step))) if step.size > 1 else 1.0
    first_zero = None
    for lag, c in zip(lags[1:], corr[1:]):
        if c <= 0:
            first_zero = float(lag * step_stride)
            break
    return {
        "definition": "Autocorrelation of a scalar LAMMPS thermo energy column, useful for checking energy relaxation/persistence during diffusion.",
        "status": "ok" if corr.size else "insufficient_or_constant_series",
        "log_path": str(log_path),
        "energy_column": key,
        "n_points": int(series.size),
        "mean": float(np.mean(series)),
        "std": float(np.std(series)),
        "lag_steps": (lags * step_stride).tolist(),
        "autocorrelation": corr.tolist(),
        "first_zero_crossing_steps": first_zero,
    }


def _rdf_contact_residence(
    frames: List[Dict[str, Any]],
    guest_types: Sequence[int],
    masses: Dict[int, float],
    type_labels: Dict[int, str],
    r_max_A: float,
    bin_width_A: float,
    contact_cutoff_A: float,
    dt_fs: float,
    chemistry_unit_map: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    bins = np.arange(0.0, r_max_A + bin_width_A, bin_width_A)
    hist = np.zeros(len(bins) - 1, dtype=float)
    contact_weights: Dict[str, float] = {}
    unit_contact_weights: Dict[str, float] = {}
    unit_type_weights: Dict[str, float] = {}
    site_contact_weights: Dict[str, float] = {}
    nearest_distances = []
    states: Dict[int, List[Optional[Tuple[int, int]]]] = {}
    unit_states: Dict[int, List[Optional[str]]] = {}
    edge_counts: Dict[Tuple[str, str], int] = {}
    timesteps = []
    chemistry_unit_map = chemistry_unit_map or {}

    guest_types_set = set(int(x) for x in guest_types)
    for frame in frames:
        timesteps.append(int(frame["timestep"]))
        com, mol_ids = _guest_com_for_frame(frame, guest_types, masses)
        if len(com) == 0:
            continue
        fw_mask = np.array([int(t) not in guest_types_set for t in frame["types"]], dtype=bool)
        fw_pos = _wrap_positions(frame["positions"][fw_mask], frame["cell"], frame["origin"])
        fw_types = frame["types"][fw_mask]
        fw_ids = frame["ids"][fw_mask]
        com_wrapped = _wrap_positions(com, frame["cell"], frame["origin"])
        disp = _minimum_image_displacements(com_wrapped, fw_pos, frame["cell"])
        dist = np.linalg.norm(disp, axis=2)
        hist += np.histogram(dist.ravel(), bins=bins)[0]

        nearest_idx = np.argmin(dist, axis=1)
        nearest_d = dist[np.arange(len(com)), nearest_idx]
        nearest_distances.extend(float(x) for x in nearest_d)
        for mol_i, mol_id in enumerate(mol_ids):
            d = float(nearest_d[mol_i])
            fw_i = int(nearest_idx[mol_i])
            state = None
            if d <= contact_cutoff_A:
                fw_type = int(fw_types[fw_i])
                fw_id = int(fw_ids[fw_i])
                label = type_labels.get(fw_type, str(fw_type))
                contact_weights[label] = contact_weights.get(label, 0.0) + 1.0
                unit = chemistry_unit_map.get(fw_id) or _fallback_unit_for_type(fw_type, type_labels)
                unit_type = unit.get("unit_type") or "unknown"
                unit_id = unit.get("unit_id") or f"{unit_type}_unknown"
                unit_label = f"{unit_type}:{unit_id}"
                site_label = f"{unit_label}|atom:{fw_id}:{label}"
                unit_contact_weights[unit_label] = unit_contact_weights.get(unit_label, 0.0) + 1.0
                unit_type_weights[unit_type] = unit_type_weights.get(unit_type, 0.0) + 1.0
                site_contact_weights[site_label] = site_contact_weights.get(site_label, 0.0) + 1.0
                state = (fw_id, fw_type)
                unit_state = site_label
            else:
                unit_state = None
            states.setdefault(int(mol_id), []).append(state)
            unit_states.setdefault(int(mol_id), []).append(unit_state)

    total_contact = float(sum(contact_weights.values()))
    contact_fraction = {
        key: float(val / total_contact)
        for key, val in sorted(contact_weights.items(), key=lambda item: item[1], reverse=True)
    } if total_contact > 0 else {}
    total_unit_contact = float(sum(unit_contact_weights.values()))
    unit_contact_fraction = {
        key: float(val / total_unit_contact)
        for key, val in sorted(unit_contact_weights.items(), key=lambda item: item[1], reverse=True)
    } if total_unit_contact > 0 else {}
    unit_type_fraction = {
        key: float(val / total_unit_contact)
        for key, val in sorted(unit_type_weights.items(), key=lambda item: item[1], reverse=True)
    } if total_unit_contact > 0 else {}
    site_contact_fraction = {
        key: float(val / total_unit_contact)
        for key, val in sorted(site_contact_weights.items(), key=lambda item: item[1], reverse=True)
    } if total_unit_contact > 0 else {}

    shell_vol = (4.0 / 3.0) * math.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
    rdf_density_like = hist / np.maximum(shell_vol, 1e-12)
    if np.max(rdf_density_like) > 0:
        rdf_scaled = rdf_density_like / np.max(rdf_density_like)
    else:
        rdf_scaled = rdf_density_like

    residence_segments = []
    hop_count = 0
    frame_dt_ps = None
    if len(timesteps) > 1:
        frame_dt_ps = float(np.median(np.diff(timesteps)) * dt_fs / 1000.0)

    for mol_id, seq in states.items():
        prev = None
        length = 0
        for state in seq + [None]:
            if state == prev:
                length += 1
                continue
            if prev is not None and length > 0:
                residence_segments.append(length)
            if prev is not None and state is not None and state != prev:
                hop_count += 1
            prev = state
            length = 1

    for mol_id, seq in unit_states.items():
        prev_unit = None
        for unit_state in seq:
            if prev_unit is not None and unit_state is not None and unit_state != prev_unit:
                edge_counts[(prev_unit, unit_state)] = edge_counts.get((prev_unit, unit_state), 0) + 1
            if unit_state is not None:
                prev_unit = unit_state

    residence_times_ps = [
        float(n * frame_dt_ps) for n in residence_segments
    ] if frame_dt_ps is not None else []

    nearest = np.array(nearest_distances, dtype=float)
    residence = np.array(residence_times_ps, dtype=float)
    return {
        "rdf_contact": {
            "definition": "Guest COM to framework atom distance distribution; RDF is scaled by shell volume and normalized to max=1.",
            "r_bin_center_A": ((bins[:-1] + bins[1:]) / 2.0).tolist(),
            "raw_pair_counts": hist.astype(int).tolist(),
            "rdf_scaled": rdf_scaled.tolist(),
            "nearest_distance_A": {
                "mean": float(np.mean(nearest)) if nearest.size else None,
                "p05": float(np.quantile(nearest, 0.05)) if nearest.size else None,
                "p50": float(np.quantile(nearest, 0.50)) if nearest.size else None,
                "p95": float(np.quantile(nearest, 0.95)) if nearest.size else None,
            },
            "contact_cutoff_A": float(contact_cutoff_A),
            "contact_type_fraction": contact_fraction,
        },
        "residence_hopping": {
            "definition": "Residence is consecutive frames where a guest COM remains nearest to the same framework atom within the cutoff.",
            "site_definition": "nearest framework atom within contact_cutoff_A",
            "frame_dt_ps": frame_dt_ps,
            "n_residence_segments": int(len(residence_segments)),
            "n_hops": int(hop_count),
            "mean_residence_time_ps": float(np.mean(residence)) if residence.size else None,
            "median_residence_time_ps": float(np.median(residence)) if residence.size else None,
            "p90_residence_time_ps": float(np.quantile(residence, 0.90)) if residence.size else None,
            "hop_rate_per_ns": (
                float(hop_count / ((len(timesteps) - 1) * frame_dt_ps / 1000.0))
                if frame_dt_ps and len(timesteps) > 1 else None
            ),
        },
        "chemistry_unit_contact": {
            "definition": "Guest contact fractions grouped by MOF chemistry unit when mofstructure source-atom mapping is available; otherwise metal/nonmetal fallback is used.",
            "unit_contact_fraction": unit_contact_fraction,
            "unit_type_fraction": unit_type_fraction,
            "mapping_mode": "mofstructure_source_atom_indices" if chemistry_unit_map else "fallback_metal_node_vs_linker",
        },
        "pore_network_hopping_graph": {
            "definition": "Directed graph of guest nearest-site transitions between chemistry-aware framework sites; edges count observed site-to-site hops.",
            "site_visit_fraction": site_contact_fraction,
            "unit_visit_fraction": unit_contact_fraction,
            "edges": [
                {"source": src, "target": dst, "count": int(count)}
                for (src, dst), count in sorted(edge_counts.items(), key=lambda item: item[1], reverse=True)
            ],
            "n_edges": int(len(edge_counts)),
        },
    }


def infer_mof_guest_from_dir(work_dir: Path) -> Dict[str, Optional[str]]:
    name = work_dir.name
    m = re.match(r"(.+?)_([A-Za-z0-9]+)_(diffusivity|mean_squared_displacement)(?:_(.*))?$", name)
    if m:
        mof, guest, prop, rest = m.groups()
        return {"mof": mof, "guest": guest, "property": prop, "condition": rest}
    return {"mof": None, "guest": None, "property": None, "condition": None}


def run_lammps_trajectory_analysis(
    work_dir: Path,
    output_dir: Optional[Path] = None,
    max_frames: int = 400,
    r_max_A: float = 12.0,
    bin_width_A: float = 0.1,
    contact_cutoff_A: float = 4.0,
    chemistry_summary: Optional[Path] = None,
) -> Dict[str, Any]:
    work_dir = Path(work_dir)
    traj_path = work_dir / "traj.lammpstrj"
    if not traj_path.exists():
        raise FileNotFoundError(f"traj.lammpstrj not found: {traj_path}")
    system_in = work_dir / "system.in"
    system_data = work_dir / "system.data"

    guest_types = _parse_guest_types(system_in)
    if not guest_types:
        raise RuntimeError(f"Could not infer guest atom types from {system_in}")
    masses, type_labels = _parse_masses_and_type_labels(system_data)
    dt_fs = _parse_dt_fs(system_in)
    n_total_frames = _count_frames(traj_path)
    frame_stride = max(1, math.ceil(n_total_frames / max(1, max_frames)))
    frames = list(_iter_lammps_frames(traj_path, frame_stride=frame_stride, max_frames=max_frames))
    if len(frames) < 3:
        raise RuntimeError(f"Need at least 3 trajectory frames, got {len(frames)}")

    timesteps: List[int] = []
    com_frames = []
    mol_ids_ref = None
    for frame in frames:
        com, mol_ids = _guest_com_for_frame(frame, guest_types, masses)
        if len(com) == 0:
            continue
        if mol_ids_ref is None:
            mol_ids_ref = mol_ids
        elif mol_ids != mol_ids_ref:
            common = [m for m in mol_ids_ref if m in set(mol_ids)]
            idx = [mol_ids.index(m) for m in common]
            com = com[idx]
            mol_ids_ref = common
        timesteps.append(int(frame["timestep"]))
        com_frames.append(com)
    if len(com_frames) < 3:
        raise RuntimeError("Not enough guest COM frames after parsing.")
    com_traj = np.stack(com_frames, axis=0)

    chemistry_unit_map, chemistry_map_status = _load_chemistry_unit_map(Path(chemistry_summary) if chemistry_summary else None)
    anisotropic = _component_msd_analysis(timesteps, com_traj, dt_fs=dt_fs)
    van_hove = _van_hove_non_gaussian(timesteps, com_traj, dt_fs=dt_fs)
    vacf = _velocity_autocorrelation(timesteps, com_traj, dt_fs=dt_fs)
    energy_acf = _energy_autocorrelation(work_dir)
    contact = _rdf_contact_residence(
        frames=frames,
        guest_types=guest_types,
        masses=masses,
        type_labels=type_labels,
        r_max_A=r_max_A,
        bin_width_A=bin_width_A,
        contact_cutoff_A=contact_cutoff_A,
        dt_fs=dt_fs,
        chemistry_unit_map=chemistry_unit_map,
    )

    meta = infer_mof_guest_from_dir(work_dir)
    result = {
        "method": "lammps_trajectory_analysis",
        "work_dir": str(work_dir),
        "trajectory_file": str(traj_path),
        "mof": meta.get("mof"),
        "guest": meta.get("guest"),
        "property": meta.get("property"),
        "condition": meta.get("condition"),
        "settings": {
            "n_total_frames": int(n_total_frames),
            "n_frames_used": int(len(frames)),
            "frame_stride": int(frame_stride),
            "dt_fs_from_system_in": float(dt_fs),
            "guest_types": guest_types,
            "guest_type_labels": {str(t): type_labels.get(t, str(t)) for t in guest_types},
            "r_max_A": float(r_max_A),
            "bin_width_A": float(bin_width_A),
            "contact_cutoff_A": float(contact_cutoff_A),
            "chemistry_summary": str(chemistry_summary) if chemistry_summary else None,
            "chemistry_unit_map": chemistry_map_status,
        },
        "n_guest_molecules": int(com_traj.shape[1]),
        "anisotropic_diffusion": anisotropic,
        "van_hove_non_gaussian": van_hove,
        "velocity_autocorrelation": vacf,
        "energy_autocorrelation": energy_acf,
        "rdf_contact": contact["rdf_contact"],
        "residence_hopping": contact["residence_hopping"],
        "chemistry_unit_contact": contact["chemistry_unit_contact"],
        "pore_network_hopping_graph": contact["pore_network_hopping_graph"],
        "limitations": [
            "RDF is a scaled shell-volume-normalized contact distribution, not a rigorously density-normalized bulk g(r).",
            "Chemistry-unit contacts use mofstructure source atom mapping when supplied; otherwise the node/linker assignment is a metal/nonmetal fallback.",
            "Directional diffusion uses the trajectory coordinate axes; crystallographic channel axes may require basis transformation.",
        ],
    }

    output_dir = output_dir or work_dir / "lammps_trajectory_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{work_dir.name}_lammps_trajectory_summary.json"
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    msd_path = output_dir / f"{work_dir.name}_anisotropic_msd.csv"
    with msd_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_ps", "msd_x_A2", "msd_y_A2", "msd_z_A2", "msd_total_A2"])
        for row in zip(
            anisotropic["time_ps"],
            anisotropic["msd_x_A2"],
            anisotropic["msd_y_A2"],
            anisotropic["msd_z_A2"],
            anisotropic["msd_total_A2"],
        ):
            writer.writerow(row)

    rdf_path = output_dir / f"{work_dir.name}_guest_host_rdf.csv"
    with rdf_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["r_A", "raw_pair_counts", "rdf_scaled"])
        for row in zip(
            result["rdf_contact"]["r_bin_center_A"],
            result["rdf_contact"]["raw_pair_counts"],
            result["rdf_contact"]["rdf_scaled"],
        ):
            writer.writerow(row)

    van_hove_path = output_dir / f"{work_dir.name}_van_hove_ngp.csv"
    with van_hove_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["lag_frames", "time_ps", "n_displacements", "mean_displacement_A", "p50_displacement_A", "p90_displacement_A", "non_gaussian_parameter_alpha2"])
        for row in result["van_hove_non_gaussian"].get("lag_summaries", []):
            writer.writerow([
                row.get("lag_frames"),
                row.get("time_ps"),
                row.get("n_displacements"),
                row.get("mean_displacement_A"),
                row.get("p50_displacement_A"),
                row.get("p90_displacement_A"),
                row.get("non_gaussian_parameter_alpha2"),
            ])

    vacf_path = output_dir / f"{work_dir.name}_vacf.csv"
    with vacf_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_ps", "vacf_normalized"])
        for row in zip(
            result["velocity_autocorrelation"].get("time_ps", []),
            result["velocity_autocorrelation"].get("vacf_normalized", []),
        ):
            writer.writerow(row)

    energy_acf_path = output_dir / f"{work_dir.name}_energy_autocorrelation.csv"
    with energy_acf_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["lag_steps", "autocorrelation"])
        for row in zip(
            result["energy_autocorrelation"].get("lag_steps", []),
            result["energy_autocorrelation"].get("autocorrelation", []),
        ):
            writer.writerow(row)

    result["summary_json"] = str(summary_path)
    result["anisotropic_msd_csv"] = str(msd_path)
    result["rdf_csv"] = str(rdf_path)
    result["van_hove_ngp_csv"] = str(van_hove_path)
    result["vacf_csv"] = str(vacf_path)
    result["energy_autocorrelation_csv"] = str(energy_acf_path)
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze LAMMPS guest trajectory for diffusion/contact/residence.")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-frames", type=int, default=400)
    parser.add_argument("--r-max", type=float, default=12.0)
    parser.add_argument("--bin-width", type=float, default=0.1)
    parser.add_argument("--contact-cutoff", type=float, default=4.0)
    parser.add_argument("--chemistry-summary", default=None)
    args = parser.parse_args(argv)

    result = run_lammps_trajectory_analysis(
        work_dir=Path(args.work_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        max_frames=args.max_frames,
        r_max_A=args.r_max,
        bin_width_A=args.bin_width,
        contact_cutoff_A=args.contact_cutoff,
        chemistry_summary=Path(args.chemistry_summary) if args.chemistry_summary else None,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
