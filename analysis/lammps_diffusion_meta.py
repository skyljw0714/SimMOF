from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import linregress


R_GAS_J_MOL_K = 8.31446261815324


def _clean_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def _temperature_from_text(*values: Any) -> Optional[float]:
    text = " ".join(str(v or "") for v in values)
    m = re.search(r"(?<!\d)(\d{2,4}(?:\.\d+)?)\s*K(?=$|[^A-Za-z0-9])", text, flags=re.IGNORECASE)
    if not m:
        return None
    temp = _clean_float(m.group(1))
    if temp is None or temp <= 0:
        return None
    return temp


def _iter_summary_files(paths: Sequence[Path]) -> Iterable[Path]:
    for path in paths:
        path = Path(path)
        if path.is_file() and path.suffix.lower() == ".json":
            yield path
        elif path.is_dir():
            yield from path.rglob("*_lammps_trajectory_summary.json")
            yield from path.rglob("combined_lammps_trajectory_analysis_summary.json")


def _flatten_records(blob: Any) -> List[Dict[str, Any]]:
    if isinstance(blob, list):
        return [x for x in blob if isinstance(x, dict)]
    if isinstance(blob, dict) and isinstance(blob.get("rows"), list):
        return [x for x in blob["rows"] if isinstance(x, dict)]
    if isinstance(blob, dict) and blob.get("method") == "lammps_trajectory_analysis":
        return [blob]
    records: List[Dict[str, Any]] = []
    if isinstance(blob, dict):
        for value in blob.values():
            records.extend(_flatten_records(value))
    return records


def _load_records(inputs: Sequence[Path]) -> List[Dict[str, Any]]:
    records = []
    seen = set()
    for path in _iter_summary_files(inputs):
        if path in seen:
            continue
        seen.add(path)
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for rec in _flatten_records(blob):
            rec = dict(rec)
            rec.setdefault("source_summary_json", str(path))
            records.append(rec)
    return records


def _record_row(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    fit = ((rec.get("anisotropic_diffusion") or {}).get("fit_total") or {})
    d_total = _clean_float(fit.get("D_m2_s"))
    if d_total is None:
        d_total = _clean_float(rec.get("D_total_m2_s"))
    if d_total is None or d_total <= 0:
        return None
    mof = rec.get("mof")
    guest = rec.get("guest")
    condition = rec.get("condition")
    temp = _temperature_from_text(condition, rec.get("work_dir"), rec.get("source_summary_json"))
    nearest = (((rec.get("rdf_contact") or {}).get("nearest_distance_A") or {}).get("p50"))
    if nearest is None:
        nearest = rec.get("nearest_p50_A")
    residence = rec.get("residence_hopping") or {}
    return {
        "mof": mof or "unknown_mof",
        "guest": guest or "unknown_guest",
        "condition": condition,
        "temperature_K": temp,
        "D_m2_s": d_total,
        "anisotropy": fit.get("anisotropy_ratio_max_min") if fit else rec.get("anisotropy_ratio"),
        "nearest_p50_A": nearest,
        "hop_rate_per_ns": residence.get("hop_rate_per_ns") if residence else rec.get("hop_rate_per_ns"),
        "source_summary_json": rec.get("summary_json") or rec.get("source_summary_json"),
        "work_dir": rec.get("work_dir"),
    }


def _stats(values: List[float]) -> Dict[str, Any]:
    arr = np.array([x for x in values if x is not None and math.isfinite(float(x))], dtype=float)
    if arr.size == 0:
        return {"n": 0}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return {
        "n": int(arr.size),
        "mean": mean,
        "std": std,
        "cv": float(std / mean) if mean != 0 else None,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _replicate_consistency(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["mof"]), str(row["guest"]), str(row.get("condition") or "unspecified"))
        grouped[key].append(row)

    out = []
    for (mof, guest, condition), items in sorted(grouped.items()):
        out.append(
            {
                "mof": mof,
                "guest": guest,
                "condition": condition,
                "temperature_K": items[0].get("temperature_K"),
                "n_replicates": len(items),
                "D_m2_s": _stats([x["D_m2_s"] for x in items]),
                "anisotropy": _stats([_clean_float(x.get("anisotropy")) for x in items]),
                "nearest_p50_A": _stats([_clean_float(x.get("nearest_p50_A")) for x in items]),
                "hop_rate_per_ns": _stats([_clean_float(x.get("hop_rate_per_ns")) for x in items]),
                "source_summary_json": [x.get("source_summary_json") for x in items],
            }
        )
    return out


def _activation_barriers(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_mof_guest_temp: Dict[Tuple[str, str, float], List[float]] = defaultdict(list)
    for row in rows:
        temp = row.get("temperature_K")
        d_total = row.get("D_m2_s")
        if temp is None or d_total is None or d_total <= 0:
            continue
        by_mof_guest_temp[(str(row["mof"]), str(row["guest"]), float(temp))].append(float(d_total))

    grouped: Dict[Tuple[str, str], List[Tuple[float, float, float, int]]] = defaultdict(list)
    for (mof, guest, temp), vals in by_mof_guest_temp.items():
        ln_vals = np.log(np.array(vals, dtype=float))
        grouped[(mof, guest)].append((temp, float(np.mean(ln_vals)), float(np.std(ln_vals, ddof=1)) if len(vals) > 1 else 0.0, len(vals)))

    out = []
    for (mof, guest), points in sorted(grouped.items()):
        points = sorted(points, key=lambda x: x[0])
        if len(points) < 3:
            out.append(
                {
                    "mof": mof,
                    "guest": guest,
                    "status": "insufficient_temperatures",
                    "n_temperatures": len(points),
                    "points": [
                        {"temperature_K": t, "mean_lnD": ln_d, "std_lnD": std_ln_d, "n": n}
                        for t, ln_d, std_ln_d, n in points
                    ],
                }
            )
            continue
        x = np.array([1.0 / t for t, _, _, _ in points], dtype=float)
        y = np.array([ln_d for _, ln_d, _, _ in points], dtype=float)
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        ea_kj_mol = float(-slope * R_GAS_J_MOL_K / 1000.0)
        out.append(
            {
                "mof": mof,
                "guest": guest,
                "status": "ok",
                "model": "ln(D) = ln(D0) - Ea/(R*T)",
                "Ea_kJ_mol": ea_kj_mol,
                "lnD0_intercept": float(intercept),
                "r2": float(r_value**2),
                "p_value": float(p_value),
                "std_err_slope": float(std_err),
                "n_temperatures": len(points),
                "points": [
                    {"temperature_K": t, "mean_lnD": ln_d, "std_lnD": std_ln_d, "n": n}
                    for t, ln_d, std_ln_d, n in points
                ],
                "caution": "Use only as an apparent activation barrier; unreliable if replicate D values vary strongly or trajectories are not diffusive.",
            }
        )
    return out


def run_lammps_diffusion_meta_analysis(inputs: Sequence[Path], output_dir: Path) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _load_records([Path(x) for x in inputs])
    rows = [row for rec in records if (row := _record_row(rec)) is not None]
    replicate = _replicate_consistency(rows)
    barriers = _activation_barriers(rows)

    result = {
        "method": "lammps_diffusion_meta_analysis",
        "n_input_records": len(records),
        "n_valid_diffusion_records": len(rows),
        "replicate_consistency": replicate,
        "activation_barrier": barriers,
        "notes": [
            "Activation barrier requires at least three temperatures for the same MOF/guest.",
            "The fit uses the mean of ln(D) at each temperature, so replicate scatter is retained as uncertainty context.",
        ],
    }

    json_path = output_dir / "lammps_diffusion_meta_analysis_summary.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    rep_csv = output_dir / "lammps_diffusion_replicate_consistency.csv"
    with rep_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mof", "guest", "condition", "temperature_K", "n_replicates", "D_mean_m2_s", "D_std_m2_s", "D_cv"])
        for row in replicate:
            d = row.get("D_m2_s") or {}
            writer.writerow([row.get("mof"), row.get("guest"), row.get("condition"), row.get("temperature_K"), row.get("n_replicates"), d.get("mean"), d.get("std"), d.get("cv")])

    barrier_csv = output_dir / "lammps_diffusion_activation_barrier.csv"
    with barrier_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mof", "guest", "status", "n_temperatures", "Ea_kJ_mol", "r2", "p_value"])
        for row in barriers:
            writer.writerow([row.get("mof"), row.get("guest"), row.get("status"), row.get("n_temperatures"), row.get("Ea_kJ_mol"), row.get("r2"), row.get("p_value")])

    result["summary_json"] = str(json_path)
    result["replicate_consistency_csv"] = str(rep_csv)
    result["activation_barrier_csv"] = str(barrier_csv)
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Meta-analysis for LAMMPS diffusion replicates and Arrhenius activation barriers.")
    parser.add_argument("inputs", nargs="+", help="Summary JSON files or directories containing LAMMPS trajectory summaries.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    result = run_lammps_diffusion_meta_analysis([Path(x) for x in args.inputs], Path(args.output_dir))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
