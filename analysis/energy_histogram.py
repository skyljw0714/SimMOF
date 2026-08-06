from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


K_TO_KJ_MOL = 0.00831446261815324
DEFAULT_STRONG_THRESHOLDS_KJ_MOL = [-30.0, -25.0, -20.0, -15.0, -10.0, -5.0]


def _clean_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def _parse_energy_histogram_settings(simulation_input: Path) -> Dict[str, Any]:
    settings: Dict[str, Any] = {}
    if not simulation_input.exists():
        return settings

    for raw in simulation_input.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.strip().split()
        if len(parts) < 2:
            continue
        key = parts[0]
        value = parts[1]
        if key == "ComputeEnergyHistogram":
            settings[key] = value
        elif key in {"WriteEnergyHistogramEvery", "EnergyHistogramSize"}:
            parsed = _clean_float(value)
            settings[key] = int(parsed) if parsed is not None else value
        elif key in {"EnergyHistogramLowerLimit", "EnergyHistogramUpperLimit"}:
            settings[key] = _clean_float(value)

    lower = settings.get("EnergyHistogramLowerLimit")
    upper = settings.get("EnergyHistogramUpperLimit")
    size = settings.get("EnergyHistogramSize")
    if isinstance(lower, (int, float)) and isinstance(upper, (int, float)) and isinstance(size, int) and size > 0:
        settings["bin_width_K"] = float(abs(upper - lower) / size)
        settings["bin_width_kJ_mol"] = settings["bin_width_K"] * K_TO_KJ_MOL

    return settings


def _parse_unit_cells(simulation_input: Path) -> Optional[Tuple[int, int, int]]:
    if not simulation_input.exists():
        return None
    for raw in simulation_input.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.strip().split()
        if len(parts) >= 4 and parts[0] == "UnitCells":
            try:
                return int(parts[1]), int(parts[2]), int(parts[3])
            except ValueError:
                return None
    return None


def _parse_average_loading_molecules_per_unit_cell(work_dir: Path) -> Dict[str, Any]:
    output_files = sorted((work_dir / "Output" / "System_0").glob("*.data"))
    for path in output_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        m = re.search(
            r"Average loading absolute\s+\[molecules/unit cell\]\s+([-+0-9.eE]+)\s+\+/-\s+([-+0-9.eE]+)",
            text,
        )
        if not m:
            continue
        value = _clean_float(m.group(1))
        error = _clean_float(m.group(2))
        if value is None:
            continue
        return {
            "average_loading_absolute_molecules_per_unit_cell": value,
            "average_loading_absolute_error_molecules_per_unit_cell": error,
            "raspa_output_file": str(path),
        }
    return {}


def _energy_kind_from_path(path: Path) -> str:
    m = re.match(r"Histogram_(.+)_Energy_", path.name)
    if not m:
        return "unknown"
    return m.group(1).lower()


def _histogram_index_from_path(path: Path) -> float:
    m = re.match(r"Histogram_.+_Energy_([-+0-9.eE]+)", path.name)
    if not m:
        return -math.inf
    try:
        return float(m.group(1))
    except ValueError:
        return -math.inf


def _read_histogram_file(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    rows: List[Tuple[float, float]] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.strip().split()
        if len(parts) < 2:
            continue
        x = _clean_float(parts[0])
        y = _clean_float(parts[1])
        if x is None or y is None or y < 0:
            continue
        rows.append((x, y))

    if not rows:
        return np.array([], dtype=float), np.array([], dtype=float)

    rows.sort(key=lambda item: item[0])
    return np.array([r[0] for r in rows], dtype=float), np.array([r[1] for r in rows], dtype=float)


def _infer_bin_width(energy_K: np.ndarray, settings: Dict[str, Any]) -> Optional[float]:
    configured = settings.get("bin_width_K")
    if isinstance(configured, (int, float)) and configured > 0:
        return float(configured)

    diffs = np.diff(np.unique(energy_K))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return None
    return float(np.median(diffs))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: Sequence[float]) -> Dict[str, float]:
    if len(values) == 0 or float(np.sum(weights)) <= 0:
        return {f"p{int(q * 100):02d}": None for q in quantiles}

    order = np.argsort(values)
    xs = values[order]
    ws = weights[order]
    cdf = np.cumsum(ws)
    cdf = cdf / cdf[-1]

    out: Dict[str, float] = {}
    for q in quantiles:
        out[f"p{int(q * 100):02d}"] = float(np.interp(q, cdf, xs))
    return out


def _summarize_distribution(
    energy_K: np.ndarray,
    density: np.ndarray,
    settings: Dict[str, Any],
    strong_thresholds_kJ_mol: Sequence[float],
) -> Dict[str, Any]:
    if len(energy_K) == 0:
        return {
            "status": "empty_histogram",
            "n_nonzero_bins": 0,
        }

    bin_width_K = _infer_bin_width(energy_K, settings)
    if bin_width_K is None or bin_width_K <= 0:
        weights = density.astype(float)
    else:
        weights = density.astype(float) * float(bin_width_K)

    total_weight = float(np.sum(weights))
    if total_weight <= 0:
        return {
            "status": "zero_weight_histogram",
            "n_nonzero_bins": int(len(energy_K)),
            "bin_width_K": bin_width_K,
        }

    probability = weights / total_weight
    energy_kJ_mol = energy_K * K_TO_KJ_MOL
    mean = float(np.sum(energy_kJ_mol * probability))
    variance = float(np.sum(((energy_kJ_mol - mean) ** 2) * probability))
    std = math.sqrt(max(variance, 0.0))
    mode_idx = int(np.argmax(density))
    quantiles = _weighted_quantile(energy_kJ_mol, probability, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])

    fractions = {
        f"fraction_energy_le_{abs(float(thr)):g}_kJ_mol": float(np.sum(probability[energy_kJ_mol <= float(thr)]))
        for thr in sorted(strong_thresholds_kJ_mol)
    }
    fractions["attractive_fraction_energy_lt_0"] = float(np.sum(probability[energy_kJ_mol < 0.0]))
    fractions["repulsive_fraction_energy_ge_0"] = float(np.sum(probability[energy_kJ_mol >= 0.0]))

    return {
        "status": "ok",
        "n_nonzero_bins": int(len(energy_K)),
        "bin_width_K": bin_width_K,
        "bin_width_kJ_mol": None if bin_width_K is None else float(bin_width_K * K_TO_KJ_MOL),
        "normalization_before_rescale": total_weight,
        "energy_unit_input": "K",
        "energy_unit": "kJ/mol",
        "mean_kJ_mol": mean,
        "std_kJ_mol": std,
        "mode_kJ_mol": float(energy_kJ_mol[mode_idx]),
        "mode_density": float(density[mode_idx]),
        "min_sampled_kJ_mol": float(np.min(energy_kJ_mol)),
        "max_sampled_kJ_mol": float(np.max(energy_kJ_mol)),
        "quantiles_kJ_mol": quantiles,
        "strong_binding_fractions": fractions,
    }


def _select_latest_histograms(hist_dir: Path) -> Dict[str, Path]:
    candidates = sorted(hist_dir.glob("Histogram_*_Energy_*.dat"))
    grouped: Dict[str, List[Path]] = {}
    for path in candidates:
        grouped.setdefault(_energy_kind_from_path(path), []).append(path)
    return {
        kind: sorted(paths, key=_histogram_index_from_path)[-1]
        for kind, paths in grouped.items()
        if paths
    }


def run_energy_histogram_analysis(
    work_dir: Path,
    mof: Optional[str] = None,
    output_dir: Optional[Path] = None,
    strong_thresholds_kJ_mol: Sequence[float] = DEFAULT_STRONG_THRESHOLDS_KJ_MOL,
    make_plot: bool = True,
) -> Dict[str, Any]:
    work_dir = Path(work_dir)
    if not work_dir.exists():
        raise FileNotFoundError(f"work_dir does not exist: {work_dir}")

    mof_name = mof or work_dir.name.split("_CH4_")[0].split("_CO2_")[0]
    hist_dir = work_dir / "EnergyHistograms" / "System_0"
    if not hist_dir.exists():
        raise FileNotFoundError(f"Energy histogram directory not found: {hist_dir}")

    output_dir = output_dir or work_dir / "energy_histogram_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = _parse_energy_histogram_settings(work_dir / "simulation.input")
    unit_cells = _parse_unit_cells(work_dir / "simulation.input")
    loading = _parse_average_loading_molecules_per_unit_cell(work_dir)
    if unit_cells:
        loading["unit_cells"] = list(unit_cells)
        loading["n_unit_cells"] = int(unit_cells[0] * unit_cells[1] * unit_cells[2])
    if loading.get("average_loading_absolute_molecules_per_unit_cell") is not None and loading.get("n_unit_cells"):
        loading["estimated_average_molecules_in_simulation_cell"] = (
            float(loading["average_loading_absolute_molecules_per_unit_cell"]) * int(loading["n_unit_cells"])
        )

    latest = _select_latest_histograms(hist_dir)
    if not latest:
        raise FileNotFoundError(f"No Histogram_*_Energy_*.dat files found in {hist_dir}")

    distributions: Dict[str, Any] = {}
    csv_rows: List[Dict[str, Any]] = []
    for kind, path in sorted(latest.items()):
        energy_K, density = _read_histogram_file(path)
        summary = _summarize_distribution(
            energy_K=energy_K,
            density=density,
            settings=settings,
            strong_thresholds_kJ_mol=strong_thresholds_kJ_mol,
        )
        summary["histogram_file"] = str(path)
        distributions[kind] = summary

        if len(energy_K):
            bin_width_K = summary.get("bin_width_K") or _infer_bin_width(energy_K, settings) or 1.0
            weights = density * float(bin_width_K)
            total = float(np.sum(weights))
            probability = weights / total if total > 0 else np.zeros_like(weights)
            for x_K, x_kj, pdf, prob in zip(energy_K, energy_K * K_TO_KJ_MOL, density, probability):
                csv_rows.append(
                    {
                        "mof": mof_name,
                        "energy_kind": kind,
                        "energy_K": float(x_K),
                        "energy_kJ_mol": float(x_kj),
                        "density_per_K": float(pdf),
                        "probability_mass": float(prob),
                    }
                )

    hist_csv = output_dir / f"{mof_name}_energy_histogram_bins.csv"
    with hist_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["mof", "energy_kind", "energy_K", "energy_kJ_mol", "density_per_K", "probability_mass"],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    plot_path = None
    if make_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8.0, 5.0))
            for kind in ["hostguest", "vdw", "coulomb", "guestguest", "total"]:
                path = latest.get(kind)
                if not path:
                    continue
                energy_K, density = _read_histogram_file(path)
                if len(energy_K) == 0:
                    continue
                plt.plot(energy_K * K_TO_KJ_MOL, density, label=kind)
            plt.xlabel("Energy (kJ/mol)")
            plt.ylabel("Histogram density")
            plt.title(f"{mof_name} RASPA energy histogram")
            plt.legend(frameon=False)
            plt.tight_layout()
            plot_path = output_dir / f"{mof_name}_energy_histogram.png"
            plt.savefig(plot_path, dpi=180)
            plt.close()
        except Exception as exc:
            plot_path = None
            distributions.setdefault("_plot_warning", str(exc))

    hostguest = distributions.get("hostguest", {})
    per_molecule_estimate: Dict[str, Any] = {
        "definition": "system-level hostguest energy divided by estimated average number of adsorbed molecules in the simulated supercell",
        "limitation": "approximate only; the histogram is not conditioned on instantaneous molecule count.",
    }
    n_mol = _clean_float(loading.get("estimated_average_molecules_in_simulation_cell"))
    if n_mol and n_mol > 0 and hostguest.get("status") == "ok":
        for key in ("mean_kJ_mol", "std_kJ_mol", "mode_kJ_mol", "min_sampled_kJ_mol", "max_sampled_kJ_mol"):
            value = _clean_float(hostguest.get(key))
            if value is not None:
                per_molecule_estimate[f"{key}_per_molecule_estimate"] = value / n_mol
        quantiles = hostguest.get("quantiles_kJ_mol") or {}
        per_molecule_estimate["quantiles_kJ_mol_per_molecule_estimate"] = {
            key: (_clean_float(value) / n_mol if _clean_float(value) is not None else None)
            for key, value in quantiles.items()
        }
        per_molecule_estimate["estimated_average_molecules_in_simulation_cell"] = n_mol
    else:
        per_molecule_estimate["status"] = "not_available"

    result = {
        "method": "energy_histogram_analysis",
        "mof": mof_name,
        "work_dir": str(work_dir),
        "histogram_dir": str(hist_dir),
        "settings": settings,
        "loading_context": loading,
        "primary_distribution": "hostguest",
        "hostguest_summary": hostguest,
        "hostguest_per_molecule_estimate": per_molecule_estimate,
        "distributions": distributions,
        "bins_csv": str(hist_csv),
        "plot_png": str(plot_path) if plot_path else None,
        "interpretation_notes": [
            "hostguest is the most direct distribution for total sampled guest-framework interaction strength.",
            "RASPA energy histograms are system-level energies; use hostguest_per_molecule_estimate for a rough per-adsorbate comparison when loading is available.",
            "More negative mean/mode/low-percentile hostguest energy indicates stronger sampled host-guest attraction.",
            "Strong-binding fractions quantify how much probability mass lies below selected energy thresholds.",
            "This is a sampled energy distribution at the simulated loading/pressure, not a zero-loading binding energy.",
        ],
    }

    json_path = output_dir / f"{mof_name}_energy_histogram_summary.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["summary_json"] = str(json_path)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize RASPA EnergyHistograms output.")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--mof", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--thresholds", default=",".join(str(x) for x in DEFAULT_STRONG_THRESHOLDS_KJ_MOL))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    result = run_energy_histogram_analysis(
        work_dir=Path(args.work_dir),
        mof=args.mof,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        strong_thresholds_kJ_mol=thresholds,
        make_plot=not args.no_plot,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
