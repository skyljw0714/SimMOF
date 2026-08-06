from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


METAL_SPECIES = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Al", "Ga", "In", "Sn", "Pb", "Bi",
}


def site_type_for_species(species: str) -> str:
    if species in METAL_SPECIES:
        return "metal_site"
    if species in {"O", "N", "S", "P", "F", "Cl", "Br", "I"}:
        return "polar_linker_or_heteroatom"
    if species == "C":
        return "organic_linker_carbon"
    if species == "H":
        return "hydrogen_contact"
    return "framework_atom"


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(v for v in weights.values() if v > 0.0))
    if total <= 0.0:
        return {}
    return {k: float(v / total) for k, v in sorted(weights.items()) if v > 0.0}


def parse_float_token(token: str) -> float:
    token = token.strip().strip("'\"")
    token = re.sub(r"\([0-9]+\)$", "", token)
    return float(token)


def cell_matrix_from_lengths_angles(cell: Dict[str, float]) -> np.ndarray:
    a = cell["a"]
    b = cell["b"]
    c = cell["c"]
    alpha = math.radians(cell["alpha"])
    beta = math.radians(cell["beta"])
    gamma = math.radians(cell["gamma"])

    va = np.array([a, 0.0, 0.0], dtype=float)
    vb = np.array([b * math.cos(gamma), b * math.sin(gamma), 0.0], dtype=float)
    cx = c * math.cos(beta)
    cy = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / max(math.sin(gamma), 1e-12)
    cz2 = c * c - cx * cx - cy * cy
    vc = np.array([cx, cy, math.sqrt(max(cz2, 0.0))], dtype=float)
    return np.vstack([va, vb, vc])


def parse_cif_atoms(cif_path: Path) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    lines = cif_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    cell: Dict[str, float] = {}
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        key = parts[0]
        if key == "_cell_length_a":
            cell["a"] = parse_float_token(parts[1])
        elif key == "_cell_length_b":
            cell["b"] = parse_float_token(parts[1])
        elif key == "_cell_length_c":
            cell["c"] = parse_float_token(parts[1])
        elif key == "_cell_angle_alpha":
            cell["alpha"] = parse_float_token(parts[1])
        elif key == "_cell_angle_beta":
            cell["beta"] = parse_float_token(parts[1])
        elif key == "_cell_angle_gamma":
            cell["gamma"] = parse_float_token(parts[1])

    missing = {"a", "b", "c", "alpha", "beta", "gamma"} - set(cell)
    if missing:
        raise ValueError(f"CIF cell fields missing from {cif_path}: {sorted(missing)}")

    atoms: List[Dict[str, Any]] = []
    for i, line in enumerate(lines):
        if "_atom_site_fract_x" not in line:
            continue

        header_start = i
        while header_start > 0 and lines[header_start - 1].strip().startswith("_"):
            header_start -= 1
        headers = [x.strip() for x in lines[header_start : i + 1]]
        j = i + 1
        while j < len(lines) and lines[j].strip().startswith("_"):
            headers.append(lines[j].strip())
            j += 1

        if "_atom_site_fract_y" not in headers or "_atom_site_fract_z" not in headers:
            continue

        sym_idx = None
        for candidate in ("_atom_site_type_symbol", "_atom_site_label"):
            if candidate in headers:
                sym_idx = headers.index(candidate)
                break
        if sym_idx is None:
            continue

        x_idx = headers.index("_atom_site_fract_x")
        y_idx = headers.index("_atom_site_fract_y")
        z_idx = headers.index("_atom_site_fract_z")
        max_idx = max(sym_idx, x_idx, y_idx, z_idx)

        while j < len(lines):
            raw = lines[j].strip()
            if not raw or raw == "loop_" or raw.startswith("_"):
                break
            parts = raw.split()
            if len(parts) > max_idx:
                label = parts[sym_idx]
                symbol_match = re.match(r"([A-Z][a-z]?)", label)
                symbol = symbol_match.group(1) if symbol_match else label
                try:
                    atoms.append(
                        {
                            "source_index": len(atoms) + 1,
                            "species": symbol,
                            "frac": np.array(
                                [
                                    parse_float_token(parts[x_idx]),
                                    parse_float_token(parts[y_idx]),
                                    parse_float_token(parts[z_idx]),
                                ],
                                dtype=float,
                            ),
                        }
                    )
                except Exception:
                    pass
            j += 1
        if atoms:
            break

    if not atoms:
        raise ValueError(f"No atom_site fractional coordinates parsed from {cif_path}")
    return cell, atoms


def parse_unit_cells(simulation_input: Path) -> Tuple[int, int, int]:
    for line in simulation_input.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if parts[:1] == ["UnitCells"] and len(parts) >= 4:
            return int(parts[1]), int(parts[2]), int(parts[3])
    return 1, 1, 1


def build_supercell_atoms(
    cif_path: Path,
    unit_cells: Tuple[int, int, int],
    unit_by_source_index: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]], np.ndarray]:
    cell, atoms = parse_cif_atoms(cif_path)
    base_cell = cell_matrix_from_lengths_angles(cell)
    ux, uy, uz = unit_cells
    positions: List[np.ndarray] = []
    records: List[Dict[str, Any]] = []

    for ix in range(ux):
        for iy in range(uy):
            for iz in range(uz):
                offset = np.array([ix, iy, iz], dtype=float)
                for atom in atoms:
                    frac_super = atom["frac"] + offset
                    pos = frac_super @ base_cell
                    source_index = int(atom["source_index"])
                    unit = (unit_by_source_index or {}).get(source_index)
                    positions.append(pos)
                    records.append(
                        {
                            "species": atom["species"],
                            "source_index": source_index,
                            "supercell_offset": [ix, iy, iz],
                            "site_type": site_type_for_species(atom["species"]),
                            "chemistry_unit": compact_chemistry_unit(unit),
                        }
                    )

    supercell = np.array(
        [
            base_cell[0] * ux,
            base_cell[1] * uy,
            base_cell[2] * uz,
        ],
        dtype=float,
    )
    return np.array(positions, dtype=float), records, supercell


def compact_chemistry_unit(unit: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not unit:
        return None
    return {
        "unit_type": unit.get("unit_type") or unit.get("kind"),
        "unit_index": unit.get("unit_index") or unit.get("index"),
        "formula": unit.get("formula"),
        "sbu_type": unit.get("sbu_type"),
        "functional_tags": unit.get("functional_tags", []),
        "smiles": unit.get("smiles"),
        "inchikey": unit.get("inchikey"),
    }


def parse_chemistry_units(summary_path: Optional[Path], cif_path: Path) -> Dict[int, Dict[str, Any]]:
    if not summary_path or not summary_path.exists():
        return {}
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    structures = summary.get("structures", []) or []
    selected = None
    cif_resolved = str(cif_path.resolve())
    for item in structures:
        try:
            if str(Path(item.get("cif_path", "")).resolve()) == cif_resolved:
                selected = item
                break
        except Exception:
            pass
    if selected is None and structures:
        selected = structures[0]
    if not selected or selected.get("status") != "ok":
        return {}

    out: Dict[int, Dict[str, Any]] = {}
    for key, unit_type in (("nodes", "node"), ("linkers", "linker")):
        for unit in selected.get(key, []) or []:
            record = {
                "unit_type": unit_type,
                "unit_index": unit.get("index"),
                "formula": unit.get("formula"),
                "sbu_type": unit.get("sbu_type"),
                "functional_tags": unit.get("functional_tags", []),
                "smiles": unit.get("smiles"),
                "inchikey": unit.get("inchikey"),
            }
            for idx in unit.get("source_atom_indices", []) or []:
                out[int(idx)] = record
    return out


def parse_structured_density_vtk(vtk_path: Path) -> Tuple[Tuple[int, int, int], np.ndarray, np.ndarray, np.ndarray]:
    lines = vtk_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    dims = None
    spacing = None
    origin = np.zeros(3, dtype=float)
    data_start = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("DIMENSIONS"):
            dims = tuple(int(x) for x in stripped.split()[1:4])
        elif stripped.startswith("ASPECT_RATIO") or stripped.startswith("SPACING"):
            spacing = np.array([float(x) for x in stripped.split()[1:4]], dtype=float)
        elif stripped.startswith("ORIGIN"):
            origin = np.array([float(x) for x in stripped.split()[1:4]], dtype=float)
        elif stripped.startswith("LOOKUP_TABLE"):
            data_start = i + 1
            break

    if dims is None or spacing is None or data_start is None:
        raise ValueError(f"Could not parse structured density VTK header: {vtk_path}")

    values: List[float] = []
    for line in lines[data_start:]:
        for token in line.split():
            try:
                values.append(float(token))
            except ValueError:
                pass
    values_array = np.array(values[: int(np.prod(dims))], dtype=float)
    return dims, spacing, origin, values_array


def parse_polydata_points_vtk(vtk_path: Path) -> np.ndarray:
    lines = vtk_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    n_points = None
    start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("POINTS"):
            parts = stripped.split()
            n_points = int(parts[1])
            start = i + 1
            break
    if n_points is None or start is None:
        raise ValueError(f"Could not parse POLYDATA POINTS from {vtk_path}")

    values: List[float] = []
    for line in lines[start:]:
        stripped = line.strip()
        if values and re.match(r"^[A-Z_]+", stripped):
            break
        for token in stripped.split():
            try:
                values.append(float(token))
            except ValueError:
                pass
        if len(values) >= n_points * 3:
            break
    return np.array(values[: n_points * 3], dtype=float).reshape(-1, 3)


def voxel_indices_to_positions(indices: np.ndarray, dims: Tuple[int, int, int], spacing: np.ndarray, origin: np.ndarray) -> np.ndarray:
    nx, ny, _nz = dims
    z = indices // (nx * ny)
    rem = indices % (nx * ny)
    y = rem // nx
    x = rem % nx
    return origin + np.vstack([x, y, z]).T.astype(float) * spacing


def image_positions(positions: np.ndarray, records: List[Dict[str, Any]], cell: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    shifts = []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                shifts.append(i * cell[0] + j * cell[1] + k * cell[2])
    all_positions = []
    source_indices = []
    for shift in shifts:
        all_positions.append(positions + shift)
        source_indices.extend(range(len(records)))
    return np.vstack(all_positions), source_indices


def nearest_contacts_for_points(
    points: np.ndarray,
    point_weights: np.ndarray,
    framework_positions: np.ndarray,
    framework_records: List[Dict[str, Any]],
    supercell: np.ndarray,
    cutoff_A: float,
    top_contacts_per_point: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    img_positions, img_to_record = image_positions(framework_positions, framework_records, supercell)
    contacts: List[Dict[str, Any]] = []

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(img_positions)
        for point_i, (point, density) in enumerate(zip(points, point_weights)):
            distances, indices = tree.query(point, k=min(top_contacts_per_point, len(img_positions)), distance_upper_bound=cutoff_A)
            distances = np.atleast_1d(distances)
            indices = np.atleast_1d(indices)
            for d, img_idx in zip(distances, indices):
                if not np.isfinite(d) or img_idx >= len(img_positions):
                    continue
                rec = framework_records[img_to_record[int(img_idx)]]
                weight = float(density) * max(0.0, 1.0 - float(d) / cutoff_A)
                if weight <= 0.0:
                    continue
                contacts.append(contact_record(point_i, point, density, rec, d, weight))
    except Exception:
        for point_i, (point, density) in enumerate(zip(points, point_weights)):
            dists = np.linalg.norm(img_positions - point[None, :], axis=1)
            candidates = np.where(dists <= cutoff_A)[0]
            if len(candidates) == 0:
                continue
            candidates = candidates[np.argsort(dists[candidates])[:top_contacts_per_point]]
            for img_idx in candidates:
                d = float(dists[img_idx])
                rec = framework_records[img_to_record[int(img_idx)]]
                weight = float(density) * max(0.0, 1.0 - d / cutoff_A)
                if weight <= 0.0:
                    continue
                contacts.append(contact_record(point_i, point, density, rec, d, weight))

    diagnostics = {
        "n_density_points": int(len(points)),
        "n_contacts": int(len(contacts)),
        "cutoff_A": float(cutoff_A),
        "top_contacts_per_point": int(top_contacts_per_point),
    }
    return contacts, diagnostics


def contact_record(point_i: int, point: np.ndarray, density: float, rec: Dict[str, Any], distance_A: float, weight: float) -> Dict[str, Any]:
    unit = rec.get("chemistry_unit")
    return {
        "hotspot_index": int(point_i),
        "hotspot_cartesian_A": [float(x) for x in point],
        "density": float(density),
        "framework_species": rec.get("species"),
        "framework_source_index": rec.get("source_index"),
        "framework_supercell_offset": rec.get("supercell_offset"),
        "distance_A": float(distance_A),
        "weight": float(weight),
        "site_type": rec.get("site_type"),
        "chemistry_unit": unit,
        "unit_type": unit.get("unit_type") if unit else "unassigned",
        "unit_formula": unit.get("formula") if unit else None,
        "sbu_type": unit.get("sbu_type") if unit else None,
    }


def select_density_hotspots(
    dims: Tuple[int, int, int],
    spacing: np.ndarray,
    origin: np.ndarray,
    values: np.ndarray,
    percentile: float,
    max_points: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    nonzero = values[values > 0.0]
    if nonzero.size == 0:
        return np.empty((0, 3)), np.empty((0,)), {"threshold": None, "n_nonzero_voxels": 0}
    threshold = float(np.percentile(nonzero, percentile))
    indices = np.where(values >= threshold)[0]
    if indices.size > max_points:
        ranked = indices[np.argsort(values[indices])[-max_points:]]
        indices = ranked
    weights = values[indices]
    points = voxel_indices_to_positions(indices, dims, spacing, origin)
    order = np.argsort(weights)[::-1]
    sorted_nonzero = np.sort(nonzero)[::-1]
    total_density = float(np.sum(sorted_nonzero))

    def top_fraction(percent: float) -> Optional[float]:
        if total_density <= 0.0:
            return None
        n = max(1, int(math.ceil(sorted_nonzero.size * percent / 100.0)))
        return float(np.sum(sorted_nonzero[:n]) / total_density)

    if total_density > 0.0:
        ascending = np.sort(nonzero)
        n = float(ascending.size)
        density_gini = float((2.0 * np.sum((np.arange(1, ascending.size + 1) * ascending)) / (n * np.sum(ascending))) - ((n + 1.0) / n))
    else:
        density_gini = None

    return points[order], weights[order], {
        "threshold": threshold,
        "percentile": float(percentile),
        "n_nonzero_voxels": int(nonzero.size),
        "n_selected_voxels": int(len(indices)),
        "max_density": float(values.max()),
        "density_concentration": {
            "definition": "Fractions of total nonzero density mass contained in the highest-density voxel subsets.",
            "total_nonzero_density": total_density,
            "selected_hotspot_density_fraction": (
                float(np.sum(weights) / total_density) if total_density > 0.0 else None
            ),
            "top_1_percent_density_fraction": top_fraction(1.0),
            "top_5_percent_density_fraction": top_fraction(5.0),
            "top_10_percent_density_fraction": top_fraction(10.0),
            "density_gini_coefficient": density_gini,
        },
    }


def summarize_contacts(contacts: List[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
    species_weights: Dict[str, float] = {}
    site_type_weights: Dict[str, float] = {}
    unit_type_weights: Dict[str, float] = {}
    unit_weights: Dict[str, Dict[str, Any]] = {}

    for c in contacts:
        w = float(c.get("weight", 0.0))
        if w <= 0.0:
            continue
        species = c.get("framework_species") or "unknown"
        site_type = c.get("site_type") or "unknown"
        unit_type = c.get("unit_type") or "unassigned"
        species_weights[species] = species_weights.get(species, 0.0) + w
        site_type_weights[site_type] = site_type_weights.get(site_type, 0.0) + w
        unit_type_weights[unit_type] = unit_type_weights.get(unit_type, 0.0) + w

        unit = c.get("chemistry_unit")
        unit_key = f"{unit_type}:{c.get('unit_formula')}:{c.get('sbu_type')}"
        rec = unit_weights.setdefault(
            unit_key,
            {
                "unit": unit,
                "unit_type": unit_type,
                "unit_formula": c.get("unit_formula"),
                "sbu_type": c.get("sbu_type"),
                "total_weight": 0.0,
                "nearest_distance_A": None,
                "contact_species": set(),
                "n_contacts": 0,
            },
        )
        rec["total_weight"] += w
        rec["n_contacts"] += 1
        rec["contact_species"].add(species)
        d = float(c.get("distance_A", 9999.0))
        if rec["nearest_distance_A"] is None or d < rec["nearest_distance_A"]:
            rec["nearest_distance_A"] = d

    unit_summary = []
    total_weight = float(sum(species_weights.values()))
    for rec in unit_weights.values():
        unit_summary.append(
            {
                "unit": rec["unit"],
                "unit_type": rec["unit_type"],
                "unit_formula": rec["unit_formula"],
                "sbu_type": rec["sbu_type"],
                "weight": float(rec["total_weight"]),
                "weight_fraction": float(rec["total_weight"] / total_weight) if total_weight > 0.0 else None,
                "nearest_distance_A": rec["nearest_distance_A"],
                "contact_species": sorted(rec["contact_species"]),
                "n_contacts": rec["n_contacts"],
            }
        )
    unit_summary.sort(key=lambda x: x["weight"], reverse=True)

    sorted_contacts = sorted(contacts, key=lambda x: x.get("weight", 0.0), reverse=True)
    return {
        "definition": "Density-weighted contact fingerprint around CH4 adsorption-density hotspots.",
        "weighting": "contact_weight = density_value * max(0, 1 - distance_A / cutoff_A)",
        "species_weight_fraction": normalize_weights(species_weights),
        "site_type_weight_fraction": normalize_weights(site_type_weights),
        "unit_type_weight_fraction": normalize_weights(unit_type_weights),
        "unit_contact_weight_summary": unit_summary[:top_k],
        "top_weighted_contacts": sorted_contacts[:top_k],
        "diagnostics": {
            "total_contact_weight": total_weight,
            "n_contacts": len(contacts),
            "chemistry_unit_coverage_fraction": (
                float(sum(1 for c in contacts if c.get("chemistry_unit")) / len(contacts)) if contacts else None
            ),
        },
    }


def run_adsorption_site_density_analysis(
    work_dir: Path,
    mof: Optional[str] = None,
    guest_label: str = "methane",
    percentile: float = 99.75,
    max_points: int = 1000,
    cutoff_A: float = 4.0,
    top_contacts_per_point: int = 8,
    top_k: int = 15,
    chemistry_summary_json: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    work_dir = Path(work_dir)
    mof = mof or work_dir.name.split("_CH4_")[0]
    output_dir = output_dir or work_dir / "adsorption_site_density_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    cif_path = work_dir / f"{mof}.cif"
    if not cif_path.exists():
        cifs = list(work_dir.glob("*.cif"))
        if not cifs:
            raise FileNotFoundError(f"No CIF found in {work_dir}")
        cif_path = cifs[0]

    sim_input = work_dir / "simulation.input"
    unit_cells = parse_unit_cells(sim_input) if sim_input.exists() else (1, 1, 1)
    unit_map = parse_chemistry_units(chemistry_summary_json, cif_path)
    framework_positions, framework_records, supercell = build_supercell_atoms(cif_path, unit_cells, unit_map)

    framework_vtk = work_dir / "VTK" / "System_0" / "FrameworkAtoms.vtk"
    coordinate_source = "cif_supercell"
    n_framework_vtk_points = None
    if framework_vtk.exists():
        try:
            vtk_points = parse_polydata_points_vtk(framework_vtk)
            n_framework_vtk_points = int(len(vtk_points))
            if len(vtk_points) >= len(framework_records):
                framework_positions = vtk_points[: len(framework_records)]
                coordinate_source = "FrameworkAtoms.vtk_first_n_points_with_cif_species_order"
        except Exception:
            pass

    vtk_path = work_dir / "VTK" / "System_0" / f"COMDensityProfile_{guest_label}.vtk"
    if not vtk_path.exists():
        matches = list((work_dir / "VTK" / "System_0").glob("COMDensityProfile_*.vtk"))
        if not matches:
            raise FileNotFoundError(f"No COMDensityProfile VTK found in {work_dir / 'VTK' / 'System_0'}")
        vtk_path = matches[0]

    dims, spacing, origin, values = parse_structured_density_vtk(vtk_path)
    hotspot_points, hotspot_weights, hotspot_diag = select_density_hotspots(
        dims, spacing, origin, values, percentile=percentile, max_points=max_points
    )
    contacts, contact_diag = nearest_contacts_for_points(
        hotspot_points,
        hotspot_weights,
        framework_positions,
        framework_records,
        supercell,
        cutoff_A=cutoff_A,
        top_contacts_per_point=top_contacts_per_point,
    )
    fingerprint = summarize_contacts(contacts, top_k=top_k)

    csv_path = output_dir / f"{mof}_adsorption_site_contacts.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "hotspot_index",
                "density",
                "x_A",
                "y_A",
                "z_A",
                "framework_species",
                "site_type",
                "distance_A",
                "weight",
                "unit_type",
                "unit_formula",
                "sbu_type",
                "framework_source_index",
            ],
        )
        writer.writeheader()
        for c in contacts:
            x, y, z = c["hotspot_cartesian_A"]
            writer.writerow(
                {
                    "hotspot_index": c["hotspot_index"],
                    "density": c["density"],
                    "x_A": x,
                    "y_A": y,
                    "z_A": z,
                    "framework_species": c["framework_species"],
                    "site_type": c["site_type"],
                    "distance_A": c["distance_A"],
                    "weight": c["weight"],
                    "unit_type": c.get("unit_type"),
                    "unit_formula": c.get("unit_formula"),
                    "sbu_type": c.get("sbu_type"),
                    "framework_source_index": c.get("framework_source_index"),
                }
            )

    result = {
        "method": "adsorption_site_density_analysis",
        "status": "ok",
        "mof": mof,
        "work_dir": str(work_dir),
        "cif_path": str(cif_path),
        "density_vtk": str(vtk_path),
        "unit_cells": list(unit_cells),
        "parameters": {
            "density_percentile": percentile,
            "max_density_points": max_points,
            "contact_cutoff_A": cutoff_A,
            "top_contacts_per_point": top_contacts_per_point,
        },
        "density_hotspots": hotspot_diag,
        "framework": {
            "n_cif_atoms": int(len(parse_cif_atoms(cif_path)[1])),
            "n_supercell_atoms": int(len(framework_records)),
            "coordinate_source": coordinate_source,
            "framework_vtk_points": n_framework_vtk_points,
            "n_atoms_with_chemistry_unit": int(sum(1 for r in framework_records if r.get("chemistry_unit"))),
        },
        "fingerprint": fingerprint,
        "contacts_csv": str(csv_path),
        "limitations": [
            "Hotspots are selected from the top density voxels, not from a continuous isosurface integral.",
            "Contacts are distance-based and describe proximity to adsorption density, not chemical bond formation.",
            "Node/linker assignment is only available when a mofstructure chemistry summary is provided.",
            "RASPA VTK coordinates can be scaled for visualization; distance_A is used for contact ranking and weighting, while weight fractions are the primary quantitative result.",
        ],
    }

    json_path = output_dir / f"{mof}_adsorption_site_density_summary.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["summary_json"] = str(json_path)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Quantify RASPA adsorption-site density against MOF framework atoms.")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--mof", default=None)
    parser.add_argument("--guest-label", default="methane")
    parser.add_argument("--percentile", type=float, default=99.75)
    parser.add_argument("--max-points", type=int, default=1000)
    parser.add_argument("--cutoff", type=float, default=4.0)
    parser.add_argument("--top-contacts-per-point", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--chemistry-summary-json", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    result = run_adsorption_site_density_analysis(
        work_dir=Path(args.work_dir),
        mof=args.mof,
        guest_label=args.guest_label,
        percentile=args.percentile,
        max_points=args.max_points,
        cutoff_A=args.cutoff,
        top_contacts_per_point=args.top_contacts_per_point,
        top_k=args.top_k,
        chemistry_summary_json=Path(args.chemistry_summary_json) if args.chemistry_summary_json else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
