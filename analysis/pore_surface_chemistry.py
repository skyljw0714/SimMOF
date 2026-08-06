from __future__ import annotations

import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from ase.data import atomic_numbers, covalent_radii, vdw_radii
from ase.geometry import find_mic
from ase.io import read


METAL_SPECIES = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Al", "Ga", "In", "Sn", "Pb", "Bi",
}
POLAR_HETEROATOMS = {"O", "N", "S", "P"}
HALOGENS = {"F", "Cl", "Br", "I"}


def _safe_token(value: Any) -> str:
    text = "".join(char if char.isalnum() or char in "._-" else "_" for char in str(value))
    return text.strip("_")[:120] or "structure"


def _fibonacci_sphere(n_points: int) -> np.ndarray:
    n_points = max(int(n_points), 32)
    index = np.arange(n_points, dtype=float)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    z = 1.0 - 2.0 * (index + 0.5) / n_points
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = golden_angle * index
    return np.column_stack((radius * np.cos(phi), radius * np.sin(phi), z))


def _radius_for_symbol(symbol: str) -> Tuple[float, str]:
    number = atomic_numbers.get(symbol)
    if number is not None and number < len(vdw_radii):
        value = vdw_radii[number]
        if value is not None and np.isfinite(value) and float(value) > 0.0:
            return float(value), "ASE_vdw_radius"
    if number is not None and number < len(covalent_radii):
        value = covalent_radii[number]
        if value is not None and np.isfinite(value) and float(value) > 0.0:
            return max(float(value) + 0.8, 1.2), "ASE_covalent_radius_plus_0.8A"
    return 1.8, "default_1.8A"


def _matching_chemistry_structure(
    chemistry_summary: Optional[Dict[str, Any]],
    cif_path: Path,
) -> Optional[Dict[str, Any]]:
    structures = (chemistry_summary or {}).get("structures", []) or []
    resolved = str(cif_path.resolve())
    for structure in structures:
        try:
            if str(Path(structure.get("cif_path", "")).resolve()) == resolved:
                return structure
        except Exception:
            continue
    for structure in structures:
        if structure.get("mof") == cif_path.stem:
            return structure
    return structures[0] if len(structures) == 1 else None


def _unit_record(unit: Dict[str, Any], unit_type: str) -> Dict[str, Any]:
    fingerprint = unit.get("functional_group_fingerprint", {}) or {}
    ring_descriptors = fingerprint.get("ring_descriptors", {}) or {}
    present_groups = fingerprint.get("present_groups", []) or []
    functional_tags = unit.get("functional_tags", []) or []
    aromatic = bool(ring_descriptors.get("aromatic_ring_count", 0))
    if not aromatic:
        aromatic = any("aromatic" in str(tag).lower() for tag in functional_tags)
    return {
        "unit_type": unit_type,
        "unit_index": unit.get("index"),
        "formula": unit.get("formula"),
        "smiles": unit.get("smiles"),
        "inchikey": unit.get("inchikey"),
        "sbu_type": unit.get("sbu_type"),
        "functional_tags": functional_tags,
        "functional_groups": present_groups,
        "is_aromatic_linker": bool(aromatic and unit_type == "linker"),
    }


def _unit_map(structure: Optional[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    if not structure:
        return out
    for unit in structure.get("nodes", []) or []:
        compact = _unit_record(unit, "node")
        for source_index in unit.get("source_atom_indices", []) or []:
            out[int(source_index)] = compact
    for unit in structure.get("linkers", []) or []:
        compact = _unit_record(unit, "linker")
        for source_index in unit.get("source_atom_indices", []) or []:
            out[int(source_index)] = compact
    return out


def _ligand_chemistry_map(
    structure: Optional[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    if not structure:
        return out
    for unit in structure.get("ligands", []) or []:
        compact = _unit_record(unit, "linker")
        for source_index in unit.get("source_atom_indices", []) or []:
            out[int(source_index)] = compact
    return out


def _framework_source_indices(
    atoms,
    chemistry_structure: Optional[Dict[str, Any]],
    unit_by_index: Dict[int, Dict[str, Any]],
) -> Tuple[List[int], str]:
    all_indices = list(range(1, len(atoms) + 1))
    if not chemistry_structure or not unit_by_index:
        return all_indices, "all_cif_atoms_no_chemistry_mapping"

    mapped = sorted(index for index in unit_by_index if 1 <= index <= len(atoms))
    expected = int(chemistry_structure.get("n_atoms_after_guest_removal") or len(atoms))
    guest_removed = int(chemistry_structure.get("guest_removed_atoms") or 0)
    coverage = len(mapped) / max(expected, 1)
    if guest_removed > 0 and coverage >= 0.9:
        return mapped, "mofstructure_guest_free_mapped_atoms"
    return all_indices, "all_cif_atoms"


def _surface_category(symbol: str, unit: Optional[Dict[str, Any]]) -> str:
    if symbol in METAL_SPECIES:
        return "metal_surface"
    if symbol in POLAR_HETEROATOMS:
        return f"{symbol}_functional_surface"
    if symbol in HALOGENS:
        return "halogen_surface"
    if symbol == "C" and unit and unit.get("is_aromatic_linker"):
        return "aromatic_linker_surface"
    if symbol == "C":
        return "nonaromatic_carbon_surface"
    if symbol == "H":
        return "hydrogen_surface"
    return "other_framework_surface"


def _normalized(values: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(values.values()))
    if total <= 0.0:
        return {}
    return {
        key: float(value / total)
        for key, value in sorted(values.items())
    }


def _axis_histograms(
    fractional_points: np.ndarray,
    weights: np.ndarray,
    bins: int,
) -> Dict[str, Any]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    histograms: Dict[str, Any] = {}
    for axis_index, axis in enumerate(("a", "b", "c")):
        hist, _ = np.histogram(
            fractional_points[:, axis_index],
            bins=edges,
            weights=weights,
        )
        total = float(hist.sum())
        histograms[axis] = {
            "bin_edges_fractional": [float(value) for value in edges],
            "surface_area_A2": [float(value) for value in hist],
            "surface_fraction": [
                float(value / total) if total > 0.0 else 0.0
                for value in hist
            ],
        }
    return histograms


def _top_spatial_regions(
    fractional_points: np.ndarray,
    weights: np.ndarray,
    bins: int,
    top_k: int = 12,
) -> List[Dict[str, Any]]:
    histogram, edges = np.histogramdd(
        fractional_points,
        bins=(bins, bins, bins),
        range=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        weights=weights,
    )
    flat = histogram.ravel()
    total = float(flat.sum())
    if total <= 0.0:
        return []
    order = np.argsort(flat)[::-1]
    regions = []
    for flat_index in order[:top_k]:
        value = float(flat[flat_index])
        if value <= 0.0:
            continue
        i, j, k = np.unravel_index(int(flat_index), histogram.shape)
        regions.append(
            {
                "fractional_bin": [int(i), int(j), int(k)],
                "fractional_center": [
                    float((edges[0][i] + edges[0][i + 1]) / 2.0),
                    float((edges[1][j] + edges[1][j + 1]) / 2.0),
                    float((edges[2][k] + edges[2][k + 1]) / 2.0),
                ],
                "surface_area_A2": value,
                "surface_fraction": float(value / total),
            }
        )
    return regions


def _write_surface_points(
    path: Path,
    rows: Iterable[Sequence[Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(str(path), "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source_atom_index",
                "species",
                "surface_category",
                "area_weight_A2",
                "fractional_a",
                "fractional_b",
                "fractional_c",
                "cartesian_x_A",
                "cartesian_y_A",
                "cartesian_z_A",
            ]
        )
        writer.writerows(rows)


def analyze_pore_surface_chemistry(
    cif_path: Path,
    output_dir: Path,
    chemistry_summary: Optional[Dict[str, Any]] = None,
    probe_radius_A: float = 1.86,
    samples_per_atom: int = 256,
    spatial_bins: int = 8,
) -> Dict[str, Any]:
    cif_path = Path(cif_path)
    output_dir = Path(output_dir)
    record: Dict[str, Any] = {
        "method": "pore_surface_chemistry_analysis",
        "status": "ok",
        "mof": cif_path.stem,
        "cif_path": str(cif_path),
        "probe_radius_A": float(probe_radius_A),
        "samples_per_atom": int(max(samples_per_atom, 32)),
        "algorithm": "periodic Shrake-Rupley-style probe-accessible atomic surface",
    }
    try:
        atoms_all = read(str(cif_path))
        chemistry_structure = _matching_chemistry_structure(chemistry_summary, cif_path)
        unit_by_index = _unit_map(chemistry_structure)
        ligand_by_index = _ligand_chemistry_map(chemistry_structure)
        source_indices1, atom_selection_method = _framework_source_indices(
            atoms_all,
            chemistry_structure,
            unit_by_index,
        )
        selected0 = [index - 1 for index in source_indices1]
        atoms = atoms_all[selected0]
        atoms.set_pbc(atoms_all.get_pbc())

        positions = np.array(atoms.get_positions(), dtype=float)
        cell = np.array(atoms.get_cell(), dtype=float)
        symbols = atoms.get_chemical_symbols()
        radii_with_source = [_radius_for_symbol(symbol) for symbol in symbols]
        radii = np.array([item[0] for item in radii_with_source], dtype=float)
        expanded_radii = radii + float(probe_radius_A)
        directions = _fibonacci_sphere(samples_per_atom)

        category_area: Dict[str, float] = defaultdict(float)
        element_area: Dict[str, float] = defaultdict(float)
        unit_type_area: Dict[str, float] = defaultdict(float)
        unit_area: Dict[str, float] = defaultdict(float)
        functional_group_carrier_area: Dict[str, float] = defaultdict(float)
        atom_records: List[Dict[str, Any]] = []
        all_fractional: List[np.ndarray] = []
        all_weights: List[np.ndarray] = []
        all_categories: List[np.ndarray] = []
        surface_rows: List[Sequence[Any]] = []
        inverse_cell = np.linalg.inv(cell)

        for local_index0, (source_index1, symbol, center, radius) in enumerate(
            zip(source_indices1, symbols, positions, radii)
        ):
            probe_centers = center + expanded_radii[local_index0] * directions
            deltas = probe_centers[:, None, :] - positions[None, :, :]
            mic_vectors, _ = find_mic(
                deltas.reshape(-1, 3),
                atoms.cell,
                pbc=atoms.pbc,
            )
            distances = np.linalg.norm(
                np.asarray(mic_vectors).reshape(len(directions), len(atoms), 3),
                axis=2,
            )
            distances[:, local_index0] = np.inf
            accessible = np.all(
                distances >= (expanded_radii[None, :] - 1e-8),
                axis=1,
            )
            accessible_directions = directions[accessible]
            area_per_sample = 4.0 * math.pi * radius * radius / len(directions)
            accessible_area = float(accessible.sum() * area_per_sample)
            unit = unit_by_index.get(int(source_index1))
            ligand_chemistry = ligand_by_index.get(int(source_index1))
            category = _surface_category(symbol, ligand_chemistry or unit)
            fraction_exposed = float(accessible.mean())

            category_area[category] += accessible_area
            element_area[symbol] += accessible_area
            if unit:
                unit_type = str(unit.get("unit_type") or "unmapped")
                unit_key = (
                    f"{unit_type}_{unit.get('unit_index')}_"
                    f"{unit.get('formula') or 'unknown'}"
                )
                unit_type_area[unit_type] += accessible_area
                unit_area[unit_key] += accessible_area
            else:
                unit_type_area["unmapped"] += accessible_area
            functional_group_source = ligand_chemistry or unit
            if functional_group_source:
                for group in (
                    functional_group_source.get("functional_groups", []) or []
                ):
                    functional_group_carrier_area[str(group)] += accessible_area

            atom_records.append(
                {
                    "source_atom_index": int(source_index1),
                    "species": symbol,
                    "surface_category": category,
                    "vdw_radius_A": float(radius),
                    "radius_source": radii_with_source[local_index0][1],
                    "accessible_surface_area_A2": accessible_area,
                    "fraction_of_atomic_surface_accessible": fraction_exposed,
                    "chemistry_unit": unit,
                    "ligand_chemistry": ligand_chemistry,
                }
            )

            if accessible_directions.size == 0:
                continue
            physical_points = center + radius * accessible_directions
            fractional_points = (physical_points @ inverse_cell) % 1.0
            weights = np.full(len(physical_points), area_per_sample, dtype=float)
            categories = np.full(len(physical_points), category, dtype=object)
            all_fractional.append(fractional_points)
            all_weights.append(weights)
            all_categories.append(categories)
            for frac, cart in zip(fractional_points, physical_points):
                surface_rows.append(
                    (
                        int(source_index1),
                        symbol,
                        category,
                        area_per_sample,
                        float(frac[0]),
                        float(frac[1]),
                        float(frac[2]),
                        float(cart[0]),
                        float(cart[1]),
                        float(cart[2]),
                    )
                )

        total_area = float(sum(category_area.values()))
        if all_fractional:
            fractional_array = np.concatenate(all_fractional, axis=0)
            weight_array = np.concatenate(all_weights, axis=0)
            category_array = np.concatenate(all_categories, axis=0)
        else:
            fractional_array = np.empty((0, 3), dtype=float)
            weight_array = np.empty(0, dtype=float)
            category_array = np.empty(0, dtype=object)

        spatial_distribution: Dict[str, Any] = {
            "fractional_axis_histograms": (
                _axis_histograms(fractional_array, weight_array, spatial_bins)
                if len(fractional_array)
                else {}
            ),
            "top_3d_regions": (
                _top_spatial_regions(fractional_array, weight_array, spatial_bins)
                if len(fractional_array)
                else []
            ),
            "target_category_distributions": {},
        }
        target_masks = {
            "metal_surface": category_array == "metal_surface",
            "O_N_S_functional_surface": np.isin(
                category_array,
                ["O_functional_surface", "N_functional_surface", "S_functional_surface"],
            ),
            "aromatic_linker_surface": category_array == "aromatic_linker_surface",
        }
        for target, mask in target_masks.items():
            if mask.any():
                spatial_distribution["target_category_distributions"][target] = {
                    "fractional_axis_histograms": _axis_histograms(
                        fractional_array[mask],
                        weight_array[mask],
                        spatial_bins,
                    ),
                    "top_3d_regions": _top_spatial_regions(
                        fractional_array[mask],
                        weight_array[mask],
                        spatial_bins,
                        top_k=8,
                    ),
                }

        structure_dir = output_dir / _safe_token(cif_path.stem)
        points_path = structure_dir / "pore_surface_points.csv.gz"
        _write_surface_points(points_path, surface_rows)
        atom_records.sort(
            key=lambda item: item["accessible_surface_area_A2"],
            reverse=True,
        )
        ons_area = sum(
            category_area.get(f"{symbol}_functional_surface", 0.0)
            for symbol in ("O", "N", "S")
        )
        target_area = {
            "metal_surface": float(category_area.get("metal_surface", 0.0)),
            "O_N_S_functional_surface": float(ons_area),
            "aromatic_linker_surface": float(
                category_area.get("aromatic_linker_surface", 0.0)
            ),
        }
        record.update(
            {
                "n_cif_atoms": len(atoms_all),
                "n_framework_atoms_analyzed": len(atoms),
                "atom_selection_method": atom_selection_method,
                "chemistry_mapping": {
                    "status": "available" if unit_by_index else "unavailable",
                    "n_reticular_unit_mapped_source_atoms": len(unit_by_index),
                    "n_ligand_chemistry_mapped_source_atoms": len(
                        ligand_by_index
                    ),
                    "matched_structure": (
                        chemistry_structure.get("mof")
                        if chemistry_structure
                        else None
                    ),
                },
                "total_probe_accessible_atomic_surface_area_A2": total_area,
                "surface_area_by_category_A2": dict(sorted(category_area.items())),
                "surface_fraction_by_category": _normalized(category_area),
                "surface_area_by_element_A2": dict(sorted(element_area.items())),
                "surface_fraction_by_element": _normalized(element_area),
                "surface_area_by_unit_type_A2": dict(sorted(unit_type_area.items())),
                "surface_fraction_by_unit_type": _normalized(unit_type_area),
                "surface_area_by_chemistry_unit_A2": dict(sorted(unit_area.items())),
                "surface_area_of_linkers_carrying_functional_group_A2": dict(
                    sorted(functional_group_carrier_area.items())
                ),
                "target_surface_area_A2": target_area,
                "target_surface_fractions": {
                    key: float(value / total_area) if total_area > 0.0 else 0.0
                    for key, value in target_area.items()
                },
                "spatial_distribution": spatial_distribution,
                "top_exposed_atoms": atom_records[:25],
                "per_atom_surface": atom_records,
                "surface_points_csv_gz": str(points_path),
                "limitations": [
                    "This is a local periodic probe-accessibility estimate, not a pore-network connectivity calculation.",
                    "Fractions depend on the probe radius, atomic radii, and surface sampling density.",
                    "Aromatic surface assignment uses linker-level RDKit aromaticity and accessible linker carbon atoms.",
                    "Node/linker ownership follows SBU decomposition; full metal-ligand-cut ligands are used only for functional-group and aromatic chemistry.",
                    "Functional-group carrier areas cover the whole accessible surface of a linker carrying that group, not only the matched functional-group atoms.",
                ],
            }
        )
        if total_area <= 0.0:
            record["status"] = "no_accessible_surface"
    except Exception as exc:
        record.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
    return record


def run_pore_surface_chemistry_analysis(
    cif_paths: Iterable[Path],
    output_dir: Path,
    chemistry_summary: Optional[Dict[str, Any]] = None,
    probe_radius_A: float = 1.86,
    samples_per_atom: int = 256,
    spatial_bins: int = 8,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    structures = [
        analyze_pore_surface_chemistry(
            Path(cif_path),
            output_dir=output_dir,
            chemistry_summary=chemistry_summary,
            probe_radius_A=probe_radius_A,
            samples_per_atom=samples_per_atom,
            spatial_bins=spatial_bins,
        )
        for cif_path in cif_paths
    ]
    status = "ok" if structures and all(item["status"] != "error" for item in structures) else "partial_error"
    if not structures:
        status = "no_cif_paths_found"
    summary = {
        "method": "pore_surface_chemistry_analysis",
        "status": status,
        "n_structures": len(structures),
        "output_dir": str(output_dir),
        "structures": structures,
    }
    summary_path = output_dir / "pore_surface_chemistry_summary.json"
    summary["summary_json"] = str(summary_path)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary
