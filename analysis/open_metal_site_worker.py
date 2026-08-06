from __future__ import annotations

import argparse
import csv
import json
import tempfile
from collections import Counter, defaultdict
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
from ase.io import read
from mofstructure import mofdeconstructor
from omsdetector_forked import mof as oms_mof


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _guest_free_structure(atoms):
    keep_raw = mofdeconstructor.remove_unbound_guest(atoms)
    if hasattr(keep_raw, "get_chemical_symbols"):
        return keep_raw, list(range(len(keep_raw)))
    keep = [int(index) for index in list(keep_raw)]
    return atoms[keep], keep


def _aggregate_sites(sites: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for site in sites:
        grouped[str(site["metal"])].append(site)

    by_metal: Dict[str, Any] = {}
    for metal, records in sorted(grouped.items()):
        coordination_numbers = Counter(int(record["coordination_number"]) for record in records)
        environments = Counter(
            "-".join(sorted(record.get("neighbor_species", []))) or "none"
            for record in records
        )
        n_open = sum(bool(record["is_open"]) for record in records)
        by_metal[metal] = {
            "n_sites": len(records),
            "n_open_sites": n_open,
            "open_site_fraction": float(n_open / len(records)) if records else 0.0,
            "coordination_number_counts": {
                str(key): int(value) for key, value in sorted(coordination_numbers.items())
            },
            "neighbor_environment_counts": dict(sorted(environments.items())),
        }
    return by_metal


def _write_sites_csv(path: Path, sites: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "site_index",
        "source_atom_index",
        "metal",
        "coordination_number",
        "neighbor_species",
        "is_open",
        "is_unique_environment",
        "problematic",
        "fractional_x",
        "fractional_y",
        "fractional_z",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for site in sites:
            frac = site.get("fractional_position", [None, None, None])
            writer.writerow(
                {
                    "site_index": site.get("site_index"),
                    "source_atom_index": site.get("source_atom_index"),
                    "metal": site.get("metal"),
                    "coordination_number": site.get("coordination_number"),
                    "neighbor_species": ",".join(site.get("neighbor_species", [])),
                    "is_open": site.get("is_open"),
                    "is_unique_environment": site.get("is_unique_environment"),
                    "problematic": site.get("problematic"),
                    "fractional_x": frac[0],
                    "fractional_y": frac[1],
                    "fractional_z": frac[2],
                }
            )


def analyze_cif(cif_path: Path, output_root: Path) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "status": "ok",
        "mof": cif_path.stem,
        "cif_path": str(cif_path),
        "engine": "mofstructure/omsdetector-forked",
        "algorithm": "omsdetector_forked.MofStructure.analyze_metals",
    }
    try:
        atoms = read(str(cif_path))
        clean_atoms, source_indices0 = _guest_free_structure(atoms)
        detector = oms_mof.MofStructure(
            lattice=clean_atoms.get_cell().tolist(),
            species=clean_atoms.get_chemical_symbols(),
            coords=clean_atoms.get_positions(),
            coords_are_cartesian=True,
            name=cif_path.stem,
        )

        with tempfile.TemporaryDirectory(prefix="simmof_oms_") as temp_dir:
            detector.analyze_metals(temp_dir)

        sites: List[Dict[str, Any]] = []
        for site_index0, (clean_index0, sphere, raw) in enumerate(
            zip(
                detector.metal_indices,
                detector.metal_coord_spheres,
                detector.summary.get("metal_sites", []),
            )
        ):
            clean_index0 = int(clean_index0)
            source_index0 = source_indices0[clean_index0]
            neighbor_species = [str(species) for species in sphere.species[1:]]
            sites.append(
                {
                    "site_index": site_index0 + 1,
                    "source_atom_index": int(source_index0) + 1,
                    "clean_structure_atom_index": clean_index0 + 1,
                    "metal": str(raw.get("metal", clean_atoms[clean_index0].symbol)),
                    "coordination_number": int(raw.get("number_of_linkers", len(neighbor_species))),
                    "neighbor_species": neighbor_species,
                    "neighbor_species_counts": dict(sorted(Counter(neighbor_species).items())),
                    "is_open": bool(raw.get("is_open")),
                    "classification": (
                        "open_metal_site"
                        if bool(raw.get("is_open"))
                        else "coordinatively_saturated_metal_site"
                    ),
                    "is_unique_environment": bool(raw.get("unique")),
                    "problematic": bool(raw.get("problematic")),
                    "t_factor": _json_value(raw.get("t_factor")),
                    "cartesian_position_A": [
                        float(value) for value in clean_atoms.positions[clean_index0]
                    ],
                    "fractional_position": [
                        float(value) % 1.0
                        for value in clean_atoms.get_scaled_positions(wrap=False)[clean_index0]
                    ],
                }
            )

        output_dir = output_root / cif_path.stem
        csv_path = output_dir / "open_metal_sites.csv"
        _write_sites_csv(csv_path, sites)
        n_open = sum(bool(site["is_open"]) for site in sites)
        open_site_indices = [
            int(site["source_atom_index"]) for site in sites if site["is_open"]
        ]
        record.update(
            {
                "n_atoms": len(atoms),
                "n_atoms_after_guest_removal": len(clean_atoms),
                "guest_removed_atoms": len(atoms) - len(clean_atoms),
                "n_metal_sites": len(sites),
                "n_open_metal_sites": n_open,
                "open_site_fraction": float(n_open / len(sites)) if sites else 0.0,
                "has_open_metal_sites": bool(n_open),
                "open_metal_elements": sorted(
                    {site["metal"] for site in sites if site["is_open"]}
                ),
                "open_site_source_atom_indices": open_site_indices,
                "sites": sites,
                "by_metal": _aggregate_sites(sites),
                "output_csv": str(csv_path),
                "library_summary": {
                    key: _json_value(detector.summary.get(key))
                    for key in (
                        "has_oms",
                        "oms_density",
                        "metal_species",
                        "non_metal_species",
                        "density",
                        "uc_volume",
                        "problematic",
                    )
                },
                "interpretation_note": (
                    "is_open is the OMS detector's geometric coordination-site "
                    "classification; coordination number alone is not used as a fixed cutoff."
                ),
            }
        )
        if not sites:
            record["status"] = "no_metal_sites"
    except Exception as exc:
        record.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with Path(args.input).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    output_root = Path(payload["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    structures = [
        analyze_cif(Path(path), output_root)
        for path in payload.get("cif_paths", [])
    ]
    status = "ok" if structures and all(item["status"] != "error" for item in structures) else "partial_error"
    if not structures:
        status = "no_cif_paths_found"
    summary = {
        "method": "open_metal_site_analysis",
        "status": status,
        "n_structures": len(structures),
        "output_dir": str(output_root),
        "summary_json": str(args.output),
        "engine": "mofstructure/omsdetector-forked",
        "versions": {
            "mofstructure": metadata.version("mofstructure"),
            "omsdetector-forked": metadata.version("omsdetector-forked"),
        },
        "structures": structures,
    }
    with Path(args.output).open("w", encoding="utf-8") as handle:
        json.dump(_json_value(summary), handle, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
