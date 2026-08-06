from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _integral(y: np.ndarray, x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    return float(np.trapz(y, x))


def _charge_density(
    densities: np.ndarray,
    component_names: Sequence[str],
) -> np.ndarray:
    names = [str(name) for name in component_names]
    if names == ["up", "down"]:
        return densities.sum(axis=-1)
    if names and names[0] == "charge":
        return densities[..., 0]
    return densities[..., 0]


def _orbital_fractions(
    orbital_density: np.ndarray,
    energies: np.ndarray,
    orbital_groups: Sequence[str],
) -> Dict[str, float]:
    areas = np.asarray(
        [
            _integral(np.clip(orbital_density[:, index], 0.0, None), energies)
            for index in range(orbital_density.shape[1])
        ],
        dtype=float,
    )
    total = float(areas.sum())
    if total <= 0.0:
        return {str(name): 0.0 for name in orbital_groups}
    return {
        str(name): float(areas[index] / total)
        for index, name in enumerate(orbital_groups)
    }


def _dominant_peaks(
    orbital_density: np.ndarray,
    energies: np.ndarray,
    orbital_groups: Sequence[str],
    top_k: int = 4,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for orbital_index, orbital_name in enumerate(orbital_groups):
        values = np.clip(orbital_density[:, orbital_index], 0.0, None)
        if not np.any(values > 0.0):
            continue
        for energy_index in range(1, max(1, len(values) - 1)):
            if (
                values[energy_index] >= values[energy_index - 1]
                and values[energy_index] >= values[energy_index + 1]
            ):
                candidates.append(
                    {
                        "orbital": str(orbital_name),
                        "energy_ev": float(energies[energy_index]),
                        "density": float(values[energy_index]),
                    }
                )
        if len(values) == 1:
            candidates.append(
                {
                    "orbital": str(orbital_name),
                    "energy_ev": float(energies[0]),
                    "density": float(values[0]),
                }
            )
    return sorted(candidates, key=lambda row: row["density"], reverse=True)[:top_k]


def _normalized_overlap(
    first: np.ndarray,
    second: np.ndarray,
    energies: np.ndarray,
) -> float:
    first = np.clip(first, 0.0, None)
    second = np.clip(second, 0.0, None)
    first_area = _integral(first, energies)
    second_area = _integral(second, energies)
    if first_area <= 0.0 or second_area <= 0.0:
        return 0.0
    overlap = _integral(
        np.minimum(first / first_area, second / second_area),
        energies,
    )
    return float(max(0.0, min(1.0, overlap)))


def _unique_valid_indices(
    indices1: Iterable[int],
    n_atoms: int,
) -> List[int]:
    return sorted(
        {
            int(index1) - 1
            for index1 in indices1
            if 1 <= int(index1) <= n_atoms
        }
    )


def analyze_binding_pdos_artifact(
    artifact_path: str,
    guest_indices1: Sequence[int],
    contacts: Sequence[Dict[str, Any]],
    *,
    energy_window_ev: Tuple[float, float] = (-10.0, 2.0),
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    path = Path(artifact_path)
    if not path.exists():
        return {"status": "missing_artifact", "artifact": str(path)}

    try:
        with np.load(path, allow_pickle=False) as data:
            energies = np.asarray(data["energies_ev"], dtype=float)
            densities = np.asarray(data["densities"], dtype=float)
            atom_symbols = [str(value) for value in data["atom_symbols"].tolist()]
            orbital_groups = [
                str(value) for value in data["orbital_groups"].tolist()
            ]
            component_names = [
                str(value) for value in data["component_names"].tolist()
            ]
    except (OSError, KeyError, ValueError) as exc:
        return {
            "status": "artifact_parse_failed",
            "artifact": str(path),
            "error": str(exc),
        }

    if densities.ndim != 4 or densities.shape[0] != len(atom_symbols):
        return {
            "status": "invalid_artifact_shape",
            "artifact": str(path),
            "density_shape": list(densities.shape),
            "n_symbols": len(atom_symbols),
        }

    low, high = sorted(float(value) for value in energy_window_ev)
    mask = (energies >= low) & (energies <= high)
    if int(mask.sum()) < 2:
        return {
            "status": "empty_energy_window",
            "artifact": str(path),
            "requested_energy_window_ev": [low, high],
            "available_energy_range_ev": [
                float(energies.min()),
                float(energies.max()),
            ],
        }

    n_atoms = densities.shape[0]
    guest_indices0 = _unique_valid_indices(guest_indices1, n_atoms)
    contact_indices1 = [
        int(row["framework_complex_index"])
        for row in contacts
        if row.get("framework_complex_index") is not None
    ]
    site_indices0 = _unique_valid_indices(contact_indices1, n_atoms)
    if not guest_indices0 or not site_indices0:
        return {
            "status": "missing_atom_selection",
            "artifact": str(path),
            "guest_indices": [int(value) for value in guest_indices1],
            "contact_indices": contact_indices1,
        }

    charge_density = _charge_density(densities, component_names)
    window_energies = energies[mask]
    guest_orbitals = charge_density[guest_indices0][:, mask, :].sum(axis=0)
    site_orbitals = charge_density[site_indices0][:, mask, :].sum(axis=0)
    guest_total = guest_orbitals.sum(axis=1)
    site_total = site_orbitals.sum(axis=1)

    per_orbital_overlap = {
        orbital_name: _normalized_overlap(
            guest_orbitals[:, orbital_index],
            site_orbitals[:, orbital_index],
            window_energies,
        )
        for orbital_index, orbital_name in enumerate(orbital_groups)
    }
    result: Dict[str, Any] = {
        "status": "ok",
        "method": "projected_dos_summary",
        "artifact": str(path),
        "energy_reference": "complex E - E_F",
        "energy_window_ev": [low, high],
        "selection": {
            "guest_complex_indices": [int(index + 1) for index in guest_indices0],
            "guest_species": [atom_symbols[index] for index in guest_indices0],
            "contact_framework_complex_indices": [
                int(index + 1) for index in site_indices0
            ],
            "contact_framework_species": [
                atom_symbols[index] for index in site_indices0
            ],
            "contacts": list(contacts),
        },
        "normalized_spectral_overlap": {
            "total": _normalized_overlap(
                guest_total,
                site_total,
                window_energies,
            ),
            "by_orbital": per_orbital_overlap,
            "definition": (
                "Integral of the pointwise minimum of area-normalized guest and "
                "contact-site PDOS; bounded from 0 to 1."
            ),
        },
        "guest_orbital_fraction": _orbital_fractions(
            guest_orbitals,
            window_energies,
            orbital_groups,
        ),
        "contact_site_orbital_fraction": _orbital_fractions(
            site_orbitals,
            window_energies,
            orbital_groups,
        ),
        "guest_dominant_peaks": _dominant_peaks(
            guest_orbitals,
            window_energies,
            orbital_groups,
        ),
        "contact_site_dominant_peaks": _dominant_peaks(
            site_orbitals,
            window_energies,
            orbital_groups,
        ),
        "limitations": [
            "Spectral overlap is a qualitative descriptor, not proof of a chemical bond.",
            "All comparisons use atom projections from the complex calculation and its Fermi level.",
        ],
    }

    if output_path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        result["summary_json"] = str(destination)
    return result
