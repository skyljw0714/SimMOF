
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from ase import Atoms
from ase.io import read
from ase.neighborlist import natural_cutoffs, neighbor_list


DEFAULT_DEFORMATION_THRESHOLD_PERCENT = 20.0


def _bond_pairs(atoms: Atoms) -> set:
    if len(atoms) < 2:
        return set()
    try:
        cutoffs = natural_cutoffs(atoms, mult=1.20)
        left, right = neighbor_list("ij", atoms, cutoffs)
    except Exception:
        return set()
    return {
        (min(int(i), int(j)), max(int(i), int(j)))
        for i, j in zip(left, right)
        if int(i) != int(j)
    }


def _cell_metrics(initial: Atoms, final: Atoms) -> Dict[str, Any]:
    cell0 = np.asarray(initial.cell, dtype=float)
    cell1 = np.asarray(final.cell, dtype=float)
    lengths0 = np.linalg.norm(cell0, axis=1)
    lengths1 = np.linalg.norm(cell1, axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        length_changes = np.where(
            lengths0 > 1.0e-12,
            np.abs(lengths1 - lengths0) / lengths0 * 100.0,
            0.0,
        )

    volume0 = abs(float(np.linalg.det(cell0)))
    volume1 = abs(float(np.linalg.det(cell1)))
    volume_change = (
        abs(volume1 - volume0) / volume0 * 100.0
        if volume0 > 1.0e-12
        else 0.0
    )

    principal_strains: List[float] = []
    try:
        deformation_gradient = np.linalg.solve(cell0, cell1)
        stretches = np.linalg.svd(deformation_gradient, compute_uv=False)
        principal_strains = [float(abs(value - 1.0) * 100.0) for value in stretches]
    except np.linalg.LinAlgError:
        pass

    return {
        "initial_lengths_A": [float(value) for value in lengths0],
        "final_lengths_A": [float(value) for value in lengths1],
        "length_change_percent": [float(value) for value in length_changes],
        "max_length_change_percent": float(np.max(length_changes)) if length_changes.size else 0.0,
        "initial_volume_A3": volume0,
        "final_volume_A3": volume1,
        "volume_change_percent": float(volume_change),
        "principal_strain_percent": principal_strains,
        "max_principal_strain_percent": max(principal_strains, default=0.0),
    }


def _internal_displacement_metrics(initial: Atoms, final: Atoms) -> Dict[str, float]:
    scaled0 = np.asarray(initial.get_scaled_positions(wrap=False), dtype=float)
    scaled1 = np.asarray(final.get_scaled_positions(wrap=False), dtype=float)
    delta = scaled1 - scaled0
    pbc = np.asarray(initial.pbc | final.pbc, dtype=bool)
    delta[:, pbc] -= np.round(delta[:, pbc])
    if len(delta):
        drift = np.median(delta, axis=0)
        delta -= drift
        delta[:, pbc] -= np.round(delta[:, pbc])
    cartesian = delta @ np.asarray(initial.cell, dtype=float)
    norms = np.linalg.norm(cartesian, axis=1)
    return {
        "translation_removed_rms_A": (
            float(np.sqrt(np.mean(np.square(norms)))) if norms.size else 0.0
        ),
        "translation_removed_max_A": float(np.max(norms)) if norms.size else 0.0,
    }


def analyze_structure_deformation(
    initial_path: str,
    final_path: str,
    threshold_percent: float = DEFAULT_DEFORMATION_THRESHOLD_PERCENT,
) -> Dict[str, Any]:
    initial_file = Path(initial_path)
    final_file = Path(final_path)
    base: Dict[str, Any] = {
        "initial_structure": str(initial_file),
        "final_structure": str(final_file),
        "threshold_percent": float(threshold_percent),
        "threshold_definition": (
            "maximum of bonded-pair distance change, principal cell strain, "
            "and cell-volume change"
        ),
    }
    if not initial_file.is_file() or not final_file.is_file():
        base.update(
            {
                "status": "missing_structure",
                "threshold_exceeded": False,
                "missing": [
                    str(path)
                    for path in (initial_file, final_file)
                    if not path.is_file()
                ],
            }
        )
        return base

    try:
        initial = read(str(initial_file))
        final = read(str(final_file))
    except Exception as exc:
        base.update(
            {
                "status": "parse_failed",
                "threshold_exceeded": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }
        )
        return base

    symbols0 = initial.get_chemical_symbols()
    symbols1 = final.get_chemical_symbols()
    if len(initial) != len(final) or symbols0 != symbols1:
        base.update(
            {
                "status": "atom_order_mismatch",
                "threshold_exceeded": False,
                "initial_atom_count": len(initial),
                "final_atom_count": len(final),
            }
        )
        return base

    pairs = _bond_pairs(initial) | _bond_pairs(final)
    pair_changes: List[float] = []
    largest_pair = None
    for left, right in sorted(pairs):
        distance0 = float(initial.get_distance(left, right, mic=True))
        distance1 = float(final.get_distance(left, right, mic=True))
        if distance0 <= 1.0e-10:
            continue
        change = abs(distance1 - distance0) / distance0 * 100.0
        pair_changes.append(change)
        if largest_pair is None or change > largest_pair["change_percent"]:
            largest_pair = {
                "indices_1based": [left + 1, right + 1],
                "species": [symbols0[left], symbols0[right]],
                "initial_distance_A": distance0,
                "final_distance_A": distance1,
                "change_percent": float(change),
            }

    max_pair_change = max(pair_changes, default=0.0)
    rms_pair_change = (
        float(np.sqrt(np.mean(np.square(pair_changes)))) if pair_changes else 0.0
    )
    cell = _cell_metrics(initial, final)
    displacement = _internal_displacement_metrics(initial, final)
    overall = max(
        max_pair_change,
        float(cell["max_principal_strain_percent"]),
        float(cell["volume_change_percent"]),
    )
    base.update(
        {
            "status": "ok",
            "atom_count": len(initial),
            "bonded_pair_count": len(pair_changes),
            "bonded_pair_rms_change_percent": rms_pair_change,
            "bonded_pair_max_change_percent": float(max_pair_change),
            "largest_bonded_pair_change": largest_pair,
            "cell": cell,
            "displacement_diagnostic": displacement,
            "overall_deformation_percent": float(overall),
            "threshold_exceeded": bool(overall >= float(threshold_percent)),
        }
    )
    return base


def _periodic_distance_matrix(reference: Atoms, candidate: Atoms) -> np.ndarray:
    reference_scaled = np.asarray(reference.get_scaled_positions(wrap=False), dtype=float)
    candidate_scaled = np.asarray(candidate.get_scaled_positions(wrap=False), dtype=float)
    delta = reference_scaled[:, None, :] - candidate_scaled[None, :, :]
    pbc = np.asarray(reference.pbc | candidate.pbc, dtype=bool)
    delta[:, :, pbc] -= np.round(delta[:, :, pbc])
    return np.linalg.norm(delta @ np.asarray(candidate.cell, dtype=float), axis=2)


def build_frozen_fragments(
    optimized_mof_path: str,
    complex_initial_path: str,
    complex_final_path: str,
    match_tolerance_A: float = 1.50,
) -> Tuple[Atoms, Atoms, Dict[str, Any]]:
    from scipy.optimize import linear_sum_assignment

    mof = read(str(optimized_mof_path))
    complex_initial = read(str(complex_initial_path))
    complex_final = read(str(complex_final_path))
    if len(complex_initial) != len(complex_final):
        raise ValueError("complex POSCAR and CONTCAR atom counts differ")
    if complex_initial.get_chemical_symbols() != complex_final.get_chemical_symbols():
        raise ValueError("complex POSCAR and CONTCAR atom ordering differs")
    if len(complex_initial) <= len(mof):
        raise ValueError("optimized complex does not contain any guest atoms")

    mof_symbols = mof.get_chemical_symbols()
    complex_symbols = complex_initial.get_chemical_symbols()
    distance_matrix = _periodic_distance_matrix(mof, complex_initial)
    framework_by_mof_index: List[int] = [-1] * len(mof)
    match_distances: List[float] = []

    for symbol in sorted(set(mof_symbols)):
        mof_indices = [i for i, value in enumerate(mof_symbols) if value == symbol]
        complex_indices = [
            i for i, value in enumerate(complex_symbols) if value == symbol
        ]
        if len(complex_indices) < len(mof_indices):
            raise ValueError(f"complex has too few {symbol} atoms for the MOF")
        block = distance_matrix[np.ix_(mof_indices, complex_indices)]
        rows, columns = linear_sum_assignment(block)
        for row, column in zip(rows, columns):
            mof_index = mof_indices[int(row)]
            complex_index = complex_indices[int(column)]
            distance = float(block[int(row), int(column)])
            framework_by_mof_index[mof_index] = complex_index
            match_distances.append(distance)

    if any(index < 0 for index in framework_by_mof_index):
        raise ValueError("not every MOF atom could be mapped into the complex")
    max_match = max(match_distances, default=0.0)
    if max_match > float(match_tolerance_A):
        raise ValueError(
            f"MOF-to-complex mapping distance {max_match:.3f} A exceeds "
            f"the {match_tolerance_A:.3f} A tolerance"
        )

    framework_set = set(framework_by_mof_index)
    guest_indices = [
        index for index in range(len(complex_initial)) if index not in framework_set
    ]
    if not guest_indices:
        raise ValueError("no unmatched guest atoms remain after framework mapping")

    framework = complex_final[framework_by_mof_index]
    guest = complex_final[guest_indices]
    for fragment in (framework, guest):
        fragment.set_cell(complex_final.cell)
        fragment.set_pbc(complex_final.pbc)

    mapping = {
        "status": "ok",
        "method": "species-wise Hungarian mapping from optimized MOF to complex POSCAR",
        "optimized_mof": str(optimized_mof_path),
        "complex_initial": str(complex_initial_path),
        "complex_final": str(complex_final_path),
        "match_tolerance_A": float(match_tolerance_A),
        "max_match_distance_A": float(max_match),
        "framework_atom_count": len(framework),
        "guest_atom_count": len(guest),
        "framework_complex_indices_1based": [
            int(index + 1) for index in framework_by_mof_index
        ],
        "guest_complex_indices_1based": [int(index + 1) for index in guest_indices],
    }
    return framework, guest, mapping


def compact_magmom(values: Sequence[float]) -> str:
    runs: List[List[float]] = []
    for value in values:
        numeric = float(value)
        if runs and abs(runs[-1][0] - numeric) < 1.0e-12:
            runs[-1][1] += 1
        else:
            runs.append([numeric, 1])
    return " ".join(
        f"{int(count)}*{value:g}" if int(count) > 1 else f"{value:g}"
        for value, count in runs
    )
