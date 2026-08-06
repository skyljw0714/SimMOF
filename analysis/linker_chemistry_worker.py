from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from ase.geometry import find_mic
from ase.io import read, write

try:
    from mofstructure import mofdeconstructor
except ImportError:
    mofdeconstructor = None


FUNCTIONAL_GROUP_SMARTS = {
    "carboxylate": "[CX3](=[OX1])[OX1-,OX1]",
    "carboxylic_acid": "[CX3](=[OX1])[OX2H1]",
    "amine": "[NX3;H0,H1,H2;!$(N-[C,S,P]=[O,S,N])]",
    "primary_amine": "[NX3;H2;!$(N-[C,S,P]=[O,S,N])]",
    "secondary_amine": "[NX3;H1;!$(N-[C,S,P]=[O,S,N])]",
    "hydroxyl": "[OX2H1;!$(O-C=O)]",
    "phenol": "[c][OX2H1]",
    "halogen": "[F,Cl,Br,I]",
    "pyridine_like_n": "[nH0;+0]",
    "pyrrolic_n": "[nH]",
    "carbonyl": "[CX3]=[OX1]",
    "ether": "[OD2]([#6])[#6]",
    "nitrile": "[CX2]#N",
    "nitro": "[$([NX3+](=O)[O-]),$([NX3](=O)=O)]",
    "thiol": "[SX2H1]",
    "thioether": "[SX2]([#6])[#6]",
    "sulfonyl": "[SX4](=O)(=O)",
}


def safe_path_token(text: Any) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text or "structure")).strip("_")
    return token[:120] or "structure"


def formula_from_symbols(symbols: List[str]) -> str:
    counts = Counter(symbols)
    return "".join(f"{el}{n if n > 1 else ''}" for el, n in sorted(counts.items()))


def _rdkit_molecule(smiles: Optional[str]):
    if not smiles:
        return None
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                pass
    return mol


def functional_group_fingerprint_from_smiles(
    smiles: Optional[str],
    formula: str = "",
) -> Dict[str, Any]:
    fingerprint: Dict[str, Any] = {
        "method": "rdkit_named_smarts_substructure_fingerprint",
        "status": "missing_smiles" if not smiles else "ok",
        "source_smiles": smiles,
        "canonical_smiles": None,
        "formula": formula,
        "features": {},
        "present_groups": [],
        "ring_descriptors": {},
        "molecular_descriptors": {},
    }
    if not smiles:
        return fingerprint

    try:
        from rdkit import Chem
        from rdkit.Chem import Lipinski, rdMolDescriptors

        mol = _rdkit_molecule(smiles)
        if mol is None:
            fingerprint["status"] = "rdkit_parse_error"
            return fingerprint

        try:
            fingerprint["canonical_smiles"] = Chem.MolToSmiles(mol)
        except Exception:
            fingerprint["canonical_smiles"] = smiles

        for name, smarts in FUNCTIONAL_GROUP_SMARTS.items():
            query = Chem.MolFromSmarts(smarts)
            matches = mol.GetSubstructMatches(query, uniquify=True) if query is not None else ()
            fingerprint["features"][name] = {
                "present": bool(matches),
                "count": len(matches),
                "matched_atom_indices": [
                    [int(atom_index) + 1 for atom_index in match]
                    for match in matches
                ],
                "smarts": smarts,
            }
            if name == "carboxylate":
                fingerprint["features"][name]["definition"] = (
                    "Carboxylate anion or metal-bond-cut radical oxygen; "
                    "neutral carboxylic acid is reported separately."
                )

        rings = list(mol.GetRingInfo().AtomRings())
        aromatic_rings = [
            ring
            for ring in rings
            if all(mol.GetAtomWithIdx(atom_index).GetIsAromatic() for atom_index in ring)
        ]
        azole_rings = []
        for ring in aromatic_rings:
            if len(ring) != 5:
                continue
            symbols = [mol.GetAtomWithIdx(atom_index).GetSymbol() for atom_index in ring]
            n_count = symbols.count("N")
            hetero_count = sum(symbol not in {"C", "H"} for symbol in symbols)
            if n_count >= 1 and hetero_count >= 2:
                azole_rings.append(ring)

        fingerprint["features"]["azole"] = {
            "present": bool(azole_rings),
            "count": len(azole_rings),
            "matched_atom_indices": [
                [int(atom_index) + 1 for atom_index in ring]
                for ring in azole_rings
            ],
            "definition": (
                "Five-membered aromatic ring containing at least one N and "
                "at least two heteroatoms."
            ),
        }
        fingerprint["ring_descriptors"] = {
            "ring_count": int(rdMolDescriptors.CalcNumRings(mol)),
            "aromatic_ring_count": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
            "heteroaromatic_ring_count": int(rdMolDescriptors.CalcNumAromaticHeterocycles(mol)),
            "azole_ring_count": len(azole_rings),
        }
        fingerprint["molecular_descriptors"] = {
            "h_bond_donor_count": int(Lipinski.NumHDonors(mol)),
            "h_bond_acceptor_count": int(Lipinski.NumHAcceptors(mol)),
            "heteroatom_count": int(Lipinski.NumHeteroatoms(mol)),
        }
        fingerprint["present_groups"] = sorted(
            name
            for name, value in fingerprint["features"].items()
            if value.get("present")
        )
        return fingerprint
    except Exception as exc:
        fingerprint["status"] = "rdkit_error"
        fingerprint["error"] = f"{type(exc).__name__}: {exc}"
        return fingerprint


def functional_tags_from_fingerprint(
    fingerprint: Dict[str, Any],
    formula: str = "",
) -> List[str]:
    tags = list(fingerprint.get("present_groups", []) or [])
    ring_descriptors = fingerprint.get("ring_descriptors", {}) or {}
    if ring_descriptors.get("aromatic_ring_count", 0):
        tags.append("aromatic_ring")
    if not tags and formula:
        tags.append("organic_linker_fragment")
    return sorted(set(tags))


def unit_tags(
    kind: str,
    formula: str,
    smiles: Optional[str],
    sbu_type: Optional[str],
    functional_group_fingerprint: Optional[Dict[str, Any]] = None,
) -> List[str]:
    if kind == "node":
        tags = []
        if sbu_type and sbu_type != "still checking!":
            tags.append(f"{sbu_type} node")
        if re.search(r"(Sc|Ti|V|Cr|Mn|Fe|Co|Ni|Cu|Zn|Y|Zr|Mo|Ru|Rh|Pd|Ag|Cd|Hf|W|Ir|Pt|Au|Hg)", formula):
            tags.append("metal node")
        if not tags:
            tags.append("node building unit")
        return sorted(set(tags))
    fingerprint = functional_group_fingerprint or functional_group_fingerprint_from_smiles(
        smiles,
        formula,
    )
    return functional_tags_from_fingerprint(fingerprint, formula)


def render_smiles_png(smiles: Optional[str], png_path: Path) -> Optional[str]:
    if not smiles:
        return None
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
        png_path.parent.mkdir(parents=True, exist_ok=True)
        Draw.MolToFile(mol, str(png_path), size=(480, 360))
        return str(png_path)
    except Exception:
        return None


def map_unit_atoms_to_source(unit_atoms, source_atoms, source_indices1: List[int], cutoff_A: float = 0.6) -> Dict[str, Any]:
    info = getattr(unit_atoms, "info", {}) or {}
    atom_indices_mapping = info.get("atom_indices_mapping")
    if atom_indices_mapping:
        mapped = []
        for group in atom_indices_mapping:
            for clean_idx0 in group:
                try:
                    mapped.append(int(source_indices1[int(clean_idx0)]))
                except Exception:
                    continue
        mapped = sorted(set(mapped))
        return {
            "source_atom_indices": mapped,
            "source_atom_mapping_count": len(mapped),
            "source_atom_mapping_complete": len(mapped) >= len(unit_atoms),
            "source_atom_mapping_max_distance_A": None,
            "source_atom_mapping_cutoff_A": None,
            "source_atom_mapping_method": "mofstructure atom_indices_mapping",
        }

    used = set()
    mapped = []
    distances = []
    source_symbols = source_atoms.get_chemical_symbols()

    for unit_atom in unit_atoms:
        best = None
        for src_i0, src_symbol in enumerate(source_symbols):
            if src_i0 in used or src_symbol != unit_atom.symbol:
                continue
            delta = np.array(unit_atom.position, dtype=float) - np.array(source_atoms[src_i0].position, dtype=float)
            try:
                mic_delta, dist = find_mic(delta, source_atoms.cell, pbc=source_atoms.pbc)
                d = float(dist)
            except Exception:
                d = float(np.linalg.norm(delta))
            if best is None or d < best[0]:
                best = (d, src_i0)
        if best is None:
            continue
        d, src_i0 = best
        if d <= cutoff_A:
            used.add(src_i0)
            mapped.append(int(source_indices1[src_i0]))
            distances.append(d)

    return {
        "source_atom_indices": mapped,
        "source_atom_mapping_count": len(mapped),
        "source_atom_mapping_complete": len(mapped) == len(unit_atoms),
        "source_atom_mapping_max_distance_A": max(distances) if distances else None,
        "source_atom_mapping_cutoff_A": cutoff_A,
        "source_atom_mapping_method": "coordinate_back_mapping",
    }


def summarize_building_unit(
    atoms,
    kind: str,
    index: int,
    unit_dir: Path,
    source_atoms=None,
    source_indices1: Optional[List[int]] = None,
) -> Dict[str, Any]:
    unit_dir.mkdir(parents=True, exist_ok=True)
    formula = formula_from_symbols(atoms.get_chemical_symbols())
    base = f"{kind}_{index:02d}_{formula or 'unit'}"
    xyz_path = unit_dir / f"{base}.xyz"
    cif_path = unit_dir / f"{base}.cif"

    exported: Dict[str, Optional[str]] = {"xyz": None, "cif": None}
    try:
        write(str(xyz_path), atoms, format="xyz")
        exported["xyz"] = str(xyz_path)
    except Exception:
        pass
    try:
        write(str(cif_path), atoms, format="cif")
        exported["cif"] = str(cif_path)
    except Exception:
        pass

    info = getattr(atoms, "info", {}) or {}
    smiles = info.get("smi")
    inchikey = info.get("inchikey")
    sbu_type = info.get("sbu_type")
    png = render_smiles_png(smiles, unit_dir / f"{base}.png")
    functional_group_fingerprint = (
        functional_group_fingerprint_from_smiles(smiles, formula)
        if kind in {"linker", "ligand"}
        else {
            "method": "rdkit_named_smarts_substructure_fingerprint",
            "status": "not_applicable_to_node",
            "features": {},
            "present_groups": [],
        }
    )

    record = {
        "kind": kind,
        "index": index,
        "n_atoms": len(atoms),
        "formula": formula,
        "smiles": smiles,
        "inchikey": inchikey,
        "sbu_type": sbu_type,
        "point_count": len(info.get("point_of_extension", [])),
        "points_of_extension": [int(x) for x in info.get("point_of_extension", [])],
        "functional_tags": unit_tags(
            kind,
            formula,
            smiles,
            sbu_type,
            functional_group_fingerprint=functional_group_fingerprint,
        ),
        "functional_group_fingerprint": functional_group_fingerprint,
        "exports": {
            "structure_xyz": exported["xyz"],
            "structure_cif": exported["cif"],
            "smiles_png": png,
        },
        "vlm_hint": (
            "Use the PNG for a 2D chemical view when available; use XYZ/CIF for the "
            "actual disconnected building-unit geometry and extension points."
        ),
    }
    if source_atoms is not None and source_indices1 is not None:
        record.update(map_unit_atoms_to_source(atoms, source_atoms, source_indices1))
    return record


def analyze_cif(cif_path: Path, output_root: Path) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "status": "ok",
        "cif_path": str(cif_path),
        "mof": cif_path.stem,
    }
    try:
        atoms = read(str(cif_path))
    except Exception:
        fixed_cif_path = output_root / f"{safe_path_token(cif_path.stem)}_cif_loop_fixed.cif"
        text = cif_path.read_text(encoding="utf-8", errors="ignore")
        fixed = re.sub(
            r"((?:^\s*'[^']+'\s*$\n)+)(\s*_atom_site_)",
            r"\1\nloop_\n\2",
            text,
            flags=re.MULTILINE,
        )
        fixed_cif_path.write_text(fixed, encoding="utf-8")
        atoms = read(str(fixed_cif_path))
        cif_path = fixed_cif_path

    try:
        if mofdeconstructor is None:
            raise ImportError(
                "mofstructure is required for CIF node/linker decomposition"
            )
        keep = mofdeconstructor.remove_unbound_guest(atoms)
        clean = atoms[keep] if isinstance(keep, (list, tuple)) else keep
        source_indices1 = [int(i) + 1 for i in keep] if isinstance(keep, (list, tuple)) else list(range(1, len(clean) + 1))
        ret = mofdeconstructor.secondary_building_units(clean)
        if len(ret) == 4:
            components, breakpoints, porphyrin_checker, all_regions = ret
        elif len(ret) == 5:
            _, components, breakpoints, porphyrin_checker, all_regions = ret
        else:
            raise ValueError(f"Unexpected secondary_building_units return length: {len(ret)}")

        metal_sbus, linkers, regions = mofdeconstructor.find_unique_building_units(
            components,
            breakpoints,
            clean,
            porphyrin_checker,
            all_regions,
            cheminfo=True,
        )

        mof_dir = output_root / safe_path_token(cif_path.stem)
        if mof_dir.exists():
            mof_dir = output_root / safe_path_token(
                cif_path.parent.name + "_" + cif_path.stem
            )

        ligand_records = []
        ligand_error = None
        try:
            ligand_ret = mofdeconstructor.ligands_and_metal_clusters(clean)
            if len(ligand_ret) == 4:
                (
                    ligand_components,
                    ligand_breakpoints,
                    ligand_porhyrin_checker,
                    ligand_regions,
                ) = ligand_ret
            elif len(ligand_ret) == 5:
                (
                    _,
                    ligand_components,
                    ligand_breakpoints,
                    ligand_porhyrin_checker,
                    ligand_regions,
                ) = ligand_ret
            else:
                raise ValueError(
                    "Unexpected ligands_and_metal_clusters return length: "
                    f"{len(ligand_ret)}"
                )
            _, ligands, _ = mofdeconstructor.find_unique_building_units(
                ligand_components,
                ligand_breakpoints,
                clean,
                ligand_porhyrin_checker,
                ligand_regions,
                cheminfo=True,
            )
            ligand_records = [
                summarize_building_unit(
                    unit,
                    "ligand",
                    i + 1,
                    mof_dir / "ligands",
                    source_atoms=clean,
                    source_indices1=source_indices1,
                )
                for i, unit in enumerate(ligands)
            ]
        except Exception as exc:
            ligand_error = f"{type(exc).__name__}: {exc}"

        linker_records = [
            summarize_building_unit(
                unit,
                "linker",
                i + 1,
                mof_dir / "linkers",
                source_atoms=clean,
                source_indices1=source_indices1,
            )
            for i, unit in enumerate(linkers)
        ]
        node_records = [
            summarize_building_unit(
                unit,
                "node",
                i + 1,
                mof_dir / "nodes",
                source_atoms=clean,
                source_indices1=source_indices1,
            )
            for i, unit in enumerate(metal_sbus)
        ]

        record.update(
            {
                "n_atoms": len(atoms),
                "n_atoms_after_guest_removal": len(clean),
                "guest_removed_atoms": len(atoms) - len(clean),
                "n_components": len(components),
                "n_breakpoints": len(breakpoints),
                "output_dir": str(mof_dir),
                "linkers": linker_records,
                "ligands": ligand_records,
                "nodes": node_records,
                "notes": [
                    "Linker fragments are disconnected building units, so carboxylate atoms may be assigned to the node side after cutting.",
                    "Ligands are cut at metal-ligand bonds and are preferred over SBU linkers for functional-group fingerprints.",
                    "Linker functional groups are matched by RDKit SMARTS against the exported linker SMILES.",
                    "SMARTS matches describe the disconnected linker representation and should be checked when a cutting point changes protonation or bond order.",
                ],
            }
        )
        if ligand_error:
            record["ligand_decomposition_warning"] = ligand_error
    except Exception as exc:
        record.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        payload = json.load(f)

    output_root = Path(payload["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    structures = [
        analyze_cif(Path(path), output_root)
        for path in payload.get("cif_paths", [])
    ]

    summary = {
        "method": "linker_chemistry_analysis",
        "status": "ok" if structures else "no_cif_paths_found",
        "n_structures": len(structures),
        "output_dir": str(output_root),
        "summary_json": str(args.output),
        "structures": structures,
        "vlm_ready": True,
        "vlm_usage": (
            "Pass each linker/node PNG plus the corresponding XYZ/CIF and descriptor record "
            "to a VLM/LLM. Let the model explain chemistry, but keep SMILES/InChIKey and "
            "extension points as the source of truth."
        ),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
