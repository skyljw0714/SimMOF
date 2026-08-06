#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.assign_charges import (
    delocalisedLBO,
    ringVBOs,
    iVBS_Oxidation_Contrib,
    redundantAON,
    get_unique_sites,
    get_metal_sites,
    get_binding_sites,
)
from modules.solvent_analysis import (
    get_rAON_atomlabels,
    remove_free_solvent,
    get_oxo,
    remove_metals,
    define_solvents,
    check_solvent,
    get_solvents_to_remove,
)
from modules.cif_manipulation import readentry, get_coordinates
from modules.outputs import output_cif


def run_samosa_single(
    cif_path: str,
    keep_bound: bool = False,
    keep_oxo: bool = False,
    removable_denticity: int = 1,
) -> None:
    cif_path = os.path.abspath(cif_path)
    cif_dir = os.path.dirname(cif_path)
    cif_name = os.path.basename(cif_path)

    print(f"[SAMOSA] Processing {cif_name} ...")

    cif = readentry(cif_path)
    mol = cif.molecule
    asymmol = cif.asymmetric_unit_molecule

    uniquesites = get_unique_sites(mol, asymmol)
    metalsites = get_metal_sites(uniquesites)
    _, binding_pairs = get_binding_sites(metalsites, uniquesites)

    dVBO = delocalisedLBO(mol)
    rVBO = ringVBOs(mol)
    AON = iVBS_Oxidation_Contrib(uniquesites, rVBO, dVBO)
    rAON = redundantAON(AON, mol)
    rAON_atomlabels = get_rAON_atomlabels(rAON)

    molecule_work, free_solvents, counterions, _ = remove_free_solvent(
        mol, rAON_atomlabels
    )

    if not keep_bound:
        oxo_mols, _ = get_oxo(uniquesites, cif_path, keep_oxo)
        molecule_no_metals = remove_metals(molecule_work)
        solvent_mols, _ = define_solvents(molecule_no_metals, rAON_atomlabels)
        solvent_mols_checked, _ = check_solvent(
            solvent_mols, binding_pairs, uniquesites, removable_denticity
        )
        solvents_to_remove = get_solvents_to_remove(
            solvent_mols_checked, free_solvents, counterions, oxo_mols
        )
    else:
        solvents_to_remove = free_solvents + counterions

    if not solvents_to_remove:
        print(f"[SAMOSA] No solvent detected in {cif_name}, keeping original.")
        return

    solvent_coordinates = get_coordinates(mol, solvents_to_remove)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_cif(tmpdir, cif_name, solvent_coordinates, cif_dir)
        cleaned = os.path.join(tmpdir, "MOFs_removed_solvent", cif_name)
        if os.path.exists(cleaned):
            shutil.copy2(cleaned, cif_path)
            print(
                f"[SAMOSA] Cleaned {cif_name}: "
                f"{len(solvents_to_remove)} solvent atoms removed."
            )
        else:
            print(f"[SAMOSA] Warning: cleaned CIF not produced, keeping original.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("cif_file")
    parser.add_argument("--keep_oxo", action="store_true", default=False)
    args = parser.parse_args()
    run_samosa_single(args.cif_file, keep_oxo=args.keep_oxo)
