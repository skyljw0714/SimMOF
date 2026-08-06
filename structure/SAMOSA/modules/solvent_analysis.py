from __future__ import annotations

import os

from ccdc.molecule import Molecule, Atom
from ccdc.io import EntryReader

from collections import defaultdict

from multiprocessing import current_process as cpr

from .log_utils import get_logger


def get_rAON_atomlabels(rAON_old: dict[Atom, float]) -> dict[str, float]:
    rAON_atomlabels = {}
    for key, value in rAON_old.items():
        label = key.label
        rAON_atomlabels[label] = value
    return rAON_atomlabels


def remove_free_solvent(
    molecule: Molecule, rAON_atomlabels: dict[str, float]
) -> tuple[
    Molecule, list[str], list[str], dict[str, list | int | str]
]:
    structure = molecule.copy()
    free_solvents = []
    counterions = []
    free_solvents_output = []
    counterions_output = []
    charge_removed = 0
    free_solvent_flag = False
    counterions_flag = False
    metal_counterion_flag = "."
    huge_counterion_flag = "."

    for component in structure.components:
        flag_metal = False
        for atom in component.atoms:
            if atom.is_metal is True:
                flag_metal = True
                break
        labels = []
        for atom in component.atoms:
            labels.append(atom.atomic_symbol)

        if flag_metal is True:
            if component.is_polymeric is False:
                metal_counterion_flag = "TRUE"
                counterion_unit = []
                charge_unit = get_component_charge(rAON_atomlabels, component)

                if charge_unit > 10:
                    huge_counterion_flag = True

                charge_removed += charge_unit
                for atom in component.atoms:
                    counterions.append(atom.label)
                    counterion_unit.append(atom.label)
                counterions_output.append(counterion_unit)
        else:
            free_output_unit = []
            counterion_unit = []
            charge_unit = get_component_charge(rAON_atomlabels, component)

            if charge_unit == 0:
                for atom in component.atoms:
                    free_solvents.append(atom.label)
                    free_output_unit.append(atom.label)
                free_solvents_output.append(free_output_unit)
            else:
                neutral_units = [
                    ["O"],
                    ["N"],
                    ["H"],
                    ["O", "O"],
                    ["N", "N"],
                    ["O", "O", "O"],
                    ["D"],
                ]
                flag_neutral = False

                if len(labels) <= 3:
                    for unit in neutral_units:
                        if labels == unit:
                            flag_neutral = True
                            break
                    if len(labels) == 2:
                        if "O" in labels and "H" in labels:
                            flag_neutral = True
                        elif "O" in labels and "C" in labels:
                            flag_neutral = True
                    elif len(labels) == 3:
                        labels.sort()
                        if labels == ["C", "O", "O"]:
                            flag_neutral = True

                if flag_neutral is False:
                    charge_removed += charge_unit
                    for atom in component.atoms:
                        counterions.append(atom.label)
                        counterion_unit.append(atom.label)
                    counterions_output.append(counterion_unit)
                else:
                    for atom in component.atoms:
                        free_solvents.append(atom.label)
                        free_output_unit.append(atom.label)
                    free_solvents_output.append(free_output_unit)


    if len(free_solvents) > 0:
        free_solvent_flag = True
    if len(counterions) > 0:
        counterions_flag = True

    for atom in structure.atoms:
        if atom.label in free_solvents or atom.label in counterions:
            structure.remove_atom(atom)

    statistics_output = {
        "free_solvents_output": free_solvents_output,
        "counterions_output": counterions_output,
        "charge_removed": charge_removed,
        "metal_counterion_flag": metal_counterion_flag,
        "huge_counterion_flag": huge_counterion_flag,
        "free_solvent_flag": free_solvent_flag,
        "counterions_flag": counterions_flag,
    }

    return structure, free_solvents, counterions, statistics_output


def get_component_charge(rAON: dict[str, float], component: Molecule) -> int | float:
    charge = 0
    for atom in component.atoms:
        key = atom.label
        charge_unit = rAON.get(key)
        charge += charge_unit
    return charge


def get_refcode(file: str) -> str:

    ref_code = file.replace(".cif", "")
    if "\\" or "/" in ref_code:
        ref_code = os.path.basename(ref_code)
    if "_" in ref_code:
        pos = ref_code.find("_")
        ref_code = ref_code[:pos]
        return ref_code
    return ref_code


def check_entry(file: str) -> bool | str:
    entry_oxo = False

    ref_code = get_refcode(file)

    try:
        csd_reader = EntryReader("CSD")
        mof = csd_reader.entry(ref_code)

        name = mof.chemical_name
        if "oxo" or "oxa" in name:
            oxo_names = [
                "-oxo-",
                "-dioxo-",
                "-trioxo-",
                "-tetraoxo-",
                "-pentaoxo-",
                "-hexaoxo-",
                "-heptaoxo-",
                "-octaoxo-",
                "-nonaoxo-",
                "-decaoxo-",
                "-undecaoxo-",
                "-undecaoxa-",
                "-dodecaoxo-",
                "-bis(oxo-",
                "-icosaoxo-",
                "-tetracosaoxo-",
            ]
            for oxo_name in oxo_names:
                if oxo_name in name:
                    entry_oxo = True
    except:
        get_logger(cpr().name).warning(
            "refcode %s not found in CCDC. All terminal-O will be removed if --keep_oxo is not specified. Check your cif file naming."
            % ref_code
        )
        entry_oxo = "FAILED REFCODE"

    return entry_oxo


def get_oxo(
    molecule: Molecule, file: str, keep_oxo: bool
) -> tuple[list[str], dict[str, list | int | str]]:

    terminal_oxo_flag = False
    oxo_OH = False

    most_probable_oxo = ["W", "U", "Mo", "V", "Np", "Ti", "Cr"]
    terminal_oxo = []

    oxo_present = check_entry(file)

    possible_oxo = []
    corresponding_metals = []
    for atom in molecule:
        if atom.atomic_symbol == "O":
            if len(atom.neighbours) == 1:
                bond = atom.bonds
                if bond[0].bond_type == "Single":
                    probable_metal = atom.neighbours
                    if probable_metal[0].is_metal is True:
                        possible_oxo.append(atom)
                        corresponding_metals.append(probable_metal[0])

                        if probable_metal[0].atomic_symbol == ("Zr" or "U"):
                            oxo_OH = True

    """If the entry says that the oxo-ligands are present:
    The metals list is checked for the presence of most probable ones.
    If the most probable ones are present, the oxo-ligands on them are kept and all
    the other ones are removed.
    If there are no most probable metals, all the oxo-ligands are kept.

    If the entry says that the oxo-ligands are not present:
    All the oxo-ligands are removed
    
    If refcode is not found in CCDC:
    All oxo-ligands are removed"""

    if len(possible_oxo) != 0:
        flag_probable = False
        if oxo_present:
            for metal in corresponding_metals:
                if metal.atomic_symbol in most_probable_oxo:
                    flag_probable = True
                    break
            if flag_probable is True:
                for index, metal in enumerate(corresponding_metals):
                    if metal.atomic_symbol not in most_probable_oxo:
                        terminal_oxo.append(possible_oxo[index].label)
            else:
                for atom in possible_oxo:
                    terminal_oxo.append(atom.label)
        else:
            for atom in possible_oxo:
                terminal_oxo.append(atom.label)

        if len(terminal_oxo) > 0:
            terminal_oxo_flag = True

    if keep_oxo:
        terminal_oxo = []

    statistics_output = {
        "entry_oxo": oxo_present,
        "terminal_oxo_flag": terminal_oxo_flag,
        "oxo_OH": oxo_OH,
        "oxo_mols": terminal_oxo,
    }

    return terminal_oxo, statistics_output


def remove_metals(work_mol: Molecule) -> Molecule:
    for atom in work_mol.atoms:
        if atom.is_metal is True:
            work_mol.remove_atom(atom)
    return work_mol


def define_solvents(
    mol_no_metals: Molecule, charges: dict[str, float]
) -> tuple[list[str], dict[str, list | int | str]]:
    OH_removed = "."

    mol_fragments = []
    mol_fragments_list = mol_no_metals.components

    for component in mol_fragments_list:
        if len(component.atoms) > 1:
            mol_fragments.append(component)
    not_charged = []
    for fragment in mol_fragments:
        charge = 0
        for atom in fragment.atoms:
            key = atom.label
            charge_unit = charges.get(key)
            charge += charge_unit
        if charge == 0:
            not_charged.append(fragment)
        else:
            if len(fragment.atoms) == 2:
                atoms = fragment.atoms
                atoms_labels = [atom.atomic_symbol for atom in atoms]
                if "O" in atoms_labels and "H" in atoms_labels:
                    OH_removed = True
                    not_charged.append(fragment)

    statistics_output = {"OH_removed": OH_removed}

    return not_charged, statistics_output


def check_solvent(
    solvents: list[str],
    binding_pairs: dict[Atom, list[Atom]],
    unique_sites: list[Atom],
    max_denticity: int,
) -> tuple[list[str], dict[str, list | int | str]]:

    solvents_final = []
    solvent_type_label = {}

    solvents_for_output = []
    flag_aromatic = "."
    flag_double = "."
    bound_solvent_flag = False

    binding_sites_labels = [site.label for site, metals in binding_pairs.items()]


    binding_pair_labels = defaultdict(list)
    bridging = []
    for atom in unique_sites:
        if not atom.is_metal:
            neighbours = atom.neighbours
            n_count = 0
            for unit in neighbours:
                if unit.is_metal is True:
                    n_count += 1
                    binding_pair_labels[atom.label].append(
                        unit.label + str(unit.coordinates[:3])
                    )
            if n_count > 1:
                bridging.append(atom.label)

    for solvent in solvents:
        solvent_labels = []

        if len(solvent.atoms) == 2:
            symbols = [atom.atomic_symbol for atom in solvent.atoms]
            if "C" in symbols and "O" in symbols:
                continue

        atoms_list = solvent.atoms
        for i in range(len(atoms_list)):
            label = atoms_list[i].label
            solvent_labels.append(label)
            solvent_type_label[label] = atoms_list[i]

        uniq_metal_bound = []
        for i in range(len(solvent_labels)):
            if solvent_labels[i] in binding_sites_labels:
                uniq_metal_bound.extend(binding_pair_labels[solvent_labels[i]])
  
        if (len(uniq_metal_bound) <= max_denticity) & (len(set(uniq_metal_bound)) == 1):
            flag_bridging = False
            for atom in solvent_labels:
                if atom in bridging:
                    flag_bridging = True

            if flag_bridging is False:
                solvents_for_output.append(solvent_labels)
                for label in solvent_labels:
                    solvents_final.append(label)

    for solvent in solvents_for_output:
        for atom in solvent:
            tmp_sites_labels = set(binding_sites_labels)
            if atom in tmp_sites_labels:
                atom_obj = solvent_type_label.get(atom)
                for bond in atom_obj.bonds:
                    if bond.bond_type == "Aromatic":
                        flag_aromatic = True
                        break
                    if bond.bond_type == "Double":
                        flag_double = True
                        break
        if flag_double == "True" and flag_aromatic == "True":
            break

    if len(solvents_final) != 0:
        bound_solvent_flag = True

    statistics_output = {
        "solvents_for_output": solvents_for_output,
        "flag_aromatic": flag_aromatic,
        "flag_double": flag_double,
        "bound_solvent_flag": bound_solvent_flag,
    }

    return solvents_final, statistics_output


def get_solvents_to_remove(
    solvent_mols: list[str],
    free_mols: list[str],
    counterion_mols: list[str],
    oxo_mols: list[str],
) -> list[str]:
    final_solvents = []
    final_solvents.extend(solvent_mols)
    final_solvents.extend(free_mols)
    final_solvents.extend(counterion_mols)
    final_solvents.extend(oxo_mols)
    return final_solvents

