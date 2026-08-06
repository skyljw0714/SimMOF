from __future__ import annotations

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import mendeleev
from ccdc.molecule import Atom, Molecule
from ccdc.descriptors import MolecularDescriptors

from collections import defaultdict
from multiprocessing import current_process as cpr

from .log_utils import get_logger


def get_unique_sites(mole: Molecule, asymmole: Molecule) -> list[Atom]:
    uniquesites = []
    labels = []
    asymmcoords = []
    molecoords = []
    duplicates = []
    for atom in asymmole.atoms:
        asymmcoords.append(
            atom.coordinates
        )
    for atom in mole.atoms:
        if (
            atom.coordinates in asymmcoords
        ):
            if (
                atom.coordinates not in molecoords
            ):
                if atom.label not in labels:
                    uniquesites.append(atom)
                    molecoords.append(atom.coordinates)
                    labels.append(atom.label)
                else:
                    duplicates.append(atom)
            else:
                duplicates.append(atom)
    if len(duplicates) >= 1:
        for datom in duplicates:
            for atom in uniquesites:
                if any(
                    [
                        (datom.coordinates == atom.coordinates),
                        (datom.label == atom.label),
                    ]
                ):
                    if datom.atomic_symbol == atom.atomic_symbol:
                        if len(datom.neighbours) > len(atom.neighbours):
                            uniquesites.remove(atom)
                            uniquesites.append(datom)
                    elif datom.label not in labels:
                        uniquesites.append(datom)
    return uniquesites


def get_metal_sites(sites: list[Atom]) -> list[Atom]:
    metalsites = []
    for site in sites:
        if site.is_metal is True:
            metalsites.append(site)
    return metalsites


def get_binding_sites(
    metalsites: list[Atom], uniquesites: list[Atom]
) -> tuple[list[Atom], dict[Atom, list[Atom]]]:
    binding_sites = []
    binding_pairs = defaultdict(list)
    for metal in metalsites:
        for ligand in metal.neighbours:
            for site in uniquesites:
                if ligand.label == site.label:
                    binding_sites.append(site)
                    binding_pairs[site].append(metal)
    return binding_sites, binding_pairs


def ringVBOs(mole: Molecule) -> dict[int, int]:
    ringVBO = {}
    unassigned = mole.atoms
    ringcopy = mole.copy()
    oncycle_atoms = []
    offcycle_atoms = []
    oncycle_labels = []
    offcycle_labels = []
    for atom in ringcopy.atoms:
        if atom.is_metal:
            ringcopy.remove_atom(atom)
    for atom in ringcopy.atoms:
        if atom.is_cyclic:
            if atom not in oncycle_atoms:
                oncycle_atoms.append(atom)
                oncycle_labels.append(atom.label)
    for atom in oncycle_atoms:
        for neighbour in atom.neighbours:
            if neighbour not in oncycle_atoms:
                if neighbour not in offcycle_atoms:
                    offcycle_atoms.append(neighbour)
                    offcycle_labels.append(neighbour.label)
    cyclicsystem = oncycle_atoms + offcycle_atoms
    for atom in ringcopy.atoms:
        if atom not in cyclicsystem:
            ringcopy.remove_atom(atom)
    for bond in ringcopy.bonds:
        if not bond.is_cyclic:
            if all((member.label in oncycle_labels for member in bond.atoms)):
                member1 = bond.atoms[0]
                member2 = bond.atoms[1]
                Hcap1 = Atom("H", coordinates=member1.coordinates)
                Hcap2 = Atom("H", coordinates=member2.coordinates)
                Hcap1_id = ringcopy.add_atom(Hcap1)
                Hcap2_id = ringcopy.add_atom(Hcap2)
                ringcopy.add_bond(bond.bond_type, Hcap1_id, member2)
                ringcopy.add_bond(bond.bond_type, Hcap2_id, member1)
                ringcopy.remove_bond(bond)
    for offatom in offcycle_atoms:
        offVBO = 0
        if any(bond.bond_type == "Delocalised" for bond in offatom.bonds):
            offdVBO = delocalisedLBO(offcycle_atoms)
        for bond in offatom.bonds:
            if bond.bond_type == "Single":
                offVBO += 1
            elif bond.bond_type == "Double":
                offVBO += 2
            elif bond.bond_type == "Triple":
                offVBO += 3
            elif bond.bond_type == "Quadruple":
                offVBO += 4
            elif bond.bond_type == "Delocalised":
                offVBO += offdVBO[offatom]
            elif bond.bond_type == "Aromatic":
                offVBO += 0
                get_logger(cpr().name).warning("impossible aromatic bond detected")
        if offVBO == 1:
            offatom.atomic_symbol = "H"
        elif offVBO == 2:
            offatom.atomic_symbol = "O"
        elif offVBO == 3:
            offatom.atomic_symbol = "N"
        elif offVBO == 4:
            offatom.atomic_symbol = "C"
        elif offVBO == 5:
            offatom.atomic_symbol = "P"
        elif offVBO == 6:
            offatom.atomic_symbol = "S"
        elif offVBO > 6:
            get_logger(cpr().name).warning(
                "issue detected in valence bond order calculations (capping)"
            )
    for cyclesys in ringcopy.components:
        cyclesys.assign_bond_types()
        cyclesys.kekulize()
        if any(bond.bond_type == "Delocalised" for bond in cyclesys.bonds):
            rdVBO = delocalisedLBO(cyclesys)
        for ratom in cyclesys.atoms:
            rVBO = 0
            if ratom.label in oncycle_labels:
                for rbond in ratom.bonds:
                    if rbond.bond_type == "Single":
                        rVBO += 1
                    elif rbond.bond_type == "Double":
                        rVBO += 2
                    elif rbond.bond_type == "Triple":
                        rVBO += 3
                    elif rbond.bond_type == "Quadruple":
                        rVBO += 4
                    elif rbond.bond_type == "Delocalised":
                        rVBO += rdVBO[ratom]
                    elif rbond.bond_type == "Aromatic":
                        rVBO += 0
                        get_logger(cpr().name).warning(
                            "impossible aromatic bond detected"
                        )
                for matom in unassigned:
                    if matom.label == ratom.label:
                        ringVBO[matom] = rVBO
                        unassigned.remove(matom)
    return ringVBO


def assign_VBS(atom: Atom, rVBO: dict[int, int], dVBO: dict[int, float]) -> int:
    VBO = 0
    if atom.is_metal:
        return 0
    if atom in rVBO:
        VBO = rVBO[atom]
    else:
        for bond in atom.bonds:
            if any(batom.is_metal for batom in bond.atoms):
                VBO += 0
            elif bond.bond_type == "Single":
                VBO += 1
            elif bond.bond_type == "Double":
                VBO += 2
            elif bond.bond_type == "Triple":
                VBO += 3
            elif bond.bond_type == "Quadruple":
                VBO += 4
            elif bond.bond_type == "Delocalised":
                VBO += dVBO[atom]
            elif bond.bond_type == "Aromatic":
                VBO += rVBO[atom]
    return VBO


def delocalisedLBO(molecule: Molecule) -> dict[int, float]:

    def TerminusCounter(atomlist: list[Atom]) -> int:
        NTerminus = 0
        for member in atomlist:
            connectivity = 0
            for bond in member.bonds:
                if bond.bond_type == "Delocalised":
                    connectivity += 1
            if connectivity == 1:
                NTerminus += 1
        return NTerminus

    def delocal_crawl(atomlist: list[Atom]) -> list[Atom]:
        for delocatom in atomlist:
            for bond in delocatom.bonds:
                if bond.bond_type == "Delocalised":
                    for member in bond.atoms:
                        if member not in atomlist:
                            atomlist.append(member)
                            return delocal_crawl(atomlist)
        return atomlist

    delocal_dict = {}
    molecule = molecule if isinstance(molecule, list) else molecule.atoms
    for atom in molecule:
        if all(
            [
                (any(bond.bond_type == "Delocalised" for bond in atom.bonds)),
                (atom not in delocal_dict),
            ]
        ):
            delocal_dict[atom] = []
            delocal_system = delocal_crawl([atom])
            NTerminus = TerminusCounter(delocal_system)
            for datom in delocal_system:
                connectivity = 0
                delocLBO = 0
                for neighbour in datom.neighbours:
                    if neighbour in delocal_system:
                        connectivity += 1
                if connectivity == 1:
                    delocLBO = (NTerminus + 1) / NTerminus
                if connectivity > 1:
                    delocLBO = (connectivity + 1) / connectivity
                delocal_dict[datom] = delocLBO
    return delocal_dict


def get_CN(atom: Atom) -> int:
    coord_number = 0
    for neighbour in atom.neighbours:
        if not neighbour.is_metal:
            coord_number += 1
    return coord_number


def valence_e(atom: Atom) -> int:
    elmnt = mendeleev.element(atom.atomic_symbol)
    if elmnt.block == "s":
        valence = elmnt.group_id
    elif elmnt.block == "p":
        valence = elmnt.group_id - 10
    elif elmnt.block == "d":
        valence = elmnt.group_id
    elif elmnt.block == "f":
        if elmnt.atomic_number in range(56, 72):
            valence = elmnt.atomic_number - 57 + 3
        elif elmnt.atomic_number in range(88, 104):
            valence = elmnt.atomic_number - 89 + 3
        else:
            get_logger(cpr().name).error("unexpected f block element")
            raise ValueError("valence_e() >> Unexpected f block element", elmnt)
    elif elmnt.group_id == 18:
        valence = 8 if elmnt.symbol != "He" else 2
    else:
        get_logger(cpr().name).error(
            "unexpected element in valence electron calculations"
        )
        raise ValueError("valence_e() >> Unexpected valence electrons", elmnt)
    return valence


def carbocation_check(atom: Atom) -> Literal["tetrahedral", "trigonal"]:
    abc = []
    for neighbours in atom.neighbours:
        if not neighbours.is_metal:
            abc.append(neighbours)
    angle1 = MolecularDescriptors.atom_angle(abc[0], atom, abc[1])
    angle2 = MolecularDescriptors.atom_angle(abc[0], atom, abc[2])
    angle3 = MolecularDescriptors.atom_angle(abc[1], atom, abc[2])
    AVGangle = abs(angle1 + angle2 + angle3) / 3
    tet = abs(AVGangle - 109.5)
    trig = abs(AVGangle - 120)
    if tet < trig:
        return "tetrahedral"
    if trig < tet:
        return "trigonal"


def carbene_type(atom: Atom) -> Literal["singlet", "triplet"]:
    alpha = atom.neighbours
    alpha_type = []
    for a in alpha:
        if not a.is_metal:
            alpha_type.append(a.atomic_symbol)
    for a in alpha_type:
        if not any([(a == "C"), (a == "H")]):
            return "singlet"
    if atom.is_cyclic is True:
        for ring in atom.rings:
            for species in ring.atoms:
                if not species.atomic_symbol == "C":
                    return "singlet"
    return "triplet"


def iVBS_Oxidation_Contrib(
    unique_atoms: list[Atom], rVBO: dict[int, int], dVBO: dict[int, float]
) -> dict[Atom, float]:
    VBS = 0
    CN = 0
    valence = 0
    oxi_contrib = {}
    for atom in unique_atoms:
        VBS = assign_VBS(atom, rVBO, dVBO)
        CN = get_CN(atom)
        valence = valence_e(atom)
        unpaired_e = 4 - abs(4 - valence)

        if atom.is_metal:
            oxi_contrib[atom] = 0
        elif VBS <= (unpaired_e):
            oxi_contrib[atom] = unpaired_e - VBS
        elif (VBS > unpaired_e) and (VBS < valence):
            diff = VBS - unpaired_e
            if diff <= 2:
                UPE = valence - unpaired_e - 2
            elif diff <= 4:
                UPE = valence - unpaired_e - 4
            elif diff <= 6:
                UPE = valence - unpaired_e - 6
            elif diff <= 8:
                UPE = valence - unpaired_e - 8
            oxi_contrib[atom] = VBS + UPE - valence
        elif VBS >= (valence):
            oxi_contrib[atom] = VBS - valence

        if any(
            [
                (atom.atomic_symbol == "C"),
                (atom.atomic_symbol == "Si"),
                (atom.atomic_symbol == "Ge"),
                (atom.atomic_symbol == "Pb"),
            ]
        ):
            if atom not in rVBO:
                if VBS == 3 and CN == 3:
                    geom = carbocation_check(atom)
                    if geom == "trigonal":
                        oxi_contrib[atom] = -1
                    if geom == "tetrahedral":
                        oxi_contrib[atom] = 1
            if VBS == 2 and CN == 2:
                carbene = carbene_type(atom)
                if carbene == "singlet":
                    oxi_contrib[atom] = 2
                if carbene == "triplet":
                    oxi_contrib[atom] = 0

        if all(
            [
                (atom.atomic_symbol == "N"),
                (VBS == 5 and CN == 3),
            ]
        ):
            N_sphere1 = atom.neighbours
            O_count = 0
            for neighbour in N_sphere1:
                if neighbour.atomic_symbol == "O":
                    O_count += 1
            geom = carbocation_check(atom)
            if O_count == 2 and geom == "trigonal":
                oxi_contrib[atom] = 0
    return oxi_contrib


def redundantAON(AON: dict[Atom, float], molecule: Molecule) -> dict[Atom, float]:
    redAON = {}
    for rsite1 in molecule.atoms:
        for usite1 in AON:
            redAON[usite1] = AON[usite1]
            if rsite1.label == usite1.label:
                redAON[rsite1] = AON[usite1]
    return redAON

