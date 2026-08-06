
import os
from collections import defaultdict
from pathlib import Path

CGENFF_RTF = str(
    Path(__file__).parent / "cgenff3.0.1" / "top_all36_cgenff.rtf"
)
CGENFF_PRM = str(
    Path(__file__).parent / "cgenff3.0.1" / "par_all36_cgenff.prm"
)

ATOMIC_MASSES = {
    "H": 1.0079,
    "C": 12.011,
    "N": 14.0067,
    "O": 15.9994,
    "F": 18.9984,
    "Cl": 35.45,
    "S": 32.06,
    "P": 30.97,
    "Br": 79.904,
    "I": 126.90,
    "Zn": 65.38,
    "Cu": 63.546,
}



def _generate_angles(bonds):
    neighbors = defaultdict(set)
    for a, b in bonds:
        neighbors[a].add(b)
        neighbors[b].add(a)
    angles = set()
    for center, nbs in neighbors.items():
        nbs = list(nbs)
        for i in range(len(nbs)):
            for j in range(i + 1, len(nbs)):
                angles.add((nbs[i], center, nbs[j]))
    return list(angles)


def _generate_dihedrals(bonds):
    neighbors = defaultdict(set)
    for a, b in bonds:
        neighbors[a].add(b)
        neighbors[b].add(a)
    dihedrals = set()
    for b in neighbors:
        for c in neighbors[b]:
            for a in neighbors[b]:
                if a == c:
                    continue
                for d in neighbors[c]:
                    if d == b:
                        continue
                    dihedrals.add((a, b, c, d))
    return list(dihedrals)


def _generate_impropers(bonds):
    neighbors = defaultdict(set)
    for a, b in bonds:
        neighbors[a].add(b)
        neighbors[b].add(a)
    impropers = set()
    for center, nbs in neighbors.items():
        nbs = list(nbs)
        if len(nbs) >= 3:
            for i in range(len(nbs)):
                for j in range(i + 1, len(nbs)):
                    for k in range(j + 1, len(nbs)):
                        impropers.add((center, nbs[i], nbs[j], nbs[k]))
    return list(impropers)


def _parse_rtf(rtf_file: str, molecule: str):
    atoms, types, charges, bonds = [], [], {}, []
    angles, dihedrals, impropers = [], [], []
    in_resi = False
    with open(rtf_file) as f:
        for line in f:
            l = line.strip()
            if l.startswith(f"RESI {molecule}") or l.startswith(f"RESI {molecule} "):
                in_resi = True
                continue
            if in_resi and l.startswith("RESI "):
                break
            if not in_resi:
                continue
            if l.startswith("ATOM"):
                parts = l.split()
                if len(parts) >= 4:
                    name, ff_type, charge = parts[1], parts[2], parts[3]
                    atoms.append((name, ff_type))
                    types.append(ff_type)
                    charges[name] = float(charge)
            elif l.startswith("BOND"):
                parts = l.split()[1:]
                bonds.extend(
                    [(parts[j], parts[j + 1]) for j in range(0, len(parts) - 1, 2)]
                )
            elif l.startswith("ANGLE"):
                parts = l.split()[1:]
                angles.extend(
                    [(parts[j], parts[j + 1], parts[j + 2]) for j in range(0, len(parts) - 2, 3)]
                )
            elif l.startswith("DIHE") or l.startswith("DIHEDRAL"):
                parts = l.split()[1:]
                dihedrals.extend(
                    [(parts[j], parts[j + 1], parts[j + 2], parts[j + 3]) for j in range(0, len(parts) - 3, 4)]
                )
            elif l.startswith("IMPH") or l.startswith("IMPROPER"):
                parts = l.split()[1:]
                impropers.extend(
                    [(parts[j], parts[j + 1], parts[j + 2], parts[j + 3]) for j in range(0, len(parts) - 3, 4)]
                )

    if not atoms:
        raise ValueError(
            f"RESI '{molecule}' not found in {rtf_file}. "
            "Add the molecule to the RTF file (e.g. via CGenFF web server)."
        )

    if not angles:
        angles = _generate_angles(bonds)
    if not dihedrals:
        dihedrals = _generate_dihedrals(bonds)
    if not impropers:
        impropers = _generate_impropers(bonds)

    return atoms, types, charges, bonds, angles, dihedrals, impropers


def _is_float(val):
    try:
        float(val)
        return True
    except ValueError:
        return False


def _parse_prm(prm_file: str):
    masses, lj, bonds, angles_2, angles_4, dihedrals, impropers = {}, {}, {}, {}, {}, {}, {}
    section = None
    with open(prm_file) as f:
        for line in f:
            l = line.strip()
            if not l or l.startswith("!"):
                continue
            if l.startswith("MASS"):
                section = "MASS"
            elif l.startswith("NONBONDED"):
                section = "NONBONDED"
            elif l.startswith("BONDS"):
                section = "BONDS"
            elif l.startswith("ANGLES"):
                section = "ANGLES"
            elif l.startswith("DIHEDRALS"):
                section = "DIHEDRALS"
            elif l.startswith("IMPROPERS"):
                section = "IMPROPERS"
            elif l.startswith("END"):
                section = None
                continue

            if section == "MASS" and l.startswith("MASS"):
                parts = l.split()
                if len(parts) >= 4:
                    masses[parts[2]] = float(parts[3])

            elif section == "NONBONDED":
                parts = l.split()
                if (
                    len(parts) >= 4
                    and parts[0][0].isalpha()
                    and all(
                        x.replace(".", "", 1).replace("-", "", 1).isdigit()
                        for x in [parts[2], parts[3]]
                    )
                ):
                    lj[parts[0]] = (float(parts[2]), float(parts[3]))

            elif section == "BONDS":
                parts = l.split()
                if len(parts) >= 4:
                    bonds[(parts[0], parts[1])] = (float(parts[2]), float(parts[3]))

            elif section == "ANGLES":
                parts = l.split()
                if len(parts) >= 5:
                    a1, a2, a3 = parts[:3]
                    k, theta = float(parts[3]), float(parts[4])
                    if len(parts) > 6 and _is_float(parts[5]) and _is_float(parts[6]):
                        angles_4[(a1, a2, a3)] = (k, theta, float(parts[5]), float(parts[6]))
                    else:
                        angles_2[(a1, a2, a3)] = (k, theta)

            elif section == "DIHEDRALS":
                lnc = l.split("!")[0].strip()
                parts = lnc.split()
                if len(parts) >= 7:
                    a1, a2, a3, a4 = parts[:4]
                    params = parts[4:]
                    for i in range(len(params) // 3):
                        k = float(params[3 * i])
                        n = int(float(params[3 * i + 1]))
                        phi = float(params[3 * i + 2])
                        dihedrals[(a1, a2, a3, a4, n)] = (k, phi)

            elif section == "IMPROPERS":
                lnc = l.split("!")[0].strip()
                parts = lnc.split()
                if len(parts) >= 7:
                    a1, a2, a3, a4 = parts[:4]
                    impropers[(a1, a2, a3, a4)] = (
                        float(parts[4]),
                        int(float(parts[5])),
                        float(parts[6]),
                    )

    return masses, lj, bonds, angles_2, angles_4, dihedrals, impropers


def _read_xyz(xyz_file: str):
    lines = Path(xyz_file).read_text().splitlines()
    coords = []
    for line in lines[2:]:
        if not line.strip():
            continue
        parts = line.split()
        coords.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return coords



def generate_lt(
    molecule: str,
    xyz_file: str,
    output_file: str,
    rtf_file: str = CGENFF_RTF,
    prm_file: str = CGENFF_PRM,
):
    atoms, types, charges, bonds, angles, dihedrals, impropers = _parse_rtf(rtf_file, molecule)
    atom_name_to_type = dict(atoms)

    masses, lj_params, bond_params, angle_2_params, angle_4_params, dihedral_params, improper_params = _parse_prm(prm_file)

    atom_types_used = set(types)
    bond_types_used = {
        tuple(sorted([atom_name_to_type[a1], atom_name_to_type[a2]]))
        for a1, a2 in bonds
    }
    angle_types_used = {
        (atom_name_to_type[a1], atom_name_to_type[a2], atom_name_to_type[a3])
        for a1, a2, a3 in angles
    }
    dihedral_types_used = {
        (atom_name_to_type[a1], atom_name_to_type[a2], atom_name_to_type[a3], atom_name_to_type[a4])
        for a1, a2, a3, a4 in dihedrals
    }
    improper_types_used = {
        (atom_name_to_type[a1], atom_name_to_type[a2], atom_name_to_type[a3], atom_name_to_type[a4])
        for a1, a2, a3, a4 in impropers
    }

    f_masses = {k: v for k, v in masses.items() if k in atom_types_used}
    f_lj = {k: v for k, v in lj_params.items() if k in atom_types_used}
    f_bonds = {k: v for k, v in bond_params.items() if set(k).issubset(atom_types_used)}
    f_ang2 = {k: v for k, v in angle_2_params.items() if k in angle_types_used or k[::-1] in angle_types_used}
    f_ang4 = {k: v for k, v in angle_4_params.items() if k in angle_types_used or k[::-1] in angle_types_used}
    f_dihe = {k: v for k, v in dihedral_params.items() if k[:4] in dihedral_types_used or k[:4][::-1] in dihedral_types_used}
    f_impr = {k: v for k, v in improper_params.items() if k in improper_types_used or k[::-1] in improper_types_used}

    coords = _read_xyz(xyz_file)

    lines = [f"{molecule} {{", ""]
    lines += [
        '  write_once("In Init") {',
        "    atom_style full",
        "    pair_style lj/cut/coul/long 10.0",
        "    kspace_style pppm 1.0e-4",
        "    bond_style harmonic",
        "    angle_style hybrid harmonic charmm",
        "    dihedral_style charmm",
        "    improper_style harmonic",
        "    special_bonds amber",
        "  }",
        "",
    ]

    lines += ['  write_once("In Settings") {']
    for at, (eps, rmin) in f_lj.items():
        lines.append(f"    pair_coeff @atom:{at} @atom:{at} {eps:.4f} {rmin:.4f}")
    for (a, b), (k, r0) in f_bonds.items():
        lines.append(f"    bond_coeff @bond:{a}_{b} {k:.1f} {r0:.4f}")
    for (a1, a2, a3), (k, theta) in f_ang2.items():
        lines.append(f"    angle_coeff @angle:{a1}_{a2}_{a3} harmonic {k:.1f} {theta:.1f}")
    for (a1, a2, a3), (k, theta, Kub, S0) in f_ang4.items():
        lines.append(f"    angle_coeff @angle:{a1}_{a2}_{a3} charmm {k:.1f} {theta:.1f} {Kub:.2f} {S0:.3f}")
    for (a1, a2, a3, a4, n), (k, phi) in f_dihe.items():
        lines.append(f"    dihedral_coeff @dihedral:{a1}_{a2}_{a3}_{a4} {k:.4f} {n} {phi:.1f} 1.0")
    for (a1, a2, a3, a4), (k, n, phi) in f_impr.items():
        lines.append(f"    improper_coeff @improper:{a1}_{a2}_{a3}_{a4} {k:.4f} {phi:.1f}")
    lines += ["  }", ""]

    lines += ['  write_once("Data Masses") {']
    for at, m in f_masses.items():
        lines.append(f"    @atom:{at} {m:.4f}")
    lines += ["  }", ""]

    lines += ['  write("Data Atoms") {']
    for i, (name, ff_type) in enumerate(atoms):
        x, y, z = coords[i]
        lines.append(
            f"    $atom:{name} $mol:{molecule} @atom:{ff_type} "
            f"{charges[name]} {x:.6f} {y:.6f} {z:.6f}"
        )
    lines += ["  }", ""]

    lines += ['  write("Data Bonds") {']
    for i, (a1, a2) in enumerate(bonds, 1):
        t1, t2 = atom_name_to_type[a1], atom_name_to_type[a2]
        lines.append(f"    $bond:b{i} @bond:{t1}_{t2} $atom:{a1} $atom:{a2}")
    lines += ["  }", ""]

    _write_interactions(lines, "Angles", angles, atom_name_to_type, f_ang2, f_ang4, prefix="a")
    _write_interactions(lines, "Dihedrals", dihedrals, atom_name_to_type, f_dihe, prefix="d")
    _write_interactions(lines, "Impropers", impropers, atom_name_to_type, f_impr, prefix="im")

    lines.append("}")

    Path(output_file).write_text("\n".join(lines) + "\n")
    print(f"[CHARMM] .lt file written → {output_file}")
    return output_file


def _write_interactions(lines, section, interactions, a2t, params_dict, prefix):
    if section == "Dihedrals":
        known_4 = {k[:4] for k in params_dict}
    else:
        known_4 = None

    lines += [f'  write("Data {section}") {{']
    for i, tup in enumerate(interactions, 1):
        types = tuple(a2t[a] for a in tup)
        rev_types = types[::-1]
        if known_4 is not None:
            found = types in known_4 or rev_types in known_4
        else:
            found = types in params_dict or rev_types in params_dict
        if found:
            type_str = "_".join(types)
            atom_str = " ".join(f"$atom:{a}" for a in tup)
            lines.append(
                f"    ${section[:-1].lower()}:{prefix}{i} "
                f"@{section[:-1].lower()}:{type_str} {atom_str}"
            )
    lines += ["  }", ""]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python charmm_lt_autogen.py <RESI_name> <molecule.xyz> <output.lt> [rtf_file] [prm_file]")
        sys.exit(1)
    mol = sys.argv[1]
    xyz = sys.argv[2]
    out = sys.argv[3]
    rtf = sys.argv[4] if len(sys.argv) > 4 else CGENFF_RTF
    prm = sys.argv[5] if len(sys.argv) > 5 else CGENFF_PRM
    generate_lt(mol, xyz, out, rtf_file=rtf, prm_file=prm)
