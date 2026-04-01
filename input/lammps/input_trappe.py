import re
import numpy as np
from jinja2 import Environment
from pathlib import Path
from collections import defaultdict

atomic_masses = {
    "H": 1.0079, "C": 12.011, "N": 14.0067, "O": 15.9994,
    "F": 18.9984, "Cl": 35.45, "S": 32.06, "P": 30.97,
    "Br": 79.904, "I": 126.90, "Zn": 65.38, "Cu": 63.546
}


def parse_topology(top_file, molecule):
    with open(top_file, 'r') as f:
        lines = f.readlines()

    start_idx = -1
    for i, line in enumerate(lines):
        if line.strip().startswith(f'RESI {molecule}'):
            start_idx = i
            break
    if start_idx == -1:
        raise ValueError(f"{molecule} not found in topology file.")

    atoms, bonds, angles, charges = [], [], [], {}
    i = start_idx + 1
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("RESI") or line.startswith("END"):
            break

        if line.startswith("ATOM"):
            _, name, ff_type, charge = line.split()[:4]
            atoms.append((name, ff_type))
            charges[name] = float(charge)

        elif line.startswith("BOND"):
            parts = line.split()[1:]
            bonds.extend([(parts[j], parts[j + 1]) for j in range(0, len(parts), 2)])

        elif line.startswith("ANGLE"):
            parts = line.split()[1:]
            angles.append(tuple(parts))

        i += 1

    
    
    atom_names = {name for name, _ in atoms}
    type_to_names = defaultdict(list)
    for name, ff_type in atoms:
        type_to_names[ff_type].append(name)

    def normalize_token(tok):
        if tok in atom_names:
            return tok
        if tok in type_to_names and len(type_to_names[tok]) == 1:
            return type_to_names[tok][0]
        return tok

    bonds = [(normalize_token(a), normalize_token(b)) for a, b in bonds]
    angles = [tuple(normalize_token(x) for x in ang) for ang in angles]

    return atoms, bonds, angles, charges


def parse_parameters(par_file):
    with open(par_file, 'r') as f:
        content = f.read()

    
    nonbonded = {}
    nb_section = re.search(r'NONBONDED_MIE(.*?)$', content, re.S)
    if nb_section:
        for line in nb_section.group(1).splitlines():
            if line.strip() and not line.startswith('!'):
                parts = line.split()
                atom, eps, sig_ij = parts[:3]
                nonbonded[atom] = (float(eps) / 1000.0, float(sig_ij))  

    
    bond_params = {}
    bond_section = re.search(r'BONDS(.*?)ANGLES', content, re.S)
    if bond_section:
        for line in bond_section.group(1).splitlines():
            if line.strip() and not line.startswith('!'):
                parts = line.split()
                a1, a2, Kb, b0 = parts[:4]
                bond_params[tuple(sorted([a1, a2]))] = (float(Kb), float(b0))

    
    angle_params = {}
    angle_section = re.search(r'ANGLES(.*?)DIHEDRALS', content, re.S)
    if angle_section:
        for line in angle_section.group(1).splitlines():
            if line.strip() and not line.startswith('!'):
                parts = line.split()
                a1, a2, a3, Ktheta, Theta0 = parts[:5]
                angle_params[(a1, a2, a3)] = (float(Ktheta), float(Theta0))

    
    dihedral_params = {}
    dihedral_params_center = {}
    dihedral_section = re.search(r'DIHEDRALS(.*?)NONBONDED_MIE', content, re.S)
    if dihedral_section:
        for line in dihedral_section.group(1).splitlines():
            if line.strip() and not line.startswith('!'):
                parts = line.split()
                if len(parts) < 7:
                    continue
                a1, a2, a3, a4, Kchi, n, delta = parts[:7]
                key_exact = (a1, a2, a3, a4)
                key_center = (a2, a3)
                value = (float(Kchi), int(n), float(delta))
                dihedral_params.setdefault(key_exact, []).append(value)
                if a1 == "X" and a4 == "X":
                    dihedral_params_center.setdefault(key_center, []).append(value)

    return nonbonded, bond_params, angle_params, dihedral_params, dihedral_params_center


def read_xyz_atoms(xyz_file):
    atoms = []
    with open(xyz_file, 'r') as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise ValueError(f"Invalid xyz file: {xyz_file}")

    n = int(lines[0].strip())
    atom_lines = lines[2:2+n]

    for line in atom_lines:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid xyz atom line: {line}")
        sym = parts[0]
        x, y, z = map(float, parts[1:4])
        atoms.append((sym, np.array([x, y, z], dtype=float)))

    return atoms


def build_neighbor_graph(bonds):
    neighbors = defaultdict(set)
    for i, j in bonds:
        neighbors[i].add(j)
        neighbors[j].add(i)
    return neighbors


def find_angles(bonds):
    neighbors = build_neighbor_graph(bonds)
    angles = set()
    for b in neighbors:
        for n1 in neighbors[b]:
            for n2 in neighbors[b]:
                if n1 < n2:
                    angles.add((n1, b, n2))
    return list(angles)


def find_dihedrals(bonds):
    neighbors = build_neighbor_graph(bonds)
    dihedrals = set()
    for b1 in neighbors:
        for b2 in neighbors[b1]:
            if b2 == b1:
                continue
            for b3 in neighbors[b2]:
                if b3 in (b2, b1):
                    continue
                for b4 in neighbors[b3]:
                    if b4 in (b3, b2, b1):
                        continue
                    dihedrals.add((b1, b2, b3, b4))
    unique_dihedrals = set()
    for d in dihedrals:
        d1 = d
        d2 = tuple(reversed(d))
        if d2 in unique_dihedrals:
            continue
        unique_dihedrals.add(d1)
    return list(unique_dihedrals)


def sort2(a, b):
    return "_".join(sorted([a, b]))


def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("Zero-length vector encountered while building pseudo-site.")
    return v / n


def build_special_site_coords(molecule, atoms, bonds, xyz_atoms):
    site_names = [name for name, _ in atoms]

    if molecule == "N2":
        if len(xyz_atoms) != 2:
            raise ValueError(f"N2 expects 2 xyz atoms, got {len(xyz_atoms)}")

        r1 = xyz_atoms[0][1]
        r2 = xyz_atoms[1][1]

        
        if len(site_names) != 3:
            raise ValueError(f"N2 topology expected 3 sites, got {len(site_names)}")

        return {
            site_names[0]: r1,
            site_names[1]: 0.5 * (r1 + r2),
            site_names[2]: r2,
        }

    if molecule == "O2":
        if len(xyz_atoms) != 2:
            raise ValueError(f"O2 expects 2 xyz atoms, got {len(xyz_atoms)}")

        r1 = xyz_atoms[0][1]
        r2 = xyz_atoms[1][1]

        
        if len(site_names) != 3:
            raise ValueError(f"O2 topology expected 3 sites, got {len(site_names)}")

        return {
            site_names[0]: r1,
            site_names[1]: 0.5 * (r1 + r2),
            site_names[2]: r2,
        }

    if molecule == "CO":
        if len(xyz_atoms) != 2:
            raise ValueError(f"CO expects 2 xyz atoms, got {len(xyz_atoms)}")

        r1 = xyz_atoms[0][1]
        r2 = xyz_atoms[1][1]

        if len(site_names) != 3:
            raise ValueError(f"CO topology expected 3 sites, got {len(site_names)}")

        return {
            site_names[0]: r1,
            site_names[1]: 0.5 * (r1 + r2),
            site_names[2]: r2,
        }

    if molecule == "TIP4":
        if len(xyz_atoms) != 3:
            raise ValueError(f"TIP4 expects 3 xyz atoms, got {len(xyz_atoms)}")

        
        O = xyz_atoms[0][1]
        H1 = xyz_atoms[1][1]
        H2 = xyz_atoms[2][1]

        hmid = 0.5 * (H1 + H2)
        direction = _normalize(hmid - O)

        
        d_OM = 0.1546
        M = O + d_OM * direction

        
        coords = {}
        for name, ff_type in atoms:
            uname = name.upper()
            utype = ff_type.upper()

            if uname == "O":
                coords[name] = O
            elif uname == "H1":
                coords[name] = H1
            elif uname == "H2":
                coords[name] = H2
            elif uname == "M" or utype == "MO":
                coords[name] = M
            else:
                raise ValueError(f"TIP4: cannot map site {name}/{ff_type}")

        return coords

    return None


def build_default_site_coords(atoms, xyz_atoms):
    if len(atoms) != len(xyz_atoms):
        raise ValueError(
            f"Topology site count ({len(atoms)}) != xyz atom count ({len(xyz_atoms)})"
        )

    coords = {}
    for (site_name, _ff_type), (_sym, r) in zip(atoms, xyz_atoms):
        coords[site_name] = r
    return coords


def generate_lt(molecule, xyz_file, top_file, par_file, output_file):
    atoms, bonds, angles, charges = parse_topology(top_file, molecule)

    
    
    if not angles:
        angles = find_angles(bonds)

    dihedrals = find_dihedrals(bonds)

    lj_params, bond_params, angle_params, dihedral_params, dihedral_params_center = parse_parameters(par_file)
    xyz_atoms = read_xyz_atoms(xyz_file)

    
    coords_np = build_special_site_coords(molecule, atoms, bonds, xyz_atoms)
    if coords_np is None:
        coords_np = build_default_site_coords(atoms, xyz_atoms)

    
    coords = {k: (float(v[0]), float(v[1]), float(v[2])) for k, v in coords_np.items()}

    atoms_dict = dict(atoms)
    atom_types_used = set(t for _, t in atoms)

    mass_dict = {}
    for atype in atom_types_used:
        if atype[:2] in atomic_masses:
            element = atype[:2]
        elif atype[:1] in atomic_masses:
            element = atype[:1]
        else:
            element = None
        mass_dict[atype] = atomic_masses.get(element, 1.0)

    has_bonds = len(bonds) > 0
    has_angles = len(angles) > 0
    has_dihedrals = len(dihedrals) > 0

    template_str = """
guest {

  write_once("In Init") {
    atom_style full
{% if has_bonds %}
    bond_style harmonic
{% endif %}
{% if has_angles %}
    angle_style harmonic
{% endif %}
{% if has_dihedrals %}
    dihedral_style charmm
{% endif %}
    special_bonds amber
  }
  
  write_once("In Settings") {
    {% for atom, (eps, sigma) in lj_params.items() %}
    pair_coeff @atom:{{ atom }} @atom:{{ atom }} {{ "%.4f"|format(eps) }} {{ "%.4f"|format(sigma) }}
    {% endfor %}
    {% for (a, b), (k, r0) in bond_params.items() %}
    bond_coeff @bond:{{ a }}_{{ b }} {{ "%.1f"|format(k) }} {{ "%.4f"|format(r0) }}
    {% endfor %}
    {% for (a1, a2, a3), (k, theta) in angle_params.items() %}
    angle_coeff @angle:{{ a1 }}_{{ a2 }}_{{ a3 }} {{ "%.1f"|format(k) }} {{ "%.1f"|format(theta) }}
    angle_coeff @angle:{{ a3 }}_{{ a2 }}_{{ a1 }} {{ "%.1f"|format(k) }} {{ "%.1f"|format(theta) }}
    {% endfor %}
    {% set seen_types = [] %}
    {% for i, (a1, a2, a3, a4) in enumerate(dihedrals, 1) %}
      {% set t1 = atoms_dict[a1] %}
      {% set t2 = atoms_dict[a2] %}
      {% set t3 = atoms_dict[a3] %}
      {% set t4 = atoms_dict[a4] %}
      {% set type_tuple = t1 ~ '_' ~ t2 ~ '_' ~ t3 ~ '_' ~ t4 %}
      {% set type_tuple_rev = t4 ~ '_' ~ t3 ~ '_' ~ t2 ~ '_' ~ t1 %}
      {% if type_tuple not in seen_types and type_tuple_rev not in seen_types %}
        {% set _ = seen_types.append(type_tuple) %}
        {% set _ = seen_types.append(type_tuple_rev) %}
        {% set terms_exact = dihedral_params.get((t1, t2, t3, t4), []) %}
        {% set terms_center = dihedral_params_center.get((t2, t3), []) %}
        {% if terms_exact %}
          {% for (k, n, phi) in terms_exact %}
    dihedral_coeff @dihedral:{{ t1 }}_{{ t2 }}_{{ t3 }}_{{ t4 }} {{ "%.4f"|format(k) }} {{ n }} {{ "%.1f"|format(phi) }} 1.0
          {% endfor %}
        {% elif terms_center %}
          {% for (k, n, phi) in terms_center %}
    dihedral_coeff @dihedral:{{ t1 }}_{{ t2 }}_{{ t3 }}_{{ t4 }} {{ "%.4f"|format(k) }} {{ n }} {{ "%.1f"|format(phi) }} 1.0
          {% endfor %}
        {% endif %}
      {% endif %}
    {% endfor %}
  }

  write_once("Data Masses") {
    {% for atom in atom_types %}
    @atom:{{ atom }} {{ "%.4f"|format(mass_dict[atom]) }}
    {% endfor %}
  }

  write("Data Atoms") {
    {% for (name, ff_type) in atoms %}
    $atom:{{ name }} $mol:{{ molecule }} @atom:{{ ff_type }} {{ "%.5f"|format(charges[name]) }} {{ "%.5f"|format(coords[name][0]) }} {{ "%.5f"|format(coords[name][1]) }} {{ "%.5f"|format(coords[name][2]) }}
    {% endfor %}
  }

  {% if has_bonds %}
  write("Data Bonds") {
    {% for i, (a1, a2) in enumerate(bonds, 1) %}
    $bond:b{{ i }} @bond:{{ sort2(atoms_dict[a1], atoms_dict[a2]) }} $atom:{{ a1 }} $atom:{{ a2 }}
    {% endfor %}
  }
  {% endif %}

  {% if has_angles %}
  write("Data Angles") {
    {% for i, (a1, a2, a3) in enumerate(angles, 1) %}
    $angle:a{{ i }} @angle:{{ atoms_dict[a1] }}_{{ atoms_dict[a2] }}_{{ atoms_dict[a3] }} $atom:{{ a1 }} $atom:{{ a2 }} $atom:{{ a3 }}
    {% endfor %}
  }
  {% endif %}

  {% if has_dihedrals %}
  write("Data Dihedrals") {
    {% for i, (a1, a2, a3, a4) in enumerate(dihedrals, 1) %}
    $dihedral:d{{ i }} @dihedral:{{ atoms_dict[a1] }}_{{ atoms_dict[a2] }}_{{ atoms_dict[a3] }}_{{ atoms_dict[a4] }} $atom:{{ a1 }} $atom:{{ a2 }} $atom:{{ a3 }} $atom:{{ a4 }}
    {% endfor %}
  }
  {% endif %}

}
"""

    env = Environment(trim_blocks=True, lstrip_blocks=True)
    tmpl = env.from_string(template_str)
    rendered = tmpl.render(
        molecule=molecule,
        atoms=atoms,
        bonds=bonds,
        angles=angles,
        dihedrals=dihedrals,
        charges=charges,
        lj_params={k: v for k, v in lj_params.items() if k in atom_types_used},
        bond_params={k: v for k, v in bond_params.items() if all(x in atom_types_used for x in k)},
        angle_params={k: v for k, v in angle_params.items() if all(x in atom_types_used for x in k)},
        dihedral_params_center={k: v for k, v in dihedral_params_center.items() if all(x in atom_types_used for x in k)},
        dihedral_params={k: v for k, v in dihedral_params.items() if all(x in atom_types_used for x in k)},
        atom_types=atom_types_used,
        coords=coords,
        atoms_dict=atoms_dict,
        mass_dict=mass_dict,
        sort2=sort2,
        enumerate=enumerate,
        has_bonds=has_bonds,
        has_angles=has_angles,
        has_dihedrals=has_dihedrals,
    )

    with open(output_file, 'w') as f:
        f.write(rendered)