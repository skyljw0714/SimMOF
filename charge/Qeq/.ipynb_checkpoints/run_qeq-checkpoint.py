import numpy as np
from ase.io import read
from scipy.special import erf


def get_parameters(qeqfile):
    eV = 3.67493245e-2
    Angstrom = 1./0.529177249
    parameters = {}
    with open(qeqfile) as f:
        for i in range(13):
            next(f)
        for line in f:
            data = line.rstrip().split()
            element = data[0]
            elecnegativity = float(data[1]) * eV
            hardness = float(data[2]) * eV
            sradius = float(data[3]) * Angstrom
            basis = 1.0 / (sradius * sradius)
            parameters[element] = [elecnegativity, hardness, basis]
    return parameters


def read_structure(filename):
    atoms = read(filename)
    positions = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()
    symbols = atoms.get_chemical_symbols()
    cell = atoms.get_cell()
    print(cell)
    return atoms, positions, numbers, symbols, cell


def minimum_image_distance(pos1, pos2, cell):
    delta = pos1 - pos2
    inv_cell = np.linalg.inv(cell)
    frac_delta = np.dot(delta, inv_cell)
    frac_delta -= np.round(frac_delta)
    cart_delta = np.dot(frac_delta, cell)
    return np.linalg.norm(cart_delta)


def calculate_coulomb_integral(a, b, R):
    p = np.sqrt(a * b / (a + b))
    return erf(p * R) / R


def fill_J(positions, J, BasisSet, CoulombMaxDistance, cell):
    nAtoms = len(positions)
    for k in range(nAtoms):
        for l in range(nAtoms):
            if k > l:
                R = minimum_image_distance(positions[k], positions[l], cell)
                if R < CoulombMaxDistance:
                    coulomb = calculate_coulomb_integral(BasisSet[k], BasisSet[l], R)
                else:
                    coulomb = 1.0 / R
                J[k][l] = coulomb
                J[l][k] = coulomb
    for i in range(nAtoms + 1):
        J[nAtoms][i] = 1.0
        J[i][nAtoms] = 1.0
    J[nAtoms][nAtoms] = 0.0


def compute_Qeq_charges(positions, symbols, total_charge, charge_past, cell, param_file):
    CoulombThreshold = 1e-9
    nAtoms = len(positions)
    ElectroNegativity = np.zeros(nAtoms)
    J = np.zeros((nAtoms + 1, nAtoms + 1))
    Voltage = charge_past.copy()
    BasisSet = np.zeros(nAtoms)
    parameters = get_parameters(param_file)

    for i in range(nAtoms):
        symbol = symbols[i]
        ElectroNegativity[i] = parameters[symbol][0]
        J[i][i] = parameters[symbol][1]
        BasisSet[i] = parameters[symbol][2]

    SmallestGaussianExponent = min(BasisSet)
    CoulombMaxDistance = 2 * np.sqrt(-np.log(CoulombThreshold) / SmallestGaussianExponent)
    CoulombMaxDistance = min(CoulombMaxDistance, 12.0)  # 강제 상한: 12 Å

    fill_J(positions, J, BasisSet, CoulombMaxDistance, cell)

    Voltage[:-1] = ElectroNegativity
    Voltage[-1] = total_charge

    charges = np.linalg.solve(J, Voltage)
    return charges


def Qeq_charge_equilibration(positions, symbols, total_charge, cell, param_file):
    nAtoms = len(positions)
    charges = np.zeros(nAtoms + 1)
    for _ in range(1):  # Single-step iteration (linear system)
        charges = compute_Qeq_charges(positions, symbols, total_charge, charges, cell, param_file)
    return charges[:-1]  # Last element is Lagrange multiplier



# 실행
# ------------------------------
if __name__ == '__main__':
    cif_file = '../../cifs/uio-66.cif'  # CIF 파일명
    qeq_param_file = 'qeq.txt'
    total_charge = 0.0  # 전체 전하 (중성 시스템)

    atoms, positions, numbers, symbols, cell = read_structure(cif_file)

    charges = Qeq_charge_equilibration(positions, symbols, total_charge, cell, qeq_param_file)

    for i, (s, c) in enumerate(zip(symbols, charges)):
        print(f"Atom {i+1:3d}: {s:>2}  Charge = {c: .4f}")

