import sys
from ase.io import read, write
from ase.optimize import BFGS
from mace.calculators import mace_mp

def optimize_structure(atoms, device="cuda:0"):
    calc = mace_mp(model='large', dispersion=True, default_dtype="float64", device=device)
    atoms.calc = calc

    # Perform geometry optimization using BFGS algorithm
    dyn = BFGS(atoms, logfile=None)
    dyn.run(fmax=0.02)  # Converge when max force < 0.02 eV/Å
    return atoms, atoms.get_potential_energy()

def main(host_file, host_guest_file, guest_file, device="cuda:0"):
    host = read(host_file)
    host_guest = read(host_guest_file)
    guest = read(guest_file)

    host.set_pbc(True)
    host_guest.set_pbc(True)
    guest.set_pbc(False)

    print("Optimizing host structure...")
    host, e_host = optimize_structure(host, device)

    print("Optimizing host+guest structure...")
    host_guest, e_host_guest = optimize_structure(host_guest, device)

    print("Optimizing guest structure...")
    guest, e_guest = optimize_structure(guest, device)

    # Compute binding energy
    # Binding Energy = E(host+guest) - E(host) - E(guest)
    binding_energy = e_host_guest - (e_host + e_guest)

    print(f"\nBinding energy: {binding_energy:.6f} eV")
    with open("binding_energy.txt", "w") as f:
        f.write(f"Binding energy (eV): {binding_energy:.6f}\n")

    # Save optimized structures
    write("optimized_host.xyz", host)
    write("optimized_host_guest.xyz", host_guest)
    write("optimized_guest.xyz", guest)

if __name__ == "__main__":
    # Check for correct number of command-line arguments
    if len(sys.argv) != 4:
        print("Usage: python calculate_binding_energy.py host.cif host_guest.cif guest.cif")
        sys.exit(1)

    # Parse file paths
    host_cif = sys.argv[1]
    host_guest_cif = sys.argv[2]
    guest_cif = sys.argv[3]

    # Run the main calculation
    main(host_cif, host_guest_cif, guest_cif)
