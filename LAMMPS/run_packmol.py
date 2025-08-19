import subprocess

from input_gen import system_dict_from_cif, write_packmol_input, xyz_to_cif_from_system_dict

def run_packmol_from_cif(cif_file, guest_list, output_inp="system.inp", packmol_exec="/home/users/taeun8991/packmol-21.0.2/packmol", tolerance=2.0):

    system_dict = system_dict_from_cif(cif_file, guest_list, tolerance=tolerance)

    write_packmol_input(system_dict, output_inp)

    with open(output_inp, "r") as inp:
        result = subprocess.run(
            [packmol_exec],
            stdin=inp,
            capture_output=True,
            text=True
        )

    print("Packmol STDOUT:\n", result.stdout)
    print("Packmol STDERR:\n", result.stderr)

    xyz_to_cif_from_system_dict(system_dict)
