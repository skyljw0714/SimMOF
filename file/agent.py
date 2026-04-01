import os
from pathlib import Path

import ase.io
import ase.io.vasp
import numpy as np

from config import VASP_EXECUTABLE, VASP_POTENTIAL_DIR_PATH

VASP_QSUB_QUEUE = "long"
VASP_QSUB_RESOURCES = "nodes=2:ppn=16:aa"


class VASPFileAgent:
    @classmethod
    def get_pathdata(cls, cif_path):
        label = Path(cif_path).stem
        out_path = os.path.dirname(cif_path)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            print("Path does not exist.. make dir to %s" % out_path)
        out_dir = out_path + '/'
        return out_dir, label

    @classmethod
    def get_vasp_file(cls, cif_path=None):
        pot_dir = str(VASP_POTENTIAL_DIR_PATH) + '/'
        out_dir, label = cls.get_pathdata(cif_path)
        atoms = ase.io.read(cif_path)
        cls.atoms_to_poscar(atoms, out_dir)
        cls.atoms_to_potcar(atoms, out_dir, pot_dir)
        cls.make_qsub(out_dir, label)
        return

    @classmethod
    def make_qsub(cls, out_dir, label):
        with open(out_dir + label + '.qsub', 'w') as f:
            print("#!/bin/sh", file=f)
            print("#PBS -r n", file=f)
            print(f"#PBS -q {VASP_QSUB_QUEUE}", file=f)
            print(f"#PBS -l {VASP_QSUB_RESOURCES}", file=f)
            print(f"#PBS -e {out_dir}{label}.pbs.err", file=f)
            print(f"#PBS -o {out_dir}{label}.pbs.out", file=f)
            print("", file=f)

            print("cd $PBS_O_WORKDIR", file=f)
            print('echo "START $(date)" > START', file=f)

            print("rm -f DONE FAILED", file=f)

            print("NPROCS=`wc -l < $PBS_NODEFILE`", file=f)
            print("", file=f)

            print(f"mpirun -v -machinefile $PBS_NODEFILE -np $NPROCS {VASP_EXECUTABLE} > out.txt 2>&1", file=f)
            print("rc=$?", file=f)
            print("", file=f)

            print("if [ $rc -ne 0 ]; then", file=f)
            print('  echo "FAILED rc=$rc $(date)" > FAILED', file=f)
            print("else", file=f)
            print('  if tail -n 200 out.txt | grep -q "reached required accuracy - stopping structural energy minimisation" \\', file=f)
            print('     || tail -n 200 OUTCAR | grep -q "General timing and accounting informations"; then', file=f)
            print('    echo "DONE $(date)" > DONE', file=f)
            print("  else", file=f)
            print('    echo "FAILED (no success marker) $(date)" > FAILED', file=f)
            print("  fi", file=f)
            print("fi", file=f)
            print("", file=f)

            print("exit $rc", file=f)

    @classmethod
    def atoms_to_poscar(cls, atoms, out_dir):
        ase.io.vasp.write_vasp(out_dir + 'POSCAR', atoms, direct=True, sort=True, vasp5=True)

    @classmethod
    def atoms_to_potcar(cls, atoms, out_dir, pot_dir):
        species = sorted(set(atoms.get_chemical_symbols()))

        with open(out_dir + 'POTCAR', 'w'):
            pass

        for s in species:
            path = os.path.abspath(os.path.join(pot_dir, s))
            ds = ['In', 'Ga', 'Ge', 'Sn', 'Tl', 'Pb', 'Bi', 'Po', 'At']
            svs = ['Li', 'Na', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Zr', 'Mo', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Cs', 'Ba', 'Fr', 'Ra', 'W']
            pvs = ['Cr', 'Mn', 'Tc', 'Ru', 'Rh', 'Ta', 'Hf']

            if s in ds:
                path += '_d'
            elif s in svs:
                path += '_sv'
            elif s in pvs:
                path += '_pv'

            potcar_file = os.path.join(path, 'POTCAR')
            if os.path.exists(potcar_file):
                os.system(f"cat {potcar_file} >> {out_dir + 'POTCAR'}")
            else:
                raise IOError(f"[CIF2VASP] No pseudo potential file.. {potcar_file}")

    @classmethod
    def atoms_to_min_kpoints(cls, atoms, out_dir, label):
        a, b, c = atoms.cell
        norm_a, norm_b, norm_c = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c)
        n_a, n_b, n_c = 1 / norm_a, 1 / norm_b, 1 / norm_c
        min_val = min([n_a, n_b, n_c])
        points = [int(i) for i in np.round(np.array([n_a, n_b, n_c]) / min_val)]
        with open(out_dir + 'KPOINTS', 'w') as f:
            print(label, file=f)
            print(0, file=f)
            print("gamma", file=f)
            print("{0:d} {1:d} {2:d}".format(*points), file=f)
            print("0 0 0", file=f)
