import os
import subprocess
from pathlib import Path
from typing import Dict, Any
import textwrap

from config import LAMMPS_EXECUTABLE

LAMMPS_QSUB_QUEUE = "long"
LAMMPS_QSUB_NODES = "2:ppn=8:aa"
LAMMPS_QSUB_NP = 8

class LAMMPSRunner:

    def __init__(self, queue: str = LAMMPS_QSUB_QUEUE, nodes: str = LAMMPS_QSUB_NODES, np: int = LAMMPS_QSUB_NP):
        self.queue = queue
        self.nodes = nodes
        self.np = np

    def _write_qsub(self, work_dir: str):
        qsub_script = textwrap.dedent(f"""\
        #!/bin/sh
        #PBS -r n
        #PBS -q {self.queue}
        #PBS -l nodes={self.nodes}
        #PBS -e /dev/null
        #PBS -o /dev/null

        cd "{work_dir}"

        rm -f START DONE FAILED
        echo "START $(date)" > START

        NPROCS=`wc -l < $PBS_NODEFILE`

        mpirun -v -machinefile $PBS_NODEFILE -np {self.np} {LAMMPS_EXECUTABLE} -in "{work_dir}/system.in" 1>out.system 2>&1
        rc=$?

        if [ $rc -eq 0 ]; then
        echo "DONE $(date)" > DONE
        else
        echo "FAILED rc=$rc $(date)" > FAILED
        fi

        exit $rc
        """)

        qsub_path = Path(work_dir) / "lammps.qsub"
        qsub_path.write_text(qsub_script)
        print(f"[LAMMPSRunner] Wrote qsub script: {qsub_path}")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if context.get("results", {}).get("lammps_input_status") != "ok":
            raise RuntimeError("LAMMPSRunner.run: refusing to submit because input generation did not succeed.")

        work_dir = context.get("work_dir")
        if not work_dir:
            raise RuntimeError("LAMMPSRunner.run: context['work_dir'] is missing.")

        print(f"\n=== LAMMPSRunner: preparing qsub in {work_dir} ===")
        self._write_qsub(work_dir)

        proc = subprocess.run(
            ["qas", "lammps.qsub"],
            cwd=work_dir,
            capture_output=True,
            text=True,
        )

        context["lammps_submit_stdout"] = proc.stdout
        context["lammps_submit_stderr"] = proc.stderr
        context["lammps_submit_returncode"] = proc.returncode
        context["lammps_submitted"] = (proc.returncode == 0)
        context["qsub_script"] = os.path.join(work_dir, "lammps.qsub")

        print("[LAMMPSRunner] submit rc=", proc.returncode)
        if proc.stdout:
            print(proc.stdout.strip())
        if proc.stderr:
            print(proc.stderr.strip())

        return context
