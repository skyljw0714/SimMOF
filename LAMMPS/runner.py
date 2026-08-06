import os
import subprocess
from pathlib import Path
from typing import Dict, Any
import textwrap

from config import LAMMPS_EXECUTABLE
from core.job_manager import get_job_manager, parse_scheduler_job_id, record_job_event
from core.resource_allocator import ResourceAllocator, count_atoms_from_context

class LAMMPSRunner:

    def __init__(self):
        pass

    def _write_qsub(self, work_dir: str, nodes_string: str, np: int, queue: str):
        qsub_script = textwrap.dedent(f"""\
        #!/bin/sh
        #PBS -r n
        #PBS -q {queue}
        #PBS -l {nodes_string}
        #PBS -e /dev/null
        #PBS -o /dev/null

        cd "{work_dir}"

        rm -f START DONE FAILED
        echo "START $(date)" > START

        NPROCS=`wc -l < $PBS_NODEFILE`

        mpirun -v -machinefile $PBS_NODEFILE -np {np} {LAMMPS_EXECUTABLE} -in "{work_dir}/system.in" 1>out.system 2>&1
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
        if context.get("lammps_status") == "needs_structure_from_user":
            context["lammps_submitted"] = False
            context.setdefault("results", {})[
                "lammps_submit_status"
            ] = "blocked_missing_structure"
            return context

        if context.get("results", {}).get("lammps_input_status") != "ok":
            raise RuntimeError("LAMMPSRunner.run: refusing to submit because input generation did not succeed.")

        work_dir = context.get("work_dir")
        if not work_dir:
            raise RuntimeError("LAMMPSRunner.run: context['work_dir'] is missing.")

        print(f"\n=== LAMMPSRunner: preparing qsub in {work_dir} ===")
        n_atoms = count_atoms_from_context(context)
        calc_type = context.get("property", "")
        spec = ResourceAllocator().recommend("LAMMPS", calc_type, n_atoms, context)
        self._write_qsub(work_dir, spec.pbs_nodes_string(), spec.np, spec.queue)

        proc = subprocess.run(
            ["qsub", "lammps.qsub"],
            cwd=work_dir,
            capture_output=True,
            text=True,
        )

        context["lammps_submit_stdout"] = proc.stdout
        context["lammps_submit_stderr"] = proc.stderr
        context["lammps_submit_returncode"] = proc.returncode
        context["lammps_submitted"] = (proc.returncode == 0)
        context["qsub_script"] = os.path.join(work_dir, "lammps.qsub")
        context["scheduler_job_id"] = parse_scheduler_job_id(proc.stdout or "")
        submit_status = "submitted" if proc.returncode == 0 else "submit_failed"
        get_job_manager().record_submission(
            context,
            qsub_path=context["qsub_script"],
            returncode=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            status=submit_status,
            metadata={
                "software": "LAMMPS",
                "resource_allocation": {
                    "software": "LAMMPS",
                    "calc_type": calc_type,
                    "n_atoms": n_atoms,
                    "nodes": spec.nodes,
                    "ppn": spec.ppn,
                    "np": spec.np,
                    "queue": spec.queue,
                    "rationale": spec.rationale,
                },
                "runtime_estimate": context.get("resource_runtime_estimate"),
                "resource_allocation_user_override": context.get("resource_allocation_user_override"),
            },
        )
        record_job_event(context, submit_status, message="LAMMPS qsub submission finished")

        print("[LAMMPSRunner] submit rc=", proc.returncode)
        if proc.stdout:
            print(proc.stdout.strip())
        if proc.stderr:
            print(proc.stderr.strip())

        return context
