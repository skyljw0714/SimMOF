import subprocess
import random
import string
import re
from pathlib import Path
from config import RASPA_DIR as _RASPA_DIR, RASPA_SIMULATE_BIN
from core.job_manager import get_job_manager, parse_scheduler_job_id, record_job_event
from core.resource_allocator import ResourceAllocator, count_atoms_from_context

RASPA_DIR = _RASPA_DIR


class RASPARunner:
    
    def __init__(self, llm=None):
        self.llm = llm
        self.raspa_dir = RASPA_DIR

    def _make_unique_pbs_name(self, base_name: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_\-]", "_", base_name)
        rand = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
        max_len = 15
        remain = max_len - (len(rand) + 1)
        if remain < 0:
            return rand
        suffix = safe[:remain] if safe else ""
        return f"{rand}_{suffix}" if suffix else rand

    def _write_qsub_script(self, work_dir: Path, pbs_job_name: str, nodes_string: str, queue: str) -> Path:
        qsub_file = work_dir / "run_raspa.qsub"
        script = f"""#!/bin/sh
#PBS -N {pbs_job_name}
#PBS -r n
#PBS -q {queue}
#PBS -l {nodes_string}
#PBS -e {work_dir}/pbs.err
#PBS -o {work_dir}/pbs.out

cd $PBS_O_WORKDIR
echo "START $(date)" > START

{RASPA_SIMULATE_BIN} > output 2>&1
rc=$?

if [ $rc -eq 0 ]; then
  echo "DONE $(date)" > DONE
else
  echo "FAILED rc=$rc $(date)" > FAILED
fi

exit $rc
"""
        with open(qsub_file, "w") as sh:
            sh.write(script)
        return qsub_file

    def run(self, context: dict) -> dict:
        work_dir_str = context.get("work_dir")
        if not work_dir_str:
            raise ValueError("[RASPARunner] context['work_dir'] is missing.")

        work_dir = Path(work_dir_str)
        if not work_dir.is_dir():
            raise FileNotFoundError(f"[RASPARunner] work_dir does not exist: {work_dir}")

        sim_input = work_dir / "simulation.input"
        if not sim_input.is_file():
            raise FileNotFoundError(
                f"[RASPARunner] {sim_input} does not exist. "
                "Please check whether RASPAInputAgent ran first."
            )

        
        for marker in ("START", "DONE", "FAILED"):
            p = work_dir / marker
            try:
                if p.exists():
                    p.unlink()
            except Exception as e:
                print(f"[RASPARunner] Warning: could not remove marker {p}: {e}")

        base_job_name = context.get("job_name", "raspa_job")
        pbs_job_name = self._make_unique_pbs_name(base_job_name)
        context["pbs_job_name"] = pbs_job_name

        n_atoms = count_atoms_from_context(context)
        calc_type = context.get("property", "")
        spec = ResourceAllocator().recommend("RASPA", calc_type, n_atoms, context)
        qsub_file = self._write_qsub_script(work_dir, pbs_job_name, spec.pbs_nodes_string(), spec.queue)

        try:
            result = subprocess.run(
                ["qsub", str(qsub_file)],
                cwd=work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        except Exception as e:
            context.setdefault("results", {})
            context["results"]["raspa_submit_status"] = "error"
            context["results"]["raspa_submit_exception"] = str(e)
            context["raspa_status"] = "submit_failed"
            record_job_event(
                context,
                "submit_failed",
                message="RASPA qsub submission raised",
                metadata={"qsub_path": str(qsub_file)},
                last_error=str(e),
            )
            print("[RASPARunner] qsub submission failed:", e)
            return context

        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        if stderr:
            print(f"[RASPARunner] qsub stderr: {repr(stderr)}")

        context["results"] = dict(context.get("results") or {})
        context["results"]["raspa_qsub_file"] = str(qsub_file)
        context["results"]["raspa_submit_returncode"] = result.returncode
        context["results"]["raspa_submit_stdout"] = stdout
        context["results"]["raspa_submit_stderr"] = stderr
        context["scheduler_job_id"] = parse_scheduler_job_id(stdout)

        if result.returncode != 0:
            context["raspa_status"] = "submit_failed"
            context["raspa_job_id"] = None
            print("[RASPARunner] qsub returned non-zero code:", result.returncode)
        else:
            context["raspa_status"] = "submitted"
            context["raspa_job_id"] = context.get("scheduler_job_id")

        submit_status = "submitted" if result.returncode == 0 else "submit_failed"
        get_job_manager().record_submission(
            context,
            qsub_path=str(qsub_file),
            returncode=result.returncode,
            stdout=stdout,
            stderr=stderr,
            status=submit_status,
            metadata={
                "pbs_job_name": pbs_job_name,
                "software": "RASPA",
                "resource_allocation": {
                    "software": "RASPA",
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
        record_job_event(context, submit_status, message="RASPA qsub submission finished")

        return context
