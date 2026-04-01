import subprocess
import random
import string
import re
from pathlib import Path
from config import RASPA_DIR as _RASPA_DIR, RASPA_SIMULATE_BIN

RASPA_QSUB_QUEUE = "long"
RASPA_QSUB_RESOURCES = "nodes=1:ppn=8:aa"
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

    def _write_qsub_script(self, work_dir: Path, pbs_job_name: str) -> Path:
        qsub_file = work_dir / "run_raspa.qsub"
        script = f"""#!/bin/sh
#PBS -N {pbs_job_name}
#PBS -r n
#PBS -q {RASPA_QSUB_QUEUE}
#PBS -l {RASPA_QSUB_RESOURCES}
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

        qsub_file = self._write_qsub_script(work_dir, pbs_job_name)

        try:
            result = subprocess.run(
                ["qas", str(qsub_file)],
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
            print("[RASPARunner] qas submission failed:", e)
            return context

        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        if stderr:
            print(f"[RASPARunner] qas stderr: {repr(stderr)}")

        context["results"] = dict(context.get("results") or {})
        context["results"]["raspa_qsub_file"] = str(qsub_file)
        context["results"]["raspa_submit_returncode"] = result.returncode
        context["results"]["raspa_submit_stdout"] = stdout
        context["results"]["raspa_submit_stderr"] = stderr

        if result.returncode != 0:
            context["raspa_status"] = "submit_failed"
            context["raspa_job_id"] = None
            print("[RASPARunner] qas returned non-zero code:", result.returncode)
        else:
            context["raspa_status"] = "submitted"

        return context
