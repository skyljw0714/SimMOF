import os
import subprocess
import traceback
import io
from pathlib import Path
from typing import Any, Dict, Optional
from contextlib import redirect_stdout, redirect_stderr

from config import WORKING_DIR
from input.lammps.pipeline_lammps import generate_lammps_inputs

class LAMMPSInputAgent:
    def __init__(self):
        pass

    def _run_generate_lammps_inputs(
        self,
        working_dir,
        mof_name,
        guest_name,
        prop,
        query_text,
        num_guest,
        job_name,
        simulation_input: Optional[Dict[str, Any]] = None,
    ):
        guest_name = "" if guest_name is None else str(guest_name)
        query_text = "" if query_text is None else str(query_text)

        if simulation_input is None:
            simulation_input = {"present": False, "snippets": []}

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        try:
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                generate_lammps_inputs(
                    working_dir=str(working_dir),
                    mof_name=str(mof_name),
                    guest_name=guest_name if guest_name != "" else None,
                    property_name=str(prop),
                    query_text=query_text,
                    simulation_input=simulation_input,
                    num_guest=int(num_guest),
                    job_name=str(job_name),
                )

            return subprocess.CompletedProcess(
                args=["generate_lammps_inputs"],
                returncode=0,
                stdout=stdout_buffer.getvalue(),
                stderr=stderr_buffer.getvalue(),
            )

        except Exception:
            traceback.print_exc(file=stderr_buffer)
            return subprocess.CompletedProcess(
                args=["generate_lammps_inputs"],
                returncode=1,
                stdout=stdout_buffer.getvalue(),
                stderr=stderr_buffer.getvalue(),
            )

    def run(self, context):
        paths = context.get("paths") if isinstance(context.get("paths"), dict) else {}

        work_dir = context.get("work_dir")
        if work_dir is None:
            work_dir = paths.get("work_dir")

        if work_dir is None:
            plan_root = context.get("plan_root")
            if plan_root is None:
                plan_root = paths.get("plan_root")
            if plan_root:
                work_dir = str(Path(plan_root))
            else:
                work_dir = str(Path(WORKING_DIR) / context["job_name"])

        os.makedirs(work_dir, exist_ok=True)
        context["work_dir"] = work_dir

        print(f"LAMMPS input will be stored in: {work_dir}")

        sim_in = context.get("simulation_input")
        if sim_in is None:
            sim_in = {"present": False, "snippets": []}

        result = self._run_generate_lammps_inputs(
            working_dir=work_dir,
            mof_name=context["mof"],
            guest_name=context.get("guest"),
            prop=context["property"],
            query_text=context.get("query_text", ""),
            num_guest=context.get("num_guest", 1),
            job_name=context.get("job_name", ""),
            simulation_input=sim_in,
        )

        print("[LAMMPSInputAgent] generate_lammps_inputs")
        print("returncode =", result.returncode)
        if result.stdout:
            print("STDOUT:\n", result.stdout)
        if result.stderr:
            print("STDERR:\n", result.stderr)

        results = context.setdefault("results", {})
        results["lammps_input_status"] = "ok" if result.returncode == 0 else "failed"
        results["lammps_input_returncode"] = result.returncode
        results["lammps_input_stdout"] = result.stdout
        results["lammps_input_stderr"] = result.stderr

        if result.returncode != 0:
            raise RuntimeError("LAMMPS input generation failed; skipping submission.")

        return context