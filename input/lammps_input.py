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
    
    def _extract_guest_types_from_system_in(self, system_in_path):
        import re
        from pathlib import Path

        text = Path(system_in_path).read_text()
        m = re.search(r'^\s*group\s+guest\s+type\s+(.+)$', text, re.MULTILINE)
        if not m:
            return []

        return [int(x) for x in m.group(1).split()]

    def _infer_production_start_step_from_system_in(self, system_in_path):
        import re
        from pathlib import Path

        total = 0
        text = Path(system_in_path).read_text()

        for line in text.splitlines():
            s = line.strip()

            if re.match(r'^compute\s+msd_guest\b', s):
                break

            m = re.match(r'^run\s+(\d+)\b', s)
            if m:
                total += int(m.group(1))

        return total

    def _parse_masses_from_system_data(self, system_data_path):
        from pathlib import Path

        masses = {}
        lines = Path(system_data_path).read_text().splitlines()

        in_masses = False
        for line in lines:
            s = line.strip()

            if not s:
                continue

            if s.lower() == "masses":
                in_masses = True
                continue

            if in_masses:
                if s[0].isalpha():
                    break

                parts = s.split()
                if len(parts) >= 2:
                    try:
                        atype = int(parts[0])
                        mass = float(parts[1])
                        masses[atype] = mass
                    except ValueError:
                        pass

        return masses
    
    def _infer_dt_fs_from_system_in(self, system_in_path):
        import re
        from pathlib import Path

        text = Path(system_in_path).read_text()
        current_timestep = None

        for line in text.splitlines():
            s = line.strip()

            m = re.match(r'^timestep\s+([0-9Ee+.\-]+)\b', s)
            if m:
                current_timestep = float(m.group(1))

            if re.match(r'^compute\s+msd_guest\b', s):
                break

        if current_timestep is None:
            return 1.0

        return current_timestep

    def _inject_diffusivity_context(self, context):
        from pathlib import Path

        prop = str(context.get("property", "")).lower()
        if prop not in ["diffusivity", "diffusion", "self_diffusivity", "self_diffusion_coefficient"]:
            return context

        work_dir = Path(context["work_dir"])
        system_in_path = work_dir / "system.in"
        system_data_path = work_dir / "system.data"

        if not system_in_path.exists():
            raise RuntimeError(f"system.in not found: {system_in_path}")

        if not system_data_path.exists():
            raise RuntimeError(f"system.data not found: {system_data_path}")

        guest_types = self._extract_guest_types_from_system_in(system_in_path)
        production_start_step = self._infer_production_start_step_from_system_in(system_in_path)
        masses_by_type = self._parse_masses_from_system_data(system_data_path)
        dt_fs = self._infer_dt_fs_from_system_in(system_in_path)

        context["guest_types"] = guest_types
        context["production_start_step"] = production_start_step
        context["masses_by_type"] = masses_by_type
        context["dt_fs"] = dt_fs
        context.setdefault("fit_start_ps", 200.0)
        context.setdefault("fit_end_ps", None)

        return context

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

        context = self._inject_diffusivity_context(context)

        return context