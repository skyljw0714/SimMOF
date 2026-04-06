import os
import subprocess
import textwrap
import json
from pathlib import Path
from typing import Any, Dict, Optional

from config import WORKING_DIR

class LAMMPSInputAgent:
    def __init__(self):
        self.lammps_env_prefix = "/home/users/skyljw0714/anaconda3/envs/lammps"

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
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        guest_name = "" if guest_name is None else str(guest_name)
        query_text = "" if query_text is None else str(query_text)

        if simulation_input is None:
            simulation_input = {"present": False, "snippets": []}

        
        project_root_lit = json.dumps(project_root)
        working_dir_lit  = json.dumps(str(working_dir))
        mof_name_lit     = json.dumps(str(mof_name))
        guest_name_lit   = json.dumps(guest_name)
        prop_lit         = json.dumps(str(prop))
        job_name_lit     = json.dumps(str(job_name))
        query_text_lit   = json.dumps(query_text)

        sim_json = json.dumps(simulation_input, ensure_ascii=False)
        sim_json_lit = json.dumps(sim_json)

        guest_expr = f"{guest_name_lit} if {guest_name_lit} != \"\" else None"

        script = textwrap.dedent(f"""
            import sys, json
            sys.path.append({project_root_lit})
            from input.lammps.pipeline_lammps import generate_lammps_inputs

            simulation_input = json.loads({sim_json_lit})

            generate_lammps_inputs(
                working_dir={working_dir_lit},
                mof_name={mof_name_lit},
                guest_name={guest_expr},
                property_name={prop_lit},
                query_text={query_text_lit},
                simulation_input=simulation_input,
                num_guest={int(num_guest)},
                job_name={job_name_lit}
            )
        """)

        result = subprocess.run(
            ["conda", "run", "-p", self.lammps_env_prefix, "python", "-c", script],
            capture_output=True,
            text=True,
        )
        return result

    def run(self, context):
        paths = context.get("paths") if isinstance(context.get("paths"), dict) else {}
        # Canonical context path fields are top-level; nested paths fall back only for compatibility.
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
            guest_name=context["guest"],
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
