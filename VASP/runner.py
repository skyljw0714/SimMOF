import os
import subprocess
from typing import Dict, Any, Optional


class VASPRunner:
    def __init__(self):
        pass

    def _submit_single_system(self, system_info: Dict[str, Any]) -> Dict[str, Any]:
        system_dir = system_info["dir"]
        label = system_info["label"]

        qsub_path = os.path.join(system_dir, f"{label}.qsub")
        result: Dict[str, Any] = {
            "label": label,
            "dir": system_dir,
            "qsub_path": qsub_path,
            "status": None,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "job_id": None,
        }

        if not os.path.exists(qsub_path):
            print(f"[VASPRunner] WARNING: qsub file not found: {qsub_path}")
            result["status"] = "missing_qsub"
            return result

        print(f"[VASPRunner] Submitting job for {label} in {system_dir}")
        try:
            proc = subprocess.run(
                ["qas", qsub_path],
                cwd=system_dir,
                capture_output=True,
                text=True,
            )
        except Exception as e:
            print(f"[VASPRunner] ERROR: failed to run qas {qsub_path}: {e}")
            result["status"] = "submit_error"
            result["stderr"] = str(e)
            return result

        result["returncode"] = proc.returncode
        result["stdout"] = proc.stdout
        result["stderr"] = proc.stderr

        if proc.returncode == 0:
            stdout = (proc.stdout or "").strip()
            job_id: Optional[str] = None
            if stdout:
                job_id = stdout.split()[0]
            result["job_id"] = job_id
            result["status"] = "submitted"
            print(f"[VASPRunner] Submitted {label}: job_id={job_id}")
        else:
            result["status"] = "failed"
            print(f"[VASPRunner] FAILED to submit {label}")
            if proc.stderr:
                print("  stderr:", proc.stderr.strip())

        return result

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
                
        system_info = context.get("vasp_system")

        
        if system_info is None:
            vasp_dir = context.get("vasp_dir")
            vasp_label = context.get("vasp_label")
            if not vasp_dir or not vasp_label:
                print("[VASPRunner] ERROR: missing vasp_system (or vasp_dir/vasp_label) in context.")
                context.setdefault("results", {})["vasp_run_status"] = "failed_no_system"
                return context
            system_info = {"dir": vasp_dir, "label": vasp_label}
        print(system_info)

        
        submit_res = self._submit_single_system(system_info)

        
        context["vasp_submit"] = submit_res
        context["vasp_job_id"] = submit_res.get("job_id")
        context["vasp_submitted"] = (submit_res.get("status") == "submitted")

        results = context.setdefault("results", {})
        results["vasp_run_status"] = submit_res.get("status", "unknown")
        results["vasp_submit_returncode"] = submit_res.get("returncode")

        return context