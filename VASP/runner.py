import os
import subprocess
from typing import Dict, Any, Optional

from core.job_manager import get_job_manager, record_job_event


class VASPRunner:
    def __init__(self):
        pass

    def _get_active_system_info(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        system_info = context.get("vasp_system")
        if not (isinstance(system_info, dict) and system_info.get("dir")):
            vasp_dir = context.get("vasp_dir")
            if not vasp_dir:
                return None
            system_info = {
                "dir": vasp_dir,
                "label": context.get("vasp_label") or context.get("mof") or "vasp_job",
            }
            if context.get("vasp_role"):
                system_info["role"] = context.get("vasp_role")

        system_info.setdefault("label", context.get("vasp_label") or context.get("mof") or "vasp_job")
        if context.get("vasp_role") and not system_info.get("role"):
            system_info["role"] = context.get("vasp_role")

        context["vasp_system"] = system_info
        context["vasp_dir"] = system_info["dir"]
        context["vasp_label"] = system_info["label"]
        if system_info.get("role"):
            context["vasp_role"] = system_info["role"]

        return system_info

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
                ["qsub", qsub_path],
                cwd=system_dir,
                capture_output=True,
                text=True,
            )
        except Exception as e:
            print(f"[VASPRunner] ERROR: failed to run qsub {qsub_path}: {e}")
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
        if context.get("vasp_status") == "needs_structure_from_user":
            context["vasp_submitted"] = False
            context["vasp_submit"] = {
                "status": "blocked_missing_structure"
            }
            context.setdefault("results", {})[
                "vasp_run_status"
            ] = "blocked_missing_structure"
            return context

        if context.get("vasp_structure_precheck_status") == "failed":
            print("[VASPRunner] Refusing to submit because VASP structure precheck failed.")
            context.setdefault("results", {})["vasp_run_status"] = "failed_structure_precheck"
            context["vasp_submitted"] = False
            context["vasp_submit"] = {"status": "failed_structure_precheck"}
            return context

        system_info = self._get_active_system_info(context)
        if system_info is None:
            print("[VASPRunner] ERROR: missing vasp_system (or vasp_dir/vasp_label) in context.")
            context.setdefault("results", {})["vasp_run_status"] = "failed_no_system"
            return context
        print(system_info)

        
        submit_res = self._submit_single_system(system_info)

        
        context["vasp_submit"] = submit_res
        context["vasp_job_id"] = submit_res.get("job_id")
        context["scheduler_job_id"] = submit_res.get("job_id")
        context["vasp_submitted"] = (submit_res.get("status") == "submitted")

        results = context.setdefault("results", {})
        results["vasp_run_status"] = submit_res.get("status", "unknown")
        results["vasp_submit_returncode"] = submit_res.get("returncode")
        submit_status = "submitted" if submit_res.get("status") == "submitted" else "submit_failed"
        get_job_manager().record_submission(
            context,
            qsub_path=submit_res.get("qsub_path", ""),
            returncode=submit_res.get("returncode") or -1,
            stdout=submit_res.get("stdout") or "",
            stderr=submit_res.get("stderr") or "",
            status=submit_status,
            scheduler_job_id=submit_res.get("job_id"),
            metadata={
                "software": "VASP",
                "vasp_label": submit_res.get("label"),
                "resource_allocation": context.get("resource_allocation"),
                "runtime_estimate": context.get("resource_runtime_estimate"),
                "resource_allocation_user_override": context.get("resource_allocation_user_override"),
            },
        )
        record_job_event(context, submit_status, message="VASP qsub submission finished")

        return context
