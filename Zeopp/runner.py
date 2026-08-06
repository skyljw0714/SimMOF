import json
import os
import subprocess
from typing import Dict, Any
from config import working_dir
from core.job_manager import record_job_event
from Zeopp.validation import store_validation_report, validate_zeopp_preflight

class ZeoppRunner:
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        work_dir = context.get("work_dir", working_dir)
        cmd      = context.get("zeopp_command")

        if not cmd:
            print("[ZeoppAgent] ERROR: zeopp_command is missing in context.")
            context.setdefault("results", {})["zeopp_status"] = "no_command"
            record_job_event(context, "input_failed", message="Zeo++ command missing")
            return context

        preflight = validate_zeopp_preflight(context)
        store_validation_report(
            context,
            preflight,
            key="zeopp_preflight_validation",
        )
        if not preflight["ok"]:
            results = context.setdefault("results", {})
            results["zeopp_returncode"] = None
            results["zeopp_stdout"] = ""
            results["zeopp_stderr"] = ""
            results["zeopp_process_status"] = "not_started"
            context["zeopp_submitted"] = False
            record_job_event(
                context,
                "validation_failed",
                message="Zeo++ pre-run validation failed",
                metadata={
                    "stage": preflight["stage"],
                    "issues": preflight["issues"],
                    "observations": preflight["observations"],
                },
                last_error=json.dumps(
                    {
                        "issues": preflight["issues"],
                        "observations": preflight["observations"],
                    },
                    ensure_ascii=False,
                )[:4000],
            )
            return context

        os.makedirs(work_dir, exist_ok=True)
        start_marker = os.path.join(work_dir, "START")
        done_marker = os.path.join(work_dir, "DONE")
        failed_marker = os.path.join(work_dir, "FAILED")
        stdout_path = os.path.join(work_dir, "zeopp.stdout")
        stderr_path = os.path.join(work_dir, "zeopp.stderr")
        returncode_path = os.path.join(work_dir, "zeopp.returncode")
        wrapper_path = os.path.join(work_dir, "zeopp_run.sh")

        wrapper = f"""#!/usr/bin/env bash
set +e
rm -f "{start_marker}" "{done_marker}" "{failed_marker}" "{stdout_path}" "{stderr_path}" "{returncode_path}"
echo "START $(date)" > "{start_marker}"
(
{cmd}
) > "{stdout_path}" 2> "{stderr_path}"
rc=$?
echo "$rc" > "{returncode_path}"
echo "DONE rc=$rc $(date)" > "{done_marker}"
if [ "$rc" -ne 0 ]; then
  echo "FAILED rc=$rc $(date)" > "{failed_marker}"
fi
exit "$rc"
"""
        with open(wrapper_path, "w", encoding="utf-8") as f:
            f.write(wrapper)
        os.chmod(wrapper_path, 0o755)

        proc = subprocess.Popen(
            ["bash", wrapper_path],
            cwd=work_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        results = context.setdefault("results", {})
        results["zeopp_status"] = "submitted"
        results["zeopp_pid"] = proc.pid
        results["zeopp_stdout_path"] = stdout_path
        results["zeopp_stderr_path"] = stderr_path
        results["zeopp_returncode_path"] = returncode_path
        results["zeopp_start_marker"] = start_marker
        results["zeopp_done_marker"] = done_marker
        results["zeopp_failed_marker"] = failed_marker
        context["zeopp_submitted"] = True
        record_job_event(
            context,
            "submitted",
            message="Zeo++ local process started",
            metadata={
                "pid": proc.pid,
                "command": cmd,
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
                "returncode_path": returncode_path,
            },
        )

        return context
