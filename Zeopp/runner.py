import os
import subprocess
from typing import Dict, Any
from config import working_dir

class ZeoppRunner:
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        work_dir = context.get("work_dir", working_dir)
        cmd      = context.get("zeopp_command")

        if not cmd:
            print("[ZeoppAgent] ERROR: zeopp_command is missing in context.")
            context.setdefault("results", {})["zeopp_status"] = "no_command"
            return context

        
        

        result = subprocess.run(
            cmd,
            shell=True,
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        results = context.setdefault("results", {})
        results["zeopp_returncode"] = result.returncode
        results["zeopp_stdout"] = result.stdout
        results["zeopp_stderr"] = result.stderr

        if result.returncode != 0:
            print(f"[ZeoppAgent] WARNING: Zeo++ exited with code {result.returncode}")
            if result.stdout:
                print("[ZeoppAgent] STDOUT (on error):\n", result.stdout)
            if result.stderr:
                print("[ZeoppAgent] STDERR (on error):\n", result.stderr)
            results["zeopp_status"] = "run_failed"
        else:
            
            results["zeopp_status"] = "ok"

        return context

