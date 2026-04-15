import os
import re
import subprocess
import time

from collections import deque
from typing import Dict, Any
from config import AGENT_LLM_MAP, LLM_DEFAULT

from .agent import ErrorAgent


class LAMMPSErrorAgent(ErrorAgent):
    def __init__(self, llm=None, log_file="log.lammps", input_files=None, max_lines=200):
        self._init_error_agent(
            llm=llm,
            default_llm=AGENT_LLM_MAP.get("LAMMPSErrorAgent", LLM_DEFAULT),
            max_lines=max_lines,
        )
        self.log_file = log_file
        self.input_files = input_files or ["system.in", "system.in.settings", "system.in.init"]

    @staticmethod
    def _run_command(cmd: str, work_dir: str):
        print(f"\n>>> Running in {work_dir}: {cmd}")
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=work_dir,
            capture_output=True,
            text=True,
        )
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
        if result.returncode != 0:
            print(f"Command failed with code {result.returncode}")
        return result.returncode

    def extract_error(self, log_path, n=10, patterns=None):
        pats = patterns or [r"\bERROR\b"]
        try:
            with open(log_path, "r", errors="ignore") as f:
                tail_lines = list(deque(f, maxlen=n))
        except FileNotFoundError:
            return ""

        tail_text = "".join(tail_lines)
        if any(re.search(p, tail_text, flags=re.IGNORECASE) for p in pats):
            return tail_text.strip()
        return ""

    def call_llm_for_fix(self, error_msg, file_dict):
        system_prompt = (
            "You are a LAMMPS simulation troubleshooting assistant.\n"
            "Given an ERROR message and input files, for each fix you suggest, you MUST begin by stating the filename to fix "
            "using this format:\n"
            "Rules for your response:\n"
            "- Always provide the smallest number of changes necessary to resolve the ERROR.\n"
            "- Never suggest contradictory changes (e.g., both removing and re-adding the same line).\n"
            "- Never duplicate the same command (e.g., do not add multiple identical `kspace_style` lines).\n"
            "- Do not propose cosmetic changes unless they are required for correctness.\n"
            "- Treat `kspace_style` as a solver-initialization command dependent on final box geometry.\n"
            "- Never suggest adding or redefining `kspace_style` after a `minimize` or `run` command.\n"
            "- If fixing `kspace_style`, place it AFTER box geometry is finalized and STRICTLY BEFORE the first `minimize` or `run`.\n"
            "If the log shows `1 atoms in group guest`, the failing command is `velocity guest create ...`, and the error is `Attempting to rescale a 0.0 temperature`, remove the `velocity guest create ...` line entirely instead of modifying it.\n"
            "\n"
            "Output format (strict):\n"
            "FILE: <filename>\n"
            "ACTION: <pattern description>\n"
            "SUGGESTED CHANGE:\n<payload>\n"
            "Use ONLY ONE of these action patterns for each fix:\n"
            "1. After the line:\n```<text>```\nadd:\n```<text to insert>```\n"
            "2. Before the line:\n```<text>```\nadd:\n```<text to insert>```\n"
            "3. Remove the line:\n```<exact line to remove>```\n"
            "4. Replace:\n```<old line(s)>```\nwith:\n```<new line(s)>```\n"
            "5. Append at end:\n```<text to append>```\n"
            "6. Overwrite entire file with:\n```<new content>```\n"
            "For EACH fix, output a separate block as above.\n"
            "If there are multiple fixes, SEPARATE EACH BLOCK by exactly four dashes `----` on a line by themselves.\n"
            "Do NOT use any other separator between blocks except `----`.\n"
            "Return your response STRICTLY as described above."
        )

        user_prompt = f"ERROR message from LAMMPS log:\n{error_msg}\n\n"
        for fname, content in file_dict.items():
            user_prompt += f"\n----- {fname} -----\n{content}\n"

        return self._invoke_llm(system_prompt, user_prompt)

    def run(self, context: dict):
        work_dir = context.get("work_dir")
        if not work_dir:
            raise RuntimeError("LammpsErrorAgent.run: context['work_dir'] is missing.")

        print(f"\n=== LammpsErrorAgent: error loop in {work_dir} ===")

        base_files = self.input_files or ["system.in", "system.in.settings", "system.in.init"]
        abs_files = [os.path.join(work_dir, f) for f in base_files]

        log_path = os.path.join(work_dir, self.log_file)

        max_trials = 5
        success = False
        max_wait_sec = 3600 * 12
        poll_interval = 60

        first_already_submitted = bool(context.get("lammps_submitted", False))

        for attempt in range(1, max_trials + 1):
            print(f"\nAttempt #{attempt}: Running LAMMPS job")
            err = ""

            if attempt == 1 and first_already_submitted:
                print("[LAMMPSErrorAgent] First job already submitted by LAMMPSAgent. Skip submit.")
            else:
                
                for fn in ["START", "DONE", "FAILED"]:
                    p = os.path.join(work_dir, fn)
                    if os.path.exists(p):
                        os.remove(p)

                self._run_command("qas lammps.qsub", work_dir=work_dir)

            
            finished = False
            waited = 0

            while waited < max_wait_sec:
                time.sleep(poll_interval)
                waited += poll_interval

                done_path = os.path.join(work_dir, "DONE")
                failed_path = os.path.join(work_dir, "FAILED")

                if os.path.exists(done_path):
                    print("\nDONE detected. LAMMPS finished successfully.")
                    success = True
                    err = ""
                    finished = True
                    break

                if os.path.exists(failed_path):
                    print("\nFAILED detected. LAMMPS failed.")
                    success = False

                    
                    err = self.extract_error(log_path, n=80)
                    if not err:
                        err = self.read_file(log_path)

                    finished = True
                    break

                print(f"[poll] waiting for DONE/FAILED... ({waited}s)")

            
            if not finished:
                print("Timeout: DONE/FAILED not created.")
                success = False
                err = "Timeout: DONE/FAILED not created."

            
            if success:
                break

            
            print(f"\nLAMMPS ERROR detected on attempt #{attempt}:\n{err}\n")

            file_dict = {f: self.read_file(f) for f in abs_files}
            fix = self.call_llm_for_fix(err, file_dict)
            print("\nLLM SUGGESTION:\n", fix)

            for block in fix.split("----"):
                if not block.strip():
                    continue
                if "FILE:" in block:
                    fname_rel = block.split("FILE:")[1].split("\n")[0].strip()
                    full_path = os.path.join(work_dir, fname_rel)
                    self.patch_file(full_path, block)

            print("\nAuto-patch applied. Proceeding to next attempt.")

            first_already_submitted = False

        else:
            print("\nMaximum attempts reached. Manual intervention required.")

        context["lammps_success"] = success
        return context
