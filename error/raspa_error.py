import os
import re
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

from langchain.schema import SystemMessage, HumanMessage
from config import LLM_DEFAULT, AGENT_LLM_MAP, RASPA_DIR as _RASPA_DIR
from RASPA.runner import RASPARunner

RASPA_DIR = Path(_RASPA_DIR)


class RASPAErrorAgent:
    MAX_TRIALS = 3  

    def __init__(self, llm=None, max_lines: int = 200):
        self.llm = llm or AGENT_LLM_MAP.get("RASPAErrorAgent", LLM_DEFAULT)
        self.max_lines = max_lines

    

    def _job_is_still_running_once(self, work_dir: Path) -> bool:
        if (work_dir / "DONE").exists() or (work_dir / "FAILED").exists():
            return False
        return True

    def _is_finished(self, work_dir: Path) -> bool:
        return (work_dir / "DONE").exists() or (work_dir / "FAILED").exists()

    def _clear_flags(self, work_dir: Path) -> None:
        for fn in ("START", "DONE", "FAILED"):
            p = work_dir / fn
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

    def _gather_error_text(self, context: Dict[str, Any]) -> str:
        work_dir_str = context.get("work_dir", "")
        if not work_dir_str:
            return "<< work_dir not set in context >>\n"

        work_dir = Path(work_dir_str)
        output_file = work_dir / "output"

        if output_file.is_file():
            return self.read_file(str(output_file))
        else:
            return f"<< {output_file} not found >>\n"

    def _raspa_has_error(self, work_dir: Path) -> Optional[bool]:
        done = (work_dir / "DONE").exists()
        failed = (work_dir / "FAILED").exists()

        if not done and not failed:
            
            return None

        if failed:
            print("[RASPAErrorAgent] FAILED flag exists → error.")
            return True

        
        output_file = work_dir / "output"
        if not output_file.is_file():
            print(f"[RASPAErrorAgent] DONE but '{output_file}' missing → error.")
            return True

        text = self.read_file(str(output_file))
        if "ERROR" in text or "Error" in text:
            print("[RASPAErrorAgent] 'ERROR' found in output → error.")
            return True

        print("[RASPAErrorAgent] DONE and no error strings → OK.")
        return False


    def read_file(self, filepath: str) -> str:
        try:
            with open(filepath, "r", errors="ignore") as f:
                lines = f.readlines()
        except FileNotFoundError:
            return f"<< {filepath} not found >>\n"

        if len(lines) > self.max_lines:
            lines = (
                lines[: self.max_lines // 2]
                + ["\n...\n"]
                + lines[-self.max_lines // 2 :]
            )
        return "".join(lines)

    def _read_cif_header_for_llm(self, filepath: str) -> str:
        try:
            with open(filepath, "r", errors="ignore") as f:
                lines = f.readlines()
        except FileNotFoundError:
            return f"<< {filepath} not found >>\n"

        header_lines = []
        in_atom_block = False

        for i, line in enumerate(lines):
            stripped = line.strip().lower()

            
            
            if stripped.startswith("_atom_site_"):
                in_atom_block = True

            if in_atom_block:
                
                break

            header_lines.append(line)

            
            if len(header_lines) > self.max_lines:
                header_lines.append("\n...\n")
                break

        return "".join(header_lines)


    def _find_framework_name_from_input(self, input_path: Path) -> Optional[str]:
        try:
            with open(input_path, "r") as f:
                for line in f:
                    stripped = line.strip().lower()
                    if stripped.startswith("frameworkname"):
                        parts = line.split()
                        if len(parts) >= 2:
                            return parts[-1].strip()
        except FileNotFoundError:
            return None
        return None

    

    def call_llm_for_fix(self, error_msg: str, file_dict: Dict[str, str]) -> str:
        system_prompt = (
            "You are a RASPA Monte Carlo simulation troubleshooting assistant.\n"
            "You will be given an ERROR message (from stdout/stderr) and one or more text files "
            "such as 'simulation.input' and CIF files.\n\n"
            "Your job is to propose the MINIMAL set of text edits needed to fix the error.\n"
            "Rules for your response:\n"
            "- Always provide the smallest number of changes necessary to resolve the ERROR.\n"
            "- Never suggest contradictory changes.\n"
            "- Do not propose cosmetic changes unless required.\n"
            "- Prefer editing 'simulation.input' first.\n"
            "- For CIF files, PREFER editing only cell / symmetry / space-group / header lines.\n"
            "  Avoid modifying the atom-site coordinate block unless the error explicitly mentions it.\n"
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

        user_prompt = f"ERROR message from RASPA (stdout/stderr):\n{error_msg}\n\n"
        for fname, content in file_dict.items():
            user_prompt += f"\n----- {fname} -----\n{content}\n"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        resp = self.llm.invoke(messages)
        return resp.content

    
    def patch_file(self, fname: str, block: str):
        try:
            with open(fname, "r", errors="ignore") as f:
                content = f.read()
        except FileNotFoundError:
            print(f"{fname} not found, cannot patch.")
            return

        original_content = content
        changed = False

        
        m = re.search(
            r"ACTION:\s*Overwrite entire file with:\s*```([\s\S]+?)```",
            block,
        )
        if m:
            new_content = m.group(1).strip() + "\n"
            content = new_content
            changed = True

        
        m = re.search(
            r"ACTION:\s*Append at end:\s*```([\s\S]+?)```",
            block,
        )
        if m:
            add = m.group(1).strip()
            if not content.endswith("\n"):
                content += "\n"
            content += add + "\n"
            changed = True

        
        m = re.search(
            r"ACTION:\s*Remove the line:\s*```([\s\S]+?)```",
            block,
        )
        if m:
            target = m.group(1).strip()
            if target in content:
                content = content.replace(target, "", 1)
                changed = True
            else:
                print(f"[patch_file] WARNING: remove-target not found in {fname}")

        
        m = re.search(
            r"ACTION:\s*Replace:\s*```([\s\S]+?)```\s*with:\s*```([\s\S]+?)```",
            block,
        )
        if m:
            old_block = m.group(1).strip()
            new_block = m.group(2).strip()
            if old_block in content:
                content = content.replace(old_block, new_block, 1)
                changed = True
            else:
                print(f"[patch_file] WARNING: old_block not found in {fname}")

        
        m = re.search(
            r"ACTION:\s*After the line:\s*```([\s\S]+?)```\s*add:\s*```([\s\S]+?)```",
            block,
        )
        if m:
            target = m.group(1).strip()
            insert = m.group(2).strip()
            if target in content:
                content = content.replace(
                    target,
                    target + "\n" + insert,
                    1,
                )
                changed = True
            else:
                print(f"[patch_file] WARNING: after-target not found in {fname}")

        
        m = re.search(
            r"ACTION:\s*Before the line:\s*```([\s\S]+?)```\s*add:\s*```([\s\S]+?)```",
            block,
        )
        if m:
            target = m.group(1).strip()
            insert = m.group(2).strip()
            if target in content:
                content = content.replace(
                    target,
                    insert + "\n" + target,
                    1,
                )
                changed = True
            else:
                print(f"[patch_file] WARNING: before-target not found in {fname}")

        if changed and content != original_content:
            with open(fname, "w") as f:
                f.write(content)
            print(f"{fname} has been automatically modified.")
        else:
            print(f"No applicable modifications found in {fname}.")


    def _run_core_once(self, context: Dict[str, Any]) -> Dict[str, Any]:
        status = context.get("raspa_status")
        if status != "failed":
            print(f"[RASPAErrorAgent] _run_core_once: raspa_status={status} -> nothing to do.")
            return context

        work_dir_str = context.get("work_dir")
        input_file_str = context.get("input_file")
        if not work_dir_str or not input_file_str:
            raise RuntimeError("[RASPAErrorAgent] work_dir or input_file is missing from context.")

        work_dir = Path(work_dir_str)
        input_path = Path(input_file_str)

        print(f"\n=== RASPAErrorAgent: error fixing in {work_dir} ===")

        error_text = self._gather_error_text(context)
        print("\n[LOG SNIPPET]\n", error_text[:2000], "\n")

        
        file_dict: Dict[str, str] = {}
        file_dict["simulation.input"] = self.read_file(str(input_path))

        fw_name = self._find_framework_name_from_input(input_path)
        if fw_name:
            candidates = [
                work_dir / f"{fw_name}.cif",
                RASPA_DIR / "share" / "raspa" / "structures" / "cif" / f"{fw_name}.cif",
            ]
            for c in candidates:
                if c.is_file():
                    file_dict[c.name] = self._read_cif_header_for_llm(str(c))
                    break

        
        fix_text = self.call_llm_for_fix(error_text, file_dict)
        print("\nLLM SUGGESTION:\n", fix_text)

        
        for block in fix_text.split("----"):
            if not block.strip():
                continue
            if "FILE:" not in block:
                continue
            fname_rel = block.split("FILE:")[1].split("\n")[0].strip()
            full_path = work_dir / fname_rel
            self.patch_file(str(full_path), block)

        context["raspa_status"] = "patched"
        context["raspa_fixed_once"] = True

        return context

    def run(self, context: dict) -> dict:
        batch = context.get("batch")
        created_batch_wrapper = False

        if not isinstance(batch, list) or len(batch) == 0:
            if context.get("work_dir") and context.get("input_file"):
                single = dict(context)
                single.pop("batch", None)
                batch = [single]
                created_batch_wrapper = True   
            else:
                raise ValueError("[RASPAErrorAgent] Neither batch nor work_dir/input_file is available.")


        
        interval_sec = int(context.get("raspa_poll_interval_sec", 20))
        timeout_sec  = int(context.get("raspa_poll_timeout_sec", 72 * 3600))
        deadline = time.time() + timeout_sec

        
        
        for item in batch:
            item.setdefault("raspa_retry", 0)
            item.setdefault("raspa_state", "pending")  

        print(f"[RASPAErrorAgent] batch polling start: n={len(batch)}, interval={interval_sec}s, timeout={timeout_sec}s")

        while time.time() < deadline:
            if all(it.get("raspa_state") in ("done_ok", "giveup") for it in batch):
                break

            progressed = False

            for it in batch:
                if it.get("raspa_state") in ("done_ok", "giveup"):
                    continue

                work_dir_str = it.get("work_dir")
                input_file_str = it.get("input_file")
                if not work_dir_str or not input_file_str:
                    it["raspa_state"] = "giveup"
                    it["raspa_status"] = "missing_paths"
                    progressed = True
                    continue

                work_dir = Path(work_dir_str)

                retry = int(it.get("raspa_retry", 0))

                has_err = self._raspa_has_error(work_dir)

                
                if has_err is None:
                    continue

                
                if has_err is False:
                    it["raspa_state"] = "done_ok"
                    it["raspa_status"] = "done_ok"
                    progressed = True
                    continue

                
                if retry >= self.MAX_TRIALS:
                    it["raspa_state"] = "giveup"
                    it["raspa_status"] = "giveup"
                    progressed = True
                    continue

                
                progressed = True

                
                
                local_ctx = dict(it)
                local_ctx.setdefault("results", it.get("results", {}))
                local_ctx["raspa_status"] = "failed"

                
                local_ctx = self._run_core_once(local_ctx)

                
                self._clear_flags(work_dir)

                
                raspa_runner = RASPARunner()
                local_ctx = raspa_runner.run(local_ctx)

                
                it["results"] = local_ctx.get("results", {})
                it["pbs_job_name"] = local_ctx.get("pbs_job_name")
                it["raspa_status"] = local_ctx.get("raspa_status")
                it["raspa_job_id"] = local_ctx.get("raspa_job_id")

                it["raspa_retry"] = retry + 1
                it["raspa_state"] = "pending"

            if not progressed:
                time.sleep(interval_sec)

        for it in batch:
            if it.get("raspa_state") not in ("done_ok", "giveup"):
                it["raspa_state"] = "giveup"
                it["raspa_status"] = "timeout"

        
        context.setdefault("results", {})
        n_ok = sum(1 for it in batch if it.get("raspa_state") == "done_ok")
        n_fail = sum(1 for it in batch if it.get("raspa_state") == "giveup")
        context["results"]["raspa_error_summary"] = {"done_ok": n_ok, "giveup": n_fail, "total": len(batch)}

        
        if created_batch_wrapper:
            sub = batch[0]
            context.setdefault("results", {})
            context["results"].update(sub.get("results", {}))
            context["raspa_status"] = sub.get("raspa_status", context.get("raspa_status"))
            context["raspa_retry"] = sub.get("raspa_retry", context.get("raspa_retry", 0))
            context["raspa_state"] = sub.get("raspa_state", context.get("raspa_state"))
            context.pop("batch", None)  
        else:
            
            context["batch"] = batch

        return context

