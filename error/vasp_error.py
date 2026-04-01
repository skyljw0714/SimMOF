import os
import re
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from langchain.schema import SystemMessage, HumanMessage
from config import LLM_DEFAULT, AGENT_LLM_MAP


class VASPErrorAgent:

    MAX_RETRY = 3

    def __init__(
        self,
        llm=None,
        max_lines: int = 250,
        wait_interval_sec: int = 30,
        wait_timeout_sec: int = 24 * 3600,
    ):
        self.llm = llm or AGENT_LLM_MAP.get("VASPErrorAgent", LLM_DEFAULT)
        self.max_lines = max_lines
        self.wait_interval_sec = wait_interval_sec
        self.wait_timeout_sec = wait_timeout_sec

    
    
    
    def _clear_flags(self, work_dir: Path) -> None:
        for fn in ("DONE", "FAILED", "START"):
            p = work_dir / fn
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

    def _is_finished(self, work_dir: Path) -> bool:
        return (work_dir / "DONE").exists() or (work_dir / "FAILED").exists()

    def _which_flag(self, work_dir: Path) -> Optional[str]:
        if (work_dir / "DONE").exists():
            return "DONE"
        if (work_dir / "FAILED").exists():
            return "FAILED"
        return None

    
    
    
    def _read_file(self, path: str) -> str:
        if not os.path.exists(path):
            return f"<< {path} not found >>"

        with open(path, "r", errors="ignore") as f:
            lines = f.readlines()

        if len(lines) > self.max_lines:
            half = self.max_lines // 2
            lines = lines[:half] + ["\n...\n"] + lines[-half:]
        return "".join(lines)

    def _read_tail(self, path: str, n_lines: int = 200) -> str:
        if not os.path.exists(path):
            return f"<< {path} not found >>"
        with open(path, "r", errors="ignore") as f:
            lines = f.readlines()
        return "".join(lines[-n_lines:])

    def _potcar_excerpt(self, path: str, head: int = 50, tail: int = 50) -> str:
        if not os.path.exists(path):
            return f"<< {path} not found >>"
        with open(path, "r", errors="ignore") as f:
            lines = f.readlines()
        if len(lines) <= head + tail:
            return "".join(lines)
        excerpt = lines[:head] + ["\n...\n"] + lines[-tail:]
        return "".join(excerpt)

    
    
    
    def _find_first_error_line(self, filepath: Path, patterns: List[str]) -> Optional[int]:
        if not filepath.is_file():
            return None
        try:
            with open(filepath, "r", errors="ignore") as f:
                for i, line in enumerate(f, start=1):
                    for p in patterns:
                        if re.search(p, line, flags=re.IGNORECASE):
                            return i
        except Exception:
            return None
        return None

    def _excerpt_around_line(self, filepath: Path, center: int, radius: int = 40) -> str:
        if not filepath.is_file():
            return f"<< {filepath} not found >>"

        with open(filepath, "r", errors="ignore") as f:
            lines = f.readlines()

        start = max(1, center - radius)
        end = min(len(lines), center + radius)

        chunk = []
        for ln in range(start, end + 1):
            chunk.append(f"{ln:6d}: {lines[ln-1]}")
        text = "".join(chunk)

        text_lines = text.splitlines(True)
        if len(text_lines) > self.max_lines:
            half = self.max_lines // 2
            text_lines = text_lines[:half] + ["\n...\n"] + text_lines[-half:]
            text = "".join(text_lines)

        return text

    
    
    
    def _detect_error(self, system_dir: Path) -> Tuple[bool, str, str]:
        out_txt = system_dir / "out.txt"
        outcar = system_dir / "OUTCAR"

        error_patterns = [
            r"Error reading item",
            r"VERY BAD NEWS",
            r"ZBRENT",
            r"BRIONS",
            r"internal error",
            r"Segmentation fault",
            r"forrtl:",
            r"^\s*ERROR\b",
            r"^\s*FATAL\b",
            r"\bnan\b",
            r"\bNaN\b",
            r"sub-space matrix is not hermitian",
            r"POSCAR.*invalid",
            r"POTCAR.*not found",
            r"EDDDAV",
            r"CNORMN",
            r"RSPHER",
            r"IERR=",
            r"ZPOTRF",
            r"MPI_ABORT",
            r"BRMIX",
        ]

        
        if out_txt.is_file():
            err_line = self._find_first_error_line(out_txt, error_patterns)
            if err_line is not None:
                excerpt = (
                    f"[out.txt error hit @ line {err_line}]\n"
                    + self._excerpt_around_line(out_txt, err_line, radius=50)
                    + "\n\n[out.txt tail]\n"
                    + self._read_tail(str(out_txt), n_lines=250)
                )
                return True, "out.txt", excerpt
            return False, "out.txt", self._read_tail(str(out_txt), n_lines=350)

        
        if outcar.is_file():
            err_line2 = self._find_first_error_line(outcar, error_patterns)
            if err_line2 is not None:
                excerpt = (
                    f"[OUTCAR error hit @ line {err_line2}]\n"
                    + self._excerpt_around_line(outcar, err_line2, radius=50)
                    + "\n\n[OUTCAR tail]\n"
                    + self._read_tail(str(outcar), n_lines=300)
                )
                return True, "OUTCAR", excerpt
            return False, "OUTCAR", self._read_tail(str(outcar), n_lines=350)

        
        return True, "FILES", "<< out.txt and OUTCAR not found >>"

    
    
    
    def _submit_qas(self, system_dir: Path, label: str) -> Dict[str, Any]:
        qsub_path = system_dir / f"{label}.qsub"
        result: Dict[str, Any] = {
            "label": label,
            "dir": str(system_dir),
            "qsub_path": str(qsub_path),
            "status": None,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "job_id": None,   
        }

        if not qsub_path.is_file():
            result["status"] = "missing_qsub"
            return result

        self._clear_flags(system_dir)

        try:
            proc = subprocess.run(
                ["qas", str(qsub_path)],
                cwd=str(system_dir),
                capture_output=True,
                text=True,
            )
        except Exception as e:
            result["status"] = "submit_error"
            result["stderr"] = str(e)
            return result

        result["returncode"] = proc.returncode
        result["stdout"] = proc.stdout or ""
        result["stderr"] = proc.stderr or ""

        if proc.returncode != 0:
            result["status"] = "failed"
            return result

        out = (proc.stdout or "").strip()
        m = re.search(r"(\d+(?:\.\w+)?)", out)
        if m:
            result["job_id"] = m.group(1)

        result["status"] = "submitted"
        return result

    
    
    
    def _call_llm_for_fix(self, error_source: str, error_text: str, file_dict: Dict[str, str]) -> str:
        system_prompt = (
            "You are a VASP troubleshooting assistant.\n"
            "Given error excerpts and VASP input files, propose minimal and safe edits.\n"
            "Prefer editing INCAR/KPOINTS first. Avoid modifying POTCAR content unless absolutely necessary.\n"
            "Output MUST follow the strict patch format:\n\n"
            "FILE: <filename>\n"
            "ACTION: <Replace|Remove|AddAfter|AddBefore|Append|Overwrite>\n"
            "PATTERN: <text or regex (optional)>\n"
            "CONTENT:\n"
            "```<text>```\n\n"
            "Notes:\n"
            "- Replace: replace first match of PATTERN with CONTENT.\n"
            "- Remove: remove first match of PATTERN.\n"
            "- AddAfter/AddBefore: insert CONTENT relative to first match of PATTERN.\n"
            "- Append: add CONTENT at end of file.\n"
            "- Overwrite: replace entire file with CONTENT.\n"
            "- Separate patches with '----' exactly.\n"
            "- DO NOT output empty assignments like 'IVDW ='.\n"
        )

        user_prompt = f"[ERROR SOURCE]\n{error_source}\n\n[ERROR EXCERPT]\n{error_text}\n\n"
        for fname, content in file_dict.items():
            user_prompt += f"\n----- {fname} -----\n{content}\n"

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        resp = self.llm.invoke(messages)
        return (resp.content or "").strip()

    
    
    
    def _parse_patch_blocks(self, patch_text: str) -> List[Dict[str, str]]:
        blocks = []
        for raw in patch_text.split("----"):
            b = raw.strip()
            if not b:
                continue
            if "FILE:" not in b or "ACTION:" not in b or "CONTENT:" not in b:
                continue

            def grab_field(name: str) -> str:
                m = re.search(rf"{name}:\s*(.*)", b)
                return (m.group(1).strip() if m else "")

            fname = grab_field("FILE")
            action = grab_field("ACTION")
            pattern = grab_field("PATTERN")

            m = re.search(r"CONTENT:\s*```([\s\S]*?)```", b)
            content = (m.group(1) if m else "").rstrip("\n")

            blocks.append({"file": fname, "action": action, "pattern": pattern, "content": content, "raw": b})
        return blocks

    def _apply_single_patch(self, filepath: Path, action: str, pattern: str, content: str) -> bool:
        if not filepath.exists() and action not in ("Overwrite", "Append"):
            print(f"[PATCH] {filepath} not found (action={action})")
            return False

        old_text = ""
        if filepath.exists():
            with open(filepath, "r", errors="ignore") as f:
                old_text = f.read()

        new_text = old_text
        changed = False
        action = action.strip()

        if action == "Overwrite":
            new_text = content.rstrip("\n") + "\n"
            changed = (new_text != old_text)

        elif action == "Append":
            add = content.rstrip("\n") + "\n"
            new_text = old_text + ("" if old_text.endswith("\n") else "\n") + add
            changed = (new_text != old_text)

        elif action in ("Replace", "Remove", "AddAfter", "AddBefore"):
            if not pattern:
                print(f"[PATCH] missing PATTERN for action={action} ({filepath.name})")
                return False

            m = re.search(pattern, old_text, flags=re.MULTILINE)
            if not m:
                if pattern in old_text:
                    idx = old_text.find(pattern)
                    start, end = idx, idx + len(pattern)
                else:
                    print(f"[PATCH] pattern not found in {filepath.name}: {pattern}")
                    return False
            else:
                start, end = m.span()

            if action == "Replace":
                new_text = old_text[:start] + content + old_text[end:]
                changed = True
            elif action == "Remove":
                new_text = old_text[:start] + old_text[end:]
                changed = True
            elif action == "AddAfter":
                new_text = old_text[:end] + ("\n" if old_text[end:end+1] != "\n" else "") + content + old_text[end:]
                changed = True
            elif action == "AddBefore":
                new_text = old_text[:start] + content + ("\n" if not content.endswith("\n") else "") + old_text[start:]
                changed = True
        else:
            print(f"[PATCH] unknown ACTION={action} ({filepath.name})")
            return False

        if changed and new_text != old_text:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w") as f:
                f.write(new_text)
            print(f"[PATCH] Updated: {filepath}")
            return True

        print(f"[PATCH] No change applied: {filepath}")
        return False

    def _apply_patches(self, system_dir: Path, patch_text: str) -> Dict[str, Any]:
        parsed = self._parse_patch_blocks(patch_text)
        applied, skipped = [], []

        for p in parsed:
            fname, action, pattern, content = p["file"], p["action"], p["pattern"], p["content"]
            if not fname:
                skipped.append(p)
                continue

            target = system_dir / fname

            
            if target.name.upper() == "POTCAR":
                print("[PATCH] Skipping POTCAR modification for safety.")
                skipped.append(p)
                continue

            ok = self._apply_single_patch(target, action, pattern, content)
            (applied if ok else skipped).append(p)

        return {"applied": applied, "skipped": skipped}

    
    
    
    def _run_single(self, context: Dict[str, Any]) -> Dict[str, Any]:
        
        sys_info = context.get("vasp_system")
        if sys_info is None:
            vasp_dir = context.get("vasp_dir")
            vasp_label = context.get("vasp_label")
            if not vasp_dir or not vasp_label:
                context.setdefault("results", {})["vasp_status"] = "no_system"
                return context
            sys_info = {"dir": vasp_dir, "label": vasp_label}
            context["vasp_system"] = sys_info

        system_dir = Path(sys_info["dir"])
        label = sys_info["label"]

        
        submit_info = context.get("vasp_submit", {}) or {}
        submitted_ok = (
            submit_info.get("status") == "submitted"
            or context.get("vasp_submitted") is True
        )
        if not submitted_ok:
            print(f"[VASPErrorAgent] job not properly submitted (label={label}, status={submit_info.get('status')})")
            context.setdefault("results", {})["vasp_status"] = "giveup_not_submitted"
            context["vasp_state"] = "giveup"
            return context

        
        retry = int(context.get("vasp_retry", 0) or 0)
        state = context.get("vasp_state", "pending") or "pending"

        start_time = time.time()
        deadline = start_time + self.wait_timeout_sec

        print(
            f"[VASPErrorAgent] Polling 1 system every {self.wait_interval_sec}s "
            f"(timeout={self.wait_timeout_sec}s) dir={system_dir}"
        )

        overall_failed = False

        while time.time() < deadline:
            if state in ("done_ok", "giveup"):
                break

            
            if not self._is_finished(system_dir):
                time.sleep(self.wait_interval_sec)
                continue

            flag = self._which_flag(system_dir)
            print(f"[VASPErrorAgent] {flag} detected.")
            
            if flag == "DONE":
                state = "done_ok"
                break

            
            has_err, err_src, err_excerpt = self._detect_error(system_dir)
            if not has_err:
                print("[VASPErrorAgent] FAILED but no clear error pattern -> giveup")
                state = "giveup"
                overall_failed = True
                break

            print(f"[VASPErrorAgent] ERROR detected (source={err_src})")

            if retry >= self.MAX_RETRY:
                print("[VASPErrorAgent] MAX_RETRY exceeded -> giveup")
                state = "giveup"
                overall_failed = True
                break

            file_dict = {
                "INCAR": self._read_file(str(system_dir / "INCAR")),
                "POSCAR": self._read_file(str(system_dir / "POSCAR")),
            }
            if (system_dir / "KPOINTS").is_file():
                file_dict["KPOINTS"] = self._read_file(str(system_dir / "KPOINTS"))
            if (system_dir / "POTCAR").is_file():
                file_dict["POTCAR(excerpt)"] = self._potcar_excerpt(str(system_dir / "POTCAR"))

            patch_text = self._call_llm_for_fix(err_src, err_excerpt, file_dict)
            print("\n[LLM PATCH SUGGESTION]\n", patch_text)

            patch_result = self._apply_patches(system_dir, patch_text)
            if len(patch_result.get("applied", [])) == 0:
                print("[VASPErrorAgent] 0 patches applied -> giveup this round.")
                retry += 1
                state = "giveup"
                overall_failed = True
                break

            
            submit_res = self._submit_qas(system_dir, label)
            context["vasp_submit"] = submit_res
            context["vasp_job_id"] = submit_res.get("job_id")
            context["vasp_submitted"] = (submit_res.get("status") == "submitted")

            retry += 1

            if submit_res.get("status") != "submitted":
                print(f"[VASPErrorAgent] resubmit failed (status={submit_res.get('status')})")
                state = "giveup"
                overall_failed = True
                break

            jid = submit_res.get("job_id")
            if jid:
                print(f"[VASPErrorAgent] resubmitted job_id={jid}")
            else:
                print("[VASPErrorAgent] resubmitted (job_id unknown)")

            
            state = "pending"
            time.sleep(self.wait_interval_sec)

        
        if state not in ("done_ok", "giveup") and time.time() >= deadline:
            print("[VASPErrorAgent] global timeout -> giveup")
            state = "giveup"
            overall_failed = True

        context["vasp_retry"] = retry
        context["vasp_state"] = state
        context.setdefault("results", {})["vasp_status"] = "partial_or_failed" if overall_failed else "ok"
        return context
        


    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        
        if "batch" in context and isinstance(context["batch"], list):
            return self.run_batch(context)
        return self._run_single(context)
