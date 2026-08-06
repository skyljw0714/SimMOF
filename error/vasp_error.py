import os
import re
import time
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from config import LLM_DEFAULT, AGENT_LLM_MAP
from core.job_manager import get_job_manager, record_job_event, record_scheduler_status
from core.llm_logging import log_llm_decision

from .agent import ErrorAgent
from .structure_regeneration import request_structure_regeneration


class VASPErrorAgent(ErrorAgent):
    def _get_active_system_info(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        sys_info = context.get("vasp_system")
        if not (isinstance(sys_info, dict) and sys_info.get("dir")):
            vasp_dir = context.get("vasp_dir")
            vasp_label = context.get("vasp_label")
            if not vasp_dir or not vasp_label:
                return None
            sys_info = {"dir": vasp_dir, "label": vasp_label}
            if context.get("vasp_role"):
                sys_info["role"] = context.get("vasp_role")

        sys_info.setdefault("label", context.get("vasp_label") or context.get("mof") or "vasp_job")
        if context.get("vasp_role") and not sys_info.get("role"):
            sys_info["role"] = context.get("vasp_role")

        context["vasp_system"] = sys_info
        context["vasp_dir"] = sys_info["dir"]
        context["vasp_label"] = sys_info["label"]
        if sys_info.get("role"):
            context["vasp_role"] = sys_info["role"]

        return sys_info


    MAX_RETRY = 3

    def __init__(
        self,
        llm=None,
        max_lines: int = 250,
        wait_interval_sec: int = 60,
    ):
        self._init_error_agent(
            llm=llm,
            default_llm=AGENT_LLM_MAP.get("VASPErrorAgent", LLM_DEFAULT),
            max_lines=max_lines,
        )
        self.wait_interval_sec = wait_interval_sec

    
    
    
    
    
    
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

    def _error_patterns(self) -> List[str]:
        patterns: List[str] = []
        try:
            from rag.vasp_error_knowledge import VASPErrorKnowledgeBase

            for entry in VASPErrorKnowledgeBase().entries:
                for pattern in entry.get("patterns", []) or []:
                    if pattern:
                        patterns.append(re.escape(str(pattern)))
        except Exception as exc:
            print(f"[VASPErrorAgent] Could not load RAG error patterns: {exc}")

        deduped = []
        seen = set()
        for pattern in patterns:
            if pattern not in seen:
                deduped.append(pattern)
                seen.add(pattern)
        return deduped

    
    def _detect_error(self, system_dir: Path) -> Tuple[bool, str, str]:
        out_txt = system_dir / "out.txt"
        outcar = system_dir / "OUTCAR"
        error_patterns = self._error_patterns()

        for log_path in (out_txt, outcar):
            err_line = self._find_first_error_line(log_path, error_patterns)
            if err_line is not None:
                excerpt = (
                    f"[{log_path.name} RAG-pattern hit @ line {err_line}]\n"
                    + self._excerpt_around_line(log_path, err_line, radius=50)
                    + f"\n\n[{log_path.name} tail]\n"
                    + self._read_tail(str(log_path), n_lines=250)
                )
                return True, log_path.name, excerpt

        existing_logs = [path for path in (out_txt, outcar) if path.is_file()]
        if existing_logs:
            excerpt = (
                "[FAILED marker detected; no exact local RAG pattern matched. "
                "The LLM must diagnose the supplied log tails using retrieved evidence.]\n\n"
                + "\n\n".join(
                    f"[{path.name} tail]\n{self._read_tail(str(path), n_lines=350)}"
                    for path in existing_logs
                )
            )
            return True, "+".join(path.name for path in existing_logs), excerpt

        return True, "FILES", "<< out.txt and OUTCAR not found >>"

    
    
    
    def _submit_qsub(self, system_dir: Path, label: str) -> Dict[str, Any]:
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
                ["qsub", str(qsub_path)],
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

    
    
    
    def _retrieve_error_knowledge_hits(
        self,
        error_source: str,
        error_text: str,
        file_dict: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        query_parts = [error_source, error_text[:3000]]
        incar = file_dict.get("INCAR")
        if incar:
            query_parts.append(incar[:1200])
        query = "\n".join(query_parts)

        try:
            from rag.vasp_error_knowledge import VASPErrorKnowledgeBase

            kb = VASPErrorKnowledgeBase()
            return kb.search(query, top_k=5)
        except Exception as exc:
            print(f"[VASPErrorAgent] VASP error knowledge retrieval skipped: {exc}")
            return []

    def _format_error_knowledge(self, hits: List[Dict[str, Any]]) -> str:
        if not hits:
            return ""
        try:
            from rag.vasp_error_knowledge import VASPErrorKnowledgeBase

            return VASPErrorKnowledgeBase().format_hits(hits, max_chars=4500)
        except Exception as exc:
            print(f"[VASPErrorAgent] VASP error knowledge formatting skipped: {exc}")
            return ""

    def _retrieve_error_knowledge(self, error_source: str, error_text: str, file_dict: Dict[str, str]) -> str:
        return self._format_error_knowledge(
            self._retrieve_error_knowledge_hits(error_source, error_text, file_dict)
        )

    def _looks_like_poscar(self, path: Path) -> bool:
        if not path.is_file():
            return False
        try:
            lines = path.read_text(errors="ignore").splitlines()
        except Exception:
            return False
        if len(lines) < 8:
            return False
        try:
            float(lines[1].split()[0])
            for i in range(2, 5):
                parts = lines[i].split()
                if len(parts) < 3:
                    return False
                [float(x) for x in parts[:3]]
        except Exception:
            return False
        return True

    def _backup_file(self, path: Path, suffix: str) -> Optional[str]:
        if not path.exists():
            return None
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup = path.with_name(f"{path.name}.simmof_{suffix}_{ts}")
        try:
            shutil.copy2(path, backup)
            return str(backup)
        except Exception as exc:
            print(f"[VASPErrorAgent] could not back up {path.name}: {exc}")
            return None

    @staticmethod
    def _response_field(block: str, name: str) -> Optional[str]:
        match = re.search(
            rf"(?ms)^\s*{re.escape(name)}\s*:\s*(.*?)"
            rf"(?=^\s*(?:RATIONALE|EVIDENCE|DECISION|FILE|ACTION|TOOL ACTION|SOURCE|DESTINATION|TARGET|TYPE MAPPINGS)\s*:|\Z)",
            block or "",
        )
        value = match.group(1).strip() if match else ""
        return value or None

    @staticmethod
    def _safe_root_filename(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        name = value.strip()
        path = Path(name)
        if path.is_absolute() or len(path.parts) != 1 or name in {".", ".."}:
            return None
        return name

    def _apply_llm_recovery_response(
        self,
        system_dir: Path,
        response: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        applied: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []
        structure_requested = False

        for raw in response.split("----"):
            block = raw.strip()
            if not block:
                continue

            rationale = self._response_field(block, "RATIONALE")
            evidence = self._response_field(block, "EVIDENCE")
            decision = (self._response_field(block, "DECISION") or "").upper()
            report = {"rationale": rationale, "evidence": evidence}
            if not rationale or not evidence:
                skipped.append({**report, "raw": block, "reason": "missing_rationale_or_evidence"})
                continue

            if decision == "TEXT_PATCH":
                fname = self._safe_root_filename(self._response_field(block, "FILE"))
                if not fname:
                    skipped.append({**report, "raw": block, "reason": "unsafe_or_missing_file"})
                    continue
                if fname.upper() in {"POSCAR", "POTCAR"}:
                    skipped.append({
                        **report,
                        "file": fname,
                        "raw": block,
                        "reason": "structure_or_potential_text_patch_not_allowed",
                    })
                    continue

                target = system_dir / fname
                before = target.read_text(errors="ignore") if target.exists() else None
                self.patch_file(str(target), block)
                after = target.read_text(errors="ignore") if target.exists() else None
                if before != after:
                    applied.append({**report, "decision": decision, "file": fname, "raw": block})
                else:
                    skipped.append({**report, "file": fname, "raw": block, "reason": "no_change"})
                continue

            if decision != "TOOL_ACTION":
                skipped.append({**report, "raw": block, "reason": "unsupported_decision"})
                continue

            tool_action = (self._response_field(block, "TOOL ACTION") or "").upper()
            if tool_action == "COPY_FILE":
                source = self._safe_root_filename(self._response_field(block, "SOURCE"))
                destination = self._safe_root_filename(self._response_field(block, "DESTINATION"))
                if (source, destination) != ("CONTCAR", "POSCAR"):
                    skipped.append({**report, "raw": block, "reason": "copy_not_allowlisted"})
                    continue
                source_path = system_dir / source
                destination_path = system_dir / destination
                if not self._looks_like_poscar(source_path):
                    skipped.append({**report, "raw": block, "reason": "invalid_or_missing_contcar"})
                    continue
                backup = self._backup_file(destination_path, "backup_before_llm_copy")
                shutil.copy2(source_path, destination_path)
                applied.append({
                    **report,
                    "decision": decision,
                    "tool_action": tool_action,
                    "source": source,
                    "destination": destination,
                    "backup": backup,
                })
                continue

            if tool_action == "DELETE_FILE":
                target_name = self._safe_root_filename(self._response_field(block, "TARGET"))
                if target_name not in {"CHGCAR", "WAVECAR"}:
                    skipped.append({**report, "raw": block, "reason": "delete_not_allowlisted"})
                    continue
                target = system_dir / target_name
                if not target.exists():
                    skipped.append({**report, "raw": block, "reason": "delete_target_missing"})
                    continue
                backup = self._backup_file(target, "backup_before_llm_delete")
                target.unlink()
                applied.append({
                    **report,
                    "decision": decision,
                    "tool_action": tool_action,
                    "target": target_name,
                    "backup": backup,
                })
                continue

            if tool_action == "REQUEST_STRUCTURE_REGENERATION":
                metadata = {
                    **report,
                    "decision": decision,
                    "tool_action": tool_action,
                }
                context["vasp_needs_structure_regeneration"] = True
                context["vasp_structure_regeneration_reason"] = metadata
                request_structure_regeneration(
                    context,
                    software="vasp",
                    reason=rationale,
                    action="regenerate_complex_geometry",
                    metadata=metadata,
                )
                applied.append(metadata)
                structure_requested = True
                continue

            if tool_action == "REGENERATE_POTCAR":
                poscar_path = system_dir / "POSCAR"
                try:
                    species = self._poscar_species_order(poscar_path)
                    if not species:
                        raise ValueError("POSCAR does not contain a VASP5 species-name line")
                    from config import VASP_POTENTIAL_DIR_PATH
                    from file.agent import VASPFileAgent

                    backup = self._backup_file(
                        system_dir / "POTCAR",
                        "backup_before_llm_regeneration",
                    )
                    sources = VASPFileAgent.species_to_potcar(
                        species,
                        str(system_dir) + os.sep,
                        str(VASP_POTENTIAL_DIR_PATH) + os.sep,
                    )
                    applied.append({
                        **report,
                        "decision": decision,
                        "tool_action": tool_action,
                        "species_order": species,
                        "potential_sources": sources,
                        "backup": backup,
                    })
                except Exception as exc:
                    skipped.append({
                        **report,
                        "raw": block,
                        "reason": "potcar_regeneration_failed",
                        "error": str(exc),
                    })
                continue

            skipped.append({**report, "raw": block, "reason": "unsupported_tool_action"})

        return {
            "applied": applied,
            "skipped": skipped,
            "structure_requested": structure_requested,
        }

    @staticmethod
    def _poscar_species_order(path: Path) -> List[str]:
        if not path.is_file():
            return []
        lines = path.read_text(errors="ignore").splitlines()
        if len(lines) < 7:
            return []
        species = lines[5].split()
        counts = lines[6].split()
        if not species or len(species) != len(counts):
            return []
        if not all(re.fullmatch(r"[A-Z][a-z]?", item) for item in species):
            return []
        try:
            if not all(int(item) >= 0 for item in counts):
                return []
        except ValueError:
            return []
        return species

    def _call_llm_for_fix(
        self,
        error_source: str,
        error_text: str,
        file_dict: Dict[str, str],
        rag_evidence: str = "",
        runtime_facts: str = "",
    ) -> str:
        system_prompt = (
            "You are a VASP troubleshooting assistant. This simulation uses VASP 5.4.1.\n"
            "Diagnose the failure from the log, input files, and retrieved local VASP recovery evidence. "
            "Do not use a memorized error-to-fix mapping when the retrieved evidence does not support it.\n"
            "Choose the minimal, safest recovery. Text edits, restart-file operations, and structure regeneration "
            "must all be explicitly selected in your response; no downstream rule-based recovery policy will infer them.\n"
            "Prefer editing INCAR and KPOINTS when the evidence supports an input-only repair. Never modify POTCAR.\n"
            "Never text-edit POSCAR or invent lattice vectors or atomic coordinates. If the diagnosed failure "
            "requires changing geometry, select REQUEST_STRUCTURE_REGENERATION so the controlled structure workflow "
            "can reacquire, clean, generate, or request a structure according to its provenance.\n"
            "When POTCAR is missing or its potential count/species order is incompatible with POSCAR, select "
            "REGENERATE_POTCAR so the trusted configured PAW library can restage potentials in POSCAR species order. "
            "Do not write or synthesize POTCAR text.\n"
            "Do not emit empty assignments like 'IVDW ='.\n"
            "Every block must report RATIONALE and EVIDENCE. EVIDENCE must cite a retrieved item such as "
            "'[1] custodian error_key=zbrent' and the matching log/input fact. If retrieval has no useful match, "
            "state that explicitly and cite the exact log/input fact used.\n"
            "\n"
            "For a text edit, output this strict format:\n"
            "RATIONALE: <why this action addresses the observed failure>\n"
            "EVIDENCE: <retrieved evidence id/source plus matching log or input fact>\n"
            "DECISION: TEXT_PATCH\n"
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
            "\nFor a non-text operation, output one of these strict blocks:\n"
            "RATIONALE: <reason>\nEVIDENCE: <retrieved evidence plus matching runtime fact>\n"
            "DECISION: TOOL_ACTION\nTOOL ACTION: COPY_FILE\nSOURCE: CONTCAR\nDESTINATION: POSCAR\n"
            "or\n"
            "RATIONALE: <reason>\nEVIDENCE: <retrieved evidence plus matching runtime fact>\n"
            "DECISION: TOOL_ACTION\nTOOL ACTION: DELETE_FILE\nTARGET: <CHGCAR or WAVECAR>\n"
            "or\n"
            "RATIONALE: <reason>\nEVIDENCE: <retrieved evidence plus matching runtime fact>\n"
            "DECISION: TOOL_ACTION\nTOOL ACTION: REQUEST_STRUCTURE_REGENERATION\n"
            "or\n"
            "RATIONALE: <reason>\nEVIDENCE: <retrieved evidence plus matching runtime fact>\n"
            "DECISION: TOOL_ACTION\nTOOL ACTION: REGENERATE_POTCAR\n"
            "Use structure regeneration only when the evidence indicates that input/restart-file edits are not a "
            "physically credible repair for the geometry.\n"
            "For EACH fix or tool operation, output a separate block.\n"
            "If there are multiple fixes, SEPARATE EACH BLOCK by exactly four dashes `----` on a line by themselves.\n"
            "Do NOT use any other separator between blocks except `----`.\n"
            "Return your response STRICTLY as described above.\n"
        )

        user_prompt = f"ERROR source from VASP logs:\n{error_source}\n\nERROR excerpt:\n{error_text}\n\n"
        if rag_evidence:
            user_prompt += (
                "----- Retrieved VASP error-recovery evidence -----\n"
                f"{rag_evidence}\n\n"
                "Use the evidence conservatively. Prefer the lowest-risk candidate action that matches the actual logs.\n\n"
            )
        if runtime_facts:
            user_prompt += f"----- Runtime action facts -----\n{runtime_facts}\n\n"
        for fname, content in file_dict.items():
            user_prompt += f"\n----- {fname} -----\n{content}\n"

        result = self._invoke_llm(system_prompt, user_prompt,
                                  agent="VASPErrorAgent", label="runtime_error_fix")
        try:
            log_llm_decision("VASPErrorAgent", "runtime_error_fix",
                             {"error_source": error_source[:200],
                              "rag_evidence": rag_evidence[:2000],
                              "patch": result[:2000]})
        except Exception:
            pass
        return result

    def _run_single(self, context: Dict[str, Any]) -> Dict[str, Any]:
        
        sys_info = self._get_active_system_info(context)
        if sys_info is None:
            context.setdefault("results", {})["vasp_status"] = "no_system"
            return context

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

        print(f"[VASPErrorAgent] Polling 1 system every {self.wait_interval_sec}s dir={system_dir}")
        record_job_event(
            context,
            "polling",
            message="VASP marker polling started",
            metadata={
                "poll_interval_sec": self.wait_interval_sec,
                "system_dir": str(system_dir),
            },
        )

        overall_failed = False

        while True:
            if state in ("done_ok", "giveup"):
                break

            
            if not self._is_finished(system_dir):
                record_scheduler_status(context)
                time.sleep(self.wait_interval_sec)
                continue

            flag = self._which_flag(system_dir)
            print(f"[VASPErrorAgent] {flag} detected.")
            
            if flag == "DONE":
                state = "done_ok"
                record_job_event(context, "done_ok", message="VASP DONE marker detected")
                break

            
            has_err, err_src, err_excerpt = self._detect_error(system_dir)
            if not has_err:
                print("[VASPErrorAgent] FAILED but no clear error pattern -> giveup")
                state = "giveup"
                overall_failed = True
                break

            print(f"[VASPErrorAgent] ERROR detected (source={err_src})")
            record_job_event(
                context,
                "failed",
                message=f"VASP FAILED marker detected; error source={err_src}",
                last_error=err_excerpt[:4000],
            )

            if retry >= self.MAX_RETRY:
                print("[VASPErrorAgent] MAX_RETRY exceeded -> giveup")
                state = "giveup"
                overall_failed = True
                record_job_event(context, "giveup", message="VASP maximum retry count reached")
                break

            file_dict = {
                "INCAR": self._read_file(str(system_dir / "INCAR")),
                "POSCAR": self._read_file(str(system_dir / "POSCAR")),
            }
            if (system_dir / "KPOINTS").is_file():
                file_dict["KPOINTS"] = self._read_file(str(system_dir / "KPOINTS"))
            if (system_dir / "POTCAR").is_file():
                file_dict["POTCAR(excerpt)"] = self._potcar_excerpt(str(system_dir / "POTCAR"))

            rag_hits = self._retrieve_error_knowledge_hits(err_src, err_excerpt, file_dict)
            rag_evidence = self._format_error_knowledge(rag_hits)
            if rag_evidence:
                print("\n[VASP ERROR KNOWLEDGE]\n", rag_evidence[:3000])

            runtime_facts = "\n".join(
                [
                    f"retry_index={retry}",
                    f"job_role={context.get('vasp_role') or sys_info.get('role') or 'unspecified'}",
                    f"CONTCAR_exists={(system_dir / 'CONTCAR').is_file()}",
                    f"CONTCAR_is_valid_POSCAR={self._looks_like_poscar(system_dir / 'CONTCAR')}",
                    f"CHGCAR_exists={(system_dir / 'CHGCAR').is_file()}",
                    f"WAVECAR_exists={(system_dir / 'WAVECAR').is_file()}",
                ]
            )
            patch_text = self._call_llm_for_fix(
                err_src,
                err_excerpt,
                file_dict,
                rag_evidence=rag_evidence,
                runtime_facts=runtime_facts,
            )
            print("\n[LLM PATCH SUGGESTION]\n", patch_text)

            recovery_result = self._apply_llm_recovery_response(
                system_dir,
                patch_text,
                context,
            )
            context.setdefault("results", {})["vasp_error_recovery_mode"] = "rag_llm"
            context["results"]["vasp_llm_recovery"] = recovery_result

            if recovery_result.get("structure_requested"):
                state = "giveup"
                overall_failed = True
                record_job_event(
                    context,
                    "structure_regeneration_requested",
                    message="VASP RAG-grounded LLM requested structure regeneration",
                    metadata=recovery_result,
                    last_error=err_excerpt[:4000],
                )
                break

            if len(recovery_result.get("applied", [])) == 0:
                print("[VASPErrorAgent] 0 patches/actions applied -> giveup this round.")
                retry += 1
                state = "giveup"
                overall_failed = True
                record_job_event(
                    context,
                    "giveup",
                    message="VASP recovery produced no applied edits or file actions",
                    metadata={
                        "retry": retry,
                        "recovery_result": recovery_result,
                    },
                    last_error=err_excerpt[:4000],
                )
                break

            
            submit_res = self._submit_qsub(system_dir, label)
            context["vasp_submit"] = submit_res
            context["vasp_job_id"] = submit_res.get("job_id")
            context["scheduler_job_id"] = submit_res.get("job_id")
            context["vasp_submitted"] = (submit_res.get("status") == "submitted")

            retry += 1
            get_job_manager().record_submission(
                context,
                qsub_path=submit_res.get("qsub_path", ""),
                returncode=submit_res.get("returncode") or -1,
                stdout=submit_res.get("stdout") or "",
                stderr=submit_res.get("stderr") or "",
                status="submitted" if submit_res.get("status") == "submitted" else "submit_failed",
                scheduler_job_id=submit_res.get("job_id"),
                metadata={"software": "VASP", "retry": retry, "vasp_label": submit_res.get("label")},
            )

            if submit_res.get("status") != "submitted":
                print(f"[VASPErrorAgent] resubmit failed (status={submit_res.get('status')})")
                state = "giveup"
                overall_failed = True
                record_job_event(
                    context,
                    "submit_failed",
                    message="VASP retry submission failed",
                    metadata={"retry": retry, "submit_status": submit_res.get("status")},
                )
                break

            jid = submit_res.get("job_id")
            if jid:
                print(f"[VASPErrorAgent] resubmitted job_id={jid}")
            else:
                print("[VASPErrorAgent] resubmitted (job_id unknown)")

            
            state = "pending"
            record_job_event(
                context,
                "retrying",
                message="VASP auto-patch applied; retry submitted",
                metadata={
                    "retry": retry,
                    "recovery_result": recovery_result,
                },
            )
            time.sleep(self.wait_interval_sec)

        context["vasp_retry"] = retry
        context["vasp_state"] = state
        context.setdefault("results", {})["vasp_status"] = "partial_or_failed" if overall_failed else "ok"
        record_job_event(
            context,
            "partial_or_failed" if overall_failed else "done_ok",
            message="VASP polling completed",
            metadata={"vasp_state": state, "vasp_retry": retry},
        )
        return context
        


    def pre_run_review(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if context.get("vasp_status") == "needs_structure_from_user":
            return context

        work_dir = (context.get("vasp_system") or {}).get("dir") \
                   or context.get("vasp_dir") \
                   or context.get("work_dir")
        if not work_dir:
            return context

        system_dir = Path(work_dir)
        incar_path = system_dir / "INCAR"
        if not incar_path.exists():
            return context

        mof   = context.get("mof", "unknown MOF")
        guest = context.get("guest", "unknown guest")

        file_dict = {}
        for fname in ["INCAR", "POSCAR", "KPOINTS"]:
            p = system_dir / fname
            if p.exists():
                file_dict[fname] = self._read_file(str(p))

        system_prompt = (
            "You are a VASP input reviewer for MOF simulations. This simulation uses VASP 5.4.1.\n"
            "Check the provided VASP input files for physics-level errors BEFORE running.\n"
            "Focus on: inappropriate ENCUT for the system, wrong ISMEAR/SIGMA for insulators/metals, "
            "missing IVDW for dispersion-dominated MOFs, NSW=0 when relaxation is needed, "
            "ISIF setting mismatches for the task (single-point vs relax), "
            "missing IBRION, spin polarization (ISPIN) inconsistency.\n"
            "If no issues: reply with exactly 'OK' and nothing else.\n"
            "If issues exist: output fixes using EXACTLY one of these ACTION patterns per block:\n"
            "  ACTION: Replace:\n```<old line(s)>```\nwith:\n```<new line(s)>```\n"
            "  ACTION: After the line:\n```<target line>```\nadd:\n```<lines to insert>```\n"
            "  ACTION: Before the line:\n```<target line>```\nadd:\n```<lines to insert>```\n"
            "  ACTION: Remove the line:\n```<exact line to remove>```\n"
            "  ACTION: Append at end:\n```<text to append>```\n"
            "  ACTION: Overwrite entire file with:\n```<full new content>```\n"
            "Format each fix as:\n"
            "REASON: <one sentence explaining the physics error>\n"
            "FILE: <filename>\nACTION: <one of the patterns above>\n"
            "Separate multiple fixes with exactly four dashes '----' on their own line.\n"
            "Only flag real physics errors — do not suggest style preferences.\n"
            "The ACTION line must start with one of: Replace:, After the line:, Before the line:, "
            "Remove the line:, Append at end:, Overwrite entire file with:"
        )

        user_prompt = f"MOF: {mof}\nGuest: {guest}\n\n"
        for fname, content in file_dict.items():
            user_prompt += f"\n----- {fname} -----\n{content}\n"

        print(f"\n[VASPErrorAgent] Pre-run review for {mof} ...")
        response = self._invoke_llm(system_prompt, user_prompt,
                                    agent="VASPErrorAgent", label="pre_run_review")

        if response.strip().upper() == "OK":
            print("[VASPErrorAgent] Pre-run review: no issues found.")
            try:
                log_llm_decision("VASPErrorAgent", "pre_run_review",
                                 {"result": "OK"}, context)
            except Exception:
                pass
            return context

        print("[VASPErrorAgent] Pre-run review found issues. Proposed fixes:\n")
        print(response)
        try:
            log_llm_decision("VASPErrorAgent", "pre_run_review",
                             {"result": "fix_proposed", "patch": response[:2000]}, context)
        except Exception:
            pass

        action, response = self._ask_user_confirmation(
            "VASPErrorAgent", response, system_prompt, user_prompt
        )
        if action == "skip":
            return context

        for block in response.split("----"):
            if not block.strip():
                continue
            if "FILE:" in block:
                fname_rel = block.split("FILE:")[1].split("\n")[0].strip()
                full_path = str(system_dir / fname_rel)
                self.patch_file(full_path, block)

        print("[VASPErrorAgent] Pre-run fixes applied.")
        return context

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if context.get("vasp_status") == "needs_structure_from_user":
            context["vasp_state"] = "giveup"
            context.setdefault("results", {})[
                "vasp_error_status"
            ] = "not_invoked_missing_structure"
            return context

        if "batch" in context and isinstance(context["batch"], list):
            return self.run_batch(context)
        return self._run_single(context)
