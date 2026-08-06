import json
import os
import re
import time
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from config import LLM_DEFAULT, AGENT_LLM_MAP, RASPA_DIR as _RASPA_DIR, INTERACTION_MODE
from core.job_manager import record_job_event, record_scheduler_status
from core.llm_logging import log_llm_decision
from RASPA.runner import RASPARunner

from .agent import ErrorAgent
from .structure_regeneration import request_structure_regeneration

RASPA_DIR = Path(_RASPA_DIR)


class RASPAErrorAgent(ErrorAgent):
    MIXING_RULE_KEYWORDS = {"Lorentz-Berthelot", "Jorgensen", "WaldmanHagler"}

    def __init__(self, llm=None, max_lines: int = 200, max_trials: int = 3):
        self._init_error_agent(
            llm=llm,
            default_llm=AGENT_LLM_MAP.get("RASPAErrorAgent", LLM_DEFAULT),
            max_lines=max_lines,
        )
        self.max_trials = int(max_trials)

    

    def _gather_error_text(self, context: Dict[str, Any]) -> str:
        work_dir_str = context.get("work_dir", "")
        if not work_dir_str:
            return ""

        work_dir = Path(work_dir_str)
        output_file = work_dir / "output"

        if output_file.is_file():
            return self.read_file(str(output_file))
        return ""

    def _observe_job_completion(
        self,
        context: Dict[str, Any],
        work_dir: Path,
    ) -> Dict[str, Any]:
        done_path = work_dir / "DONE"
        failed_path = work_dir / "FAILED"
        output_path = work_dir / "output"
        results = context.get("results") or {}
        submit_returncode = results.get("raspa_submit_returncode")
        try:
            submission_returned_nonzero = (
                submit_returncode is not None
                and int(submit_returncode) != 0
            )
        except (TypeError, ValueError):
            submission_returned_nonzero = False
        submission_failed = (
            context.get("raspa_status") == "submit_failed"
            or submission_returned_nonzero
        )

        output_size = None
        if output_path.is_file():
            try:
                output_size = output_path.stat().st_size
            except OSError:
                output_size = None

        result_data_files = []
        result_root = work_dir / "Output"
        if result_root.is_dir():
            for result_path in sorted(result_root.rglob("*.data"))[:20]:
                try:
                    size_bytes = result_path.stat().st_size
                except OSError:
                    size_bytes = None
                result_data_files.append(
                    {
                        "path": str(result_path),
                        "size_bytes": size_bytes,
                    }
                )

        return {
            "source": "runtime",
            "operation": "observe_raspa_job",
            "work_dir": str(work_dir),
            "result": {
                "complete": bool(
                    done_path.exists()
                    or failed_path.exists()
                    or submission_failed
                ),
                "done_marker_exists": done_path.exists(),
                "failed_marker_exists": failed_path.exists(),
                "done_marker_text": (
                    self.read_file(str(done_path)).strip()
                    if done_path.is_file()
                    else None
                ),
                "failed_marker_text": (
                    self.read_file(str(failed_path)).strip()
                    if failed_path.is_file()
                    else None
                ),
                "output_path": str(output_path),
                "output_exists": output_path.is_file(),
                "output_size_bytes": output_size,
                "result_data_files": result_data_files,
                "submission_status": context.get("raspa_status"),
                "submission_returncode": submit_returncode,
                "submission_stdout": results.get("raspa_submit_stdout", ""),
                "submission_stderr": results.get("raspa_submit_stderr", ""),
            },
        }

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

    

    def _retrieve_error_knowledge_hits(
        self,
        error_msg: str,
        file_dict: Dict[str, str],
    ):
        try:
            from rag.raspa_error_knowledge import RASPAErrorKnowledgeBase

            kb = RASPAErrorKnowledgeBase()
            del file_dict
            hits = kb.search(error_msg, top_k=5)
            formatted = []
            for hit in hits:
                item = dict(hit)
                item["match_scope"] = "error_log"
                formatted.append(item)
            return formatted
        except Exception as exc:
            print(f"[RASPAErrorAgent] RASPA error knowledge retrieval disabled: {exc}")
            return []

    def _format_error_knowledge(self, hits) -> str:
        if not hits:
            return ""
        try:
            from rag.raspa_error_knowledge import RASPAErrorKnowledgeBase

            return RASPAErrorKnowledgeBase().format_hits(hits, max_chars=4500)
        except Exception:
            return ""

    @staticmethod
    def _format_model_selection_evidence(context: Dict[str, Any]) -> str:
        hints = context.get("raspa_rag_hints")
        if not isinstance(hints, dict):
            hints = {}

        sections: List[str] = []
        for key, label in (
            ("forcefield_hints", "Framework force-field literature hints"),
            ("molecule_hints", "Guest molecule-model literature hints"),
            ("charge_hints", "Charge-model literature hints"),
        ):
            value = str(hints.get(key) or "").strip()
            if value:
                sections.append(f"{label}:\n{value}")

        user_query = str(
            context.get("user_query")
            or context.get("query_text")
            or ""
        ).strip()
        if user_query:
            sections.append(
                "USER_QUERY (selection authority only when it explicitly names a model):\n"
                f"{user_query}"
            )
        return "\n\n".join(sections)

    def _consult_rag_agent_for_models(
        self,
        context: Dict[str, Any],
        top_files: int = 10,
    ) -> Dict[str, Any]:
        existing_hints = context.get("raspa_rag_hints")
        if isinstance(existing_hints, dict) and any(
            str(existing_hints.get(key) or "").strip()
            for key in ("forcefield_hints", "molecule_hints", "charge_hints")
        ):
            result = {
                "status": "evidence_already_available",
                "mof": str(context.get("mof") or "").strip(),
                "guest": str(context.get("guest") or "").strip(),
                "property": str(context.get("property") or "").strip(),
            }
            context["raspa_status"] = "model_evidence_retrieved"
            context.setdefault("results", {}).setdefault(
                "raspa_model_rag_consultations",
                [],
            ).append(result)
            return result

        rag_context = {
            "job_name": context.get("job_name") or "",
            "mof": context.get("mof") or "",
            "guest": context.get("guest") or "",
            "property": context.get("property") or "",
            "query_text": (
                context.get("user_query")
                or context.get("query_text")
                or ""
            ),
        }
        consultation: Dict[str, Any] = {
            "status": "failed",
            "mof": str(rag_context["mof"]).strip(),
            "guest": str(rag_context["guest"]).strip(),
            "property": str(rag_context["property"]).strip(),
        }

        try:
            from rag.agent import RagAgent

            rag_result = RagAgent(agent_name="RagAgent").run_for_raspa_models(
                rag_context,
                top_files=top_files,
            )
            hints = {
                "forcefield_hints": str(
                    rag_result.get("forcefield_hints") or ""
                ).strip(),
                "molecule_hints": str(
                    rag_result.get("molecule_hints") or ""
                ).strip(),
                "charge_hints": str(
                    rag_result.get("charge_hints") or ""
                ).strip(),
            }
            context["raspa_rag_hints"] = hints
            consultation.update(
                {
                    "status": (
                        "evidence_retrieved"
                        if any(hints.values())
                        else "no_relevant_evidence"
                    ),
                    "hints": hints,
                    "queries": [
                        {
                            "intent": str(
                                query.get("intent", "")
                                if isinstance(query, dict)
                                else getattr(query, "intent", "")
                            ),
                            "query": str(
                                query.get("query", query)
                                if isinstance(query, dict)
                                else getattr(query, "query", query)
                            ),
                        }
                        for query in (rag_result.get("raspa_model_queries") or [])
                    ],
                    "source_filenames": [
                        str(item.get("filename") or "")
                        for item in (rag_result.get("top_file_hits") or [])
                        if isinstance(item, dict) and item.get("filename")
                    ],
                }
            )
        except Exception as exc:
            consultation["error"] = str(exc)

        context.setdefault("results", {}).setdefault(
            "raspa_model_rag_consultations",
            [],
        ).append(consultation)

        if consultation["status"] == "evidence_retrieved":
            context["raspa_status"] = "model_evidence_retrieved"
        else:
            self._request_user_model_selection(
                context,
                "RagAgent could not retrieve literature evidence supporting a "
                "compatible framework force field or guest molecule model",
                consultation.get("error")
                or "No relevant literature model-selection evidence was retrieved.",
            )
        return consultation

    @staticmethod
    def _model_selection_fields_in_patch(block: str) -> List[str]:
        fields = set()
        for payload in re.findall(r"```([\s\S]+?)```", block or ""):
            for line in payload.splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                key = parts[0].lower()
                if key == "forcefield":
                    fields.add("Forcefield")
                elif key == "moleculedefinition":
                    fields.add("MoleculeDefinition")
        return sorted(fields)

    def _review_model_change_with_llm(
        self,
        context: Dict[str, Any],
        input_path: Path,
        block: str,
        fields: List[str],
    ) -> Dict[str, Any]:
        evidence = self._format_model_selection_evidence(context)
        if not evidence:
            return {
                "supported": False,
                "reason": "no literature or explicit-user model-selection evidence is available",
                "citations": [],
                "fields": fields,
            }

        system_prompt = (
            "You are an independent scientific-evidence reviewer for a proposed RASPA model change.\n"
            "Judge only whether the proposed Forcefield and/or MoleculeDefinition change is supported "
            "by the supplied literature hints or an explicit user model request.\n"
            "Installed-file availability proves only technical availability, not scientific compatibility.\n"
            "Do not transfer guest molecule-model evidence to a framework Forcefield choice, or framework "
            "force-field evidence to a MoleculeDefinition choice.\n"
            "A generic statement that a model is common is insufficient when the evidence does not support "
            "its use for the stated MOF, guest, or closely matching simulation context.\n"
            "Return JSON only with this schema:\n"
            '{"supported": true|false, "reason": "...", "citations": ["exact source filename", ...]}\n'
            "Use USER_QUERY as a citation only when the user explicitly requested the proposed model."
        )
        user_prompt = (
            f"MOF: {context.get('mof') or ''}\n"
            f"Guest: {context.get('guest') or ''}\n"
            f"Property: {context.get('property') or ''}\n"
            f"Fields changed: {', '.join(fields)}\n\n"
            f"Current simulation.input:\n{self.read_file(str(input_path))}\n\n"
            f"Proposed patch:\n{block}\n\n"
            f"Available model-selection evidence:\n{evidence}"
        )

        try:
            raw = self._invoke_llm(
                system_prompt,
                user_prompt,
                agent="RASPAErrorAgent",
                label="runtime_model_change_review",
            )
            text = str(raw or "").strip()
            if text.startswith("```"):
                text = "\n".join(text.splitlines()[1:-1]).strip()
            obj = json.loads(text)
            citations = [
                str(item).strip()
                for item in (obj.get("citations") or [])
                if str(item).strip()
            ]
            evidence_lower = evidence.lower()
            verified_citations = [
                item
                for item in citations
                if item.lower() in evidence_lower
            ]
            supported = bool(obj.get("supported")) and bool(verified_citations)
            review = {
                "supported": supported,
                "reason": str(obj.get("reason") or "").strip(),
                "citations": verified_citations,
                "fields": fields,
                "raw_response": raw,
            }
        except Exception as exc:
            review = {
                "supported": False,
                "reason": f"model-change evidence review failed: {exc}",
                "citations": [],
                "fields": fields,
            }

        try:
            log_llm_decision(
                "RASPAErrorAgent",
                "runtime_model_change_review",
                review,
                context,
            )
        except Exception:
            pass
        return review

    def call_llm_for_fix(
        self,
        error_msg: str,
        file_dict: Dict[str, str],
        rag_evidence: str = "",
        local_reference_evidence: str = "",
        model_selection_evidence: str = "",
        runtime_facts: str = "",
    ) -> str:
        system_prompt = (
            "You are a RASPA Monte Carlo simulation troubleshooting assistant.\n"
            "You will be given the output and direct runtime observations from a completed "
            "or failed RASPA job, together with relevant input files.\n\n"
            "First decide whether the result is trustworthy. If it is not, identify the "
            "failure from the complete evidence and choose the minimal evidence-grounded "
            "recovery. No downstream keyword-to-action policy will infer the cause or "
            "recovery for you.\n"
            "Rules for your response:\n"
            "- Always provide the smallest number of changes necessary to resolve the observed issue.\n"
            "- Never suggest contradictory changes.\n"
            "- Do not propose cosmetic changes unless required.\n"
            "- Use the retrieved RASPA error-recovery evidence when it matches the log.\n"
            "- Treat RASPA source-code evidence as error-string evidence, and RASPA manual evidence as recovery-protocol guidance.\n"
            "- Never invent force-field parameters, masses, charges, atom types, or source priorities.\n"
            "- Change a thermodynamic or simulation-control parameter only when the replacement value is explicitly "
            "present in the user/workflow request or is directly supported by supplied evidence. If the intended "
            "physical value is unavailable, do not invent a plausible default.\n"
            "- Installed force-field and molecule files prove technical availability only, not scientific compatibility.\n"
            "- Preserve the current physical model when its missing files can be restored.\n"
            "- If resolving the failure requires selecting or replacing Forcefield or MoleculeDefinition and no "
            "literature model-selection evidence is supplied, first use CONSULT_RAG_AGENT_FOR_MODEL. The RagAgent, "
            "not this ErrorAgent, is responsible for retrieving candidate-model literature evidence.\n"
            "- Change Forcefield or MoleculeDefinition only when the supplied literature model-selection evidence "
            "explicitly supports the proposed model for the stated MOF, guest, or closely matching context. "
            "Do not use guest-model evidence to justify a framework Forcefield.\n"
            "- If no supported replacement model is available, use REQUEST_USER_MODEL_SELECTION instead of choosing "
            "an installed candidate arbitrarily.\n"
            "- If Runtime recovery facts include an independent model-review rejection, revise the proposal using "
            "the cited literature-supported candidates. Do not repeat the rejected candidate.\n"
            "- For missing VDW parameters, use only exact lines retrieved from installed local force-field files. "
            "Choose and report the source force field and source atom type for every target type.\n"
            "- Never use TEXT_PATCH for a CIF file. Cell lengths, angles, symmetry, space-group data, "
            "and atom coordinates are scientific structure data and must not be guessed or inferred from "
            "neighboring values.\n"
            "- For a missing, unreadable, or malformed CIF, use REFETCH_CIF whenever a MOF or FrameworkName is "
            "available. The structure agent, rather than this troubleshooting LLM, must attempt retrieval and "
            "validation before the workflow asks the user for a CIF. Use REQUEST_USER_CIF only after runtime facts "
            "show that an automatic structure-recovery attempt failed. Use CONVERT_CIF_TO_P1 only when the existing "
            "cell and coordinates are valid and the evidence specifically supports a symmetry-representation conversion.\n"
            "- Every response block must contain RATIONALE and EVIDENCE. Cite retrieved evidence by item/source "
            "when available, and always cite the matching runtime, input, or local-file fact.\n"
            "\n"
            "For a text edit, use this strict format:\n"
            "RATIONALE: <why the edit addresses this failure>\n"
            "EVIDENCE: <retrieved source/item and matching runtime fact>\n"
            "DECISION: TEXT_PATCH\n"
            "FILE: <filename>\n"
            "Use ONLY ONE of these action patterns for each fix:\n"
            "1. ACTION: After the line:\n```<text>```\nadd:\n```<text to insert>```\n"
            "2. ACTION: Before the line:\n```<text>```\nadd:\n```<text to insert>```\n"
            "3. ACTION: Remove the line:\n```<exact line to remove>```\n"
            "4. ACTION: Replace:\n```<old line(s)>```\nwith:\n```<new line(s)>```\n"
            "5. ACTION: Append at end:\n```<text to append>```\n"
            "6. ACTION: Overwrite entire file with:\n```<new content>```\n"
            "\nFor a controlled non-text operation, use one of these strict formats:\n"
            "RATIONALE: <reason>\nEVIDENCE: <retrieved source/item and matching runtime fact>\n"
            "DECISION: TOOL_ACTION\nTOOL ACTION: REFETCH_CIF\n"
            "or\n"
            "RATIONALE: <reason>\nEVIDENCE: <retrieved source/item and matching runtime fact>\n"
            "DECISION: TOOL_ACTION\nTOOL ACTION: CONVERT_CIF_TO_P1\n"
            "or\n"
            "RATIONALE: <reason>\nEVIDENCE: <retrieved source/item and matching runtime fact>\n"
            "DECISION: TOOL_ACTION\nTOOL ACTION: REQUEST_USER_CIF\n"
            "or\n"
            "RATIONALE: <why model-selection literature evidence is required>\n"
            "EVIDENCE: <runtime/input facts showing that Forcefield or MoleculeDefinition selection is implicated>\n"
            "DECISION: TOOL_ACTION\nTOOL ACTION: CONSULT_RAG_AGENT_FOR_MODEL\n"
            "or\n"
            "RATIONALE: <reason>\nEVIDENCE: <why literature/user evidence is insufficient for a model replacement>\n"
            "DECISION: TOOL_ACTION\nTOOL ACTION: REQUEST_USER_MODEL_SELECTION\n"
            "or\n"
            "RATIONALE: <reason>\nEVIDENCE: <local force-field paths/lines for every mapping>\n"
            "DECISION: TOOL_ACTION\nTOOL ACTION: AUTOFILL_MISSING_VDW\n"
            "TYPE MAPPINGS: <target=source_forcefield:source_type; target=source_forcefield:source_type>\n"
            "\nIf no modification is justified, use:\n"
            "RATIONALE: <why the completed result is trustworthy without modification>\n"
            "EVIDENCE: <retrieved source/item and exact runtime value/fact>\n"
            "DECISION: NO_CHANGE\n"
            "For EACH fix or tool operation, output a separate block.\n"
            "If there are multiple fixes, SEPARATE EACH BLOCK by exactly four dashes `----` on a line by themselves.\n"
            "Do NOT use any other separator between blocks except `----`.\n"
            "Return your response STRICTLY as described above."
        )

        user_prompt = f"RASPA runtime output (stdout/stderr):\n{error_msg}\n\n"
        if rag_evidence:
            user_prompt += "Retrieved RASPA error-recovery evidence:\n"
            user_prompt += rag_evidence
            user_prompt += "\n\n"
        if local_reference_evidence:
            user_prompt += "Retrieved installed RASPA force-field/molecule evidence:\n"
            user_prompt += local_reference_evidence
            user_prompt += "\n\n"
        if model_selection_evidence:
            user_prompt += "Retrieved literature model-selection evidence:\n"
            user_prompt += model_selection_evidence
            user_prompt += "\n\n"
        if runtime_facts:
            user_prompt += "Runtime recovery facts:\n"
            user_prompt += runtime_facts
            user_prompt += "\n\n"
        for fname, content in file_dict.items():
            user_prompt += f"\n----- {fname} -----\n{content}\n"

        result = self._invoke_llm(system_prompt, user_prompt,
                                  agent="RASPAErrorAgent", label="runtime_error_fix")
        try:
            log_llm_decision("RASPAErrorAgent", "runtime_error_fix",
                                 {
                                     "error_preview": error_msg[:300],
                                     "rag_evidence": rag_evidence[:2000],
                                     "local_reference_evidence": local_reference_evidence[:4000],
                                     "model_selection_evidence": model_selection_evidence[:4000],
                                     "patch": result[:2000],
                                 })
        except Exception:
            pass
        return result

    def pre_run_review(self, context: Dict[str, Any]) -> Dict[str, Any]:
        work_dir_str = context.get("work_dir")
        input_file_str = context.get("input_file")
        if not work_dir_str or not input_file_str:
            return context

        input_path = str(Path(input_file_str))
        if not os.path.exists(input_path):
            return context

        mof   = context.get("mof", "unknown MOF")
        guest = context.get("guest", "unknown guest")

        file_dict = {"simulation.input": self.read_file(input_path)}
        model_hints = context.get("raspa_rag_hints") or {}
        manual_hints = context.get("raspa_manual_hints") or ""

        system_prompt = (
            "You are a RASPA2 Monte Carlo simulation input reviewer for MOF simulations.\n"
            "Check the provided simulation.input before execution. Base every finding on "
            "the supplied file and retrieved evidence; do not use a memorized "
            "keyword-to-fix mapping. Preserve the requested physical model and make only "
            "the minimum correction supported by evidence. Do not invent parameters or "
            "add unsupported keywords. Installed-file availability is not scientific "
            "compatibility: do not change Forcefield or MoleculeDefinition unless the "
            "retrieved model-selection evidence explicitly supports that model for the "
            "stated MOF/guest context.\n"
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
        if model_hints:
            user_prompt += (
                "Retrieved model-selection evidence:\n"
                f"{json.dumps(model_hints, indent=2, ensure_ascii=False)}\n\n"
            )
        if manual_hints:
            user_prompt += f"Retrieved RASPA manual evidence:\n{manual_hints}\n\n"
        for fname, content in file_dict.items():
            user_prompt += f"\n----- {fname} -----\n{content}\n"

        print(f"\n[RASPAErrorAgent] Pre-run review for {mof} / {guest} ...")
        response = self._invoke_llm(system_prompt, user_prompt,
                                    agent="RASPAErrorAgent", label="pre_run_review")

        if response.strip().upper() == "OK":
            print("[RASPAErrorAgent] Pre-run review: no issues found.")
            try:
                log_llm_decision("RASPAErrorAgent", "pre_run_review",
                                 {"result": "OK"}, context)
            except Exception:
                pass
            return context

        print("[RASPAErrorAgent] Pre-run review found issues. Proposed fixes:\n")
        print(response)
        try:
            log_llm_decision("RASPAErrorAgent", "pre_run_review",
                             {"result": "fix_proposed", "patch": response[:2000]}, context)
        except Exception:
            pass

        action, response = self._ask_user_confirmation(
            "RASPAErrorAgent", response, system_prompt, user_prompt
        )
        if action == "skip":
            return context

        work_dir = Path(work_dir_str)
        for block in re.split(r"(?m)^\s*----\s*$", response):
            if not block.strip():
                continue
            if "FILE:" in block:
                fname_rel = block.split("FILE:")[1].split("\n")[0].strip()
                full_path = str(work_dir / fname_rel)
                self.patch_file(full_path, block)

        print("[RASPAErrorAgent] Pre-run fixes applied.")
        return context

    def _find_raspa_cif_path(self, context: Dict[str, Any], input_path: Path) -> Optional[Path]:
        candidates = []
        if context.get("mof_path"):
            candidates.append(Path(context["mof_path"]))

        fw_name = self._find_framework_name_from_input(input_path)
        if fw_name:
            work_dir = Path(context.get("work_dir") or input_path.parent)
            candidates.append(work_dir / f"{fw_name}.cif")
            candidates.append(RASPA_DIR / "share" / "raspa" / "structures" / "cif" / f"{fw_name}.cif")

        mof = context.get("mof")
        if mof:
            work_dir = Path(context.get("work_dir") or input_path.parent)
            candidates.append(work_dir / f"{mof}.cif")
            candidates.append(RASPA_DIR / "share" / "raspa" / "structures" / "cif" / f"{mof}.cif")

        for path in candidates:
            if path and path.is_file():
                return path
        return None

    def _cif_has_required_cell_info(self, cif_path: Path) -> bool:
        try:
            text = cif_path.read_text(errors="ignore")
        except Exception:
            return False

        required = [
            "_cell_length_a",
            "_cell_length_b",
            "_cell_length_c",
            "_cell_angle_alpha",
            "_cell_angle_beta",
            "_cell_angle_gamma",
        ]
        for key in required:
            m = re.search(rf"^\s*{re.escape(key)}\s+(\S+)", text, flags=re.IGNORECASE | re.MULTILINE)
            if not m:
                return False
            try:
                value = float(str(m.group(1)).strip("'\"").split("(")[0])
                if not (value > 0):
                    return False
            except Exception:
                return False
        return True

    def _copy_cif_to_raspa_structure_dir(self, context: Dict[str, Any], cif_path: Path) -> Optional[str]:
        fw_name = context.get("mof") or cif_path.stem
        target = RASPA_DIR / "share" / "raspa" / "structures" / "cif" / f"{fw_name}.cif"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if cif_path.resolve() != target.resolve():
                shutil.copy2(cif_path, target)
            return str(target)
        except Exception as exc:
            print(f"[RASPAErrorAgent] Warning: could not copy CIF to RASPA structure dir: {exc}")
            return None

    def _request_user_structure(self, context: Dict[str, Any], reason: str) -> Dict[str, Any]:
        message = (
            "RASPA could not recover a valid CIF automatically. "
            "Please provide a corrected CIF path for this MOF and rerun the job."
        )
        context["raspa_state"] = "giveup"
        context["raspa_status"] = "needs_structure_from_user"
        context["raspa_structure_request"] = {
            "status": "needs_user_cif",
            "reason": reason,
            "message": message,
            "mof": context.get("mof"),
            "mode": INTERACTION_MODE,
        }
        context.setdefault("results", {})["raspa_structure_request"] = context["raspa_structure_request"]
        request_structure_regeneration(
            context,
            software="raspa",
            reason=reason,
            action="needs_user_cif",
            status="blocked",
            metadata=context["raspa_structure_request"],
        )
        print(f"[RASPAErrorAgent] {message} Reason: {reason}")
        return context

    def _request_user_model_selection(
        self,
        context: Dict[str, Any],
        reason: str,
        evidence: str,
    ) -> Dict[str, Any]:
        message = (
            "RASPA cannot select a replacement force field or guest molecule model "
            "without sufficient scientific evidence. Please provide or confirm the "
            "intended model before rerunning the job."
        )
        request = {
            "status": "needs_user_model_selection",
            "reason": reason,
            "evidence": evidence,
            "message": message,
            "mof": context.get("mof"),
            "guest": context.get("guest"),
            "property": context.get("property"),
            "mode": INTERACTION_MODE,
        }
        context["raspa_state"] = "giveup"
        context["raspa_status"] = "needs_model_selection_from_user"
        context["raspa_model_selection_request"] = request
        context.setdefault("results", {})["raspa_model_selection_request"] = request
        print(f"[RASPAErrorAgent] {message} Reason: {reason}")
        return context

    def _refetch_structure_for_missing_cif_info(
        self,
        context: Dict[str, Any],
        input_path: Path,
    ) -> Dict[str, Any]:
        context["raspa_structure_recovery_attempted"] = True
        context.setdefault("results", {})[
            "raspa_structure_recovery_attempted"
        ] = True
        try:
            from structure.agent import RASPAStructureAgent, validate_mof
        except Exception as exc:
            return self._request_user_structure(
                context,
                f"automatic CIF recovery is unavailable: {exc}",
            )

        work_dir = Path(context["work_dir"])
        mof = context.get("mof") or self._find_framework_name_from_input(input_path)
        if not mof:
            return self._request_user_structure(context, "cannot identify MOF/framework name for CIF refetch")

        old_cif = self._find_raspa_cif_path(context, input_path)
        if old_cif and old_cif.exists():
            backup = old_cif.with_name(f"{old_cif.name}.simmof_before_refetch")
            try:
                shutil.copy2(old_cif, backup)
            except Exception:
                backup = None
        else:
            backup = None

        struct_agent = RASPAStructureAgent()
        try:
            struct_agent._run_fetch_subprocess(mof, str(work_dir))
            fetched = Path(work_dir) / f"{mof}.cif"
            struct_agent._after_fetch(fetched, work_dir)
            validate_mof(fetched, work_dir)
        except Exception as exc:
            return self._request_user_structure(context, f"automatic CIF refetch failed: {exc}")

        if not self._cif_has_required_cell_info(fetched):
            return self._request_user_structure(context, "refetched CIF still lacks valid cell length/angle fields")

        context["mof_path"] = str(fetched)
        target = self._copy_cif_to_raspa_structure_dir(context, fetched)
        context.setdefault("results", {})["raspa_structure_recovery"] = {
            "status": "patched",
            "action": "refetch_cif",
            "mof": mof,
            "cif": str(fetched),
            "backup": str(backup) if backup else None,
            "raspa_structure_cif": target,
        }
        context["raspa_status"] = "patched"
        context["raspa_fixed_once"] = True
        print(f"[RASPAErrorAgent] Refetched CIF for missing cell information: {fetched}")
        return context

    def _convert_structure_to_p1(
        self,
        context: Dict[str, Any],
        input_path: Path,
    ) -> Dict[str, Any]:
        from structure.agent import RASPAStructureAgent

        cif_path = self._find_raspa_cif_path(context, input_path)
        if not cif_path:
            return self._request_user_structure(context, "cannot locate CIF for P1 conversion")

        struct_agent = RASPAStructureAgent()
        try:
            result = struct_agent.convert_cif_to_p1(cif_path, out_path=cif_path, backup=True)
        except Exception as exc:
            return self._request_user_structure(context, f"P1 conversion failed: {exc}")

        context["mof_path"] = str(cif_path)
        target = self._copy_cif_to_raspa_structure_dir(context, cif_path)
        result["raspa_structure_cif"] = target
        context.setdefault("results", {})["raspa_structure_recovery"] = {
            "status": "patched",
            "action": "convert_cif_to_p1",
            **result,
        }
        context["raspa_status"] = "patched"
        context["raspa_fixed_once"] = True
        print(f"[RASPAErrorAgent] Converted CIF to P1 for RASPA retry: {cif_path}")
        return context

    def _selected_forcefield_from_input(self, input_path: Path) -> Optional[str]:
        try:
            for line in input_path.read_text(errors="ignore").splitlines():
                parts = line.split()
                if parts and parts[0].lower() == "forcefield" and len(parts) >= 2:
                    return parts[1].strip()
        except Exception:
            return None
        return None

    def _parse_missing_vdw_pairs(self, text: str) -> List[Tuple[str, str]]:
        from rag.raspa_forcefield_reference import parse_missing_vdw_pairs

        return parse_missing_vdw_pairs(text)

    def _parse_forcefield_file_entries(self, path: Path) -> Tuple[List[str], Dict[str, str], List[str]]:
        if not path.exists():
            return [], {}, []
        lines = path.read_text(errors="ignore").splitlines()
        count_idx = None
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                int(stripped.split()[0])
                count_idx = idx
                break
            except Exception:
                continue
        if count_idx is None:
            return lines, {}, []

        header = lines[:count_idx]
        entries: Dict[str, str] = {}
        footer: List[str] = []
        for line in lines[count_idx + 1:]:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            first = stripped.split()[0]
            if first in self.MIXING_RULE_KEYWORDS:
                footer.append(line)
                continue
            entries[first] = line
        return header, entries, footer

    def _write_forcefield_file_entries(
        self,
        path: Path,
        header: List[str],
        entries: Dict[str, str],
        footer: List[str],
    ) -> None:
        lines = list(header)
        lines.append(str(len(entries)))
        lines.append("# type interaction")
        lines.extend(entries.values())
        if footer:
            lines.append("# general mixing rule for Lennard-Jones")
            lines.extend(footer)
        path.write_text("\n".join(lines).rstrip() + "\n")

    def _parse_pseudo_atom_entries(self, path: Path) -> Tuple[List[str], Dict[str, str]]:
        if not path.exists():
            return [], {}
        lines = path.read_text(errors="ignore").splitlines()
        count_idx = None
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                int(stripped.split()[0])
                count_idx = idx
                break
            except Exception:
                continue
        if count_idx is None:
            return lines, {}

        header = lines[:count_idx]
        entries: Dict[str, str] = {}
        for line in lines[count_idx + 1:]:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if parts:
                entries[parts[0]] = line
        return header, entries

    def _write_pseudo_atom_entries(self, path: Path, header: List[str], entries: Dict[str, str]) -> None:
        lines = list(header)
        lines.append(str(len(entries)))
        lines.append(
            "#type      print   as  scatt oxidation\tmass       charge  polarization\t"
            "B-factor radii  connectivity\tanisotropic\tanisotropic-type\ttinker-type"
        )
        lines.extend(entries.values())
        path.write_text("\n".join(lines).rstrip() + "\n")

    def _replace_forcefield_in_input(self, input_path: Path, new_forcefield: str) -> None:
        text = input_path.read_text(errors="ignore")
        patched = re.sub(
            r"(?m)^(\s*Forcefield\s+)\S+(\s*)$",
            rf"\g<1>{new_forcefield}\2",
            text,
            count=1,
        )
        if patched == text:
            patched = text.rstrip() + f"\nForcefield                    {new_forcefield}\n"
        input_path.write_text(patched)

    def _replace_input_keyword_value(self, input_path: Path, keyword: str, new_value: str) -> bool:
        if not input_path.exists():
            return False
        text = input_path.read_text(errors="ignore")
        pattern = rf"(?m)^(\s*{re.escape(keyword)}\s+)\S+(\s*)$"
        patched, n = re.subn(pattern, rf"\g<1>{new_value}\2", text, count=1)
        if n == 0:
            return False
        input_path.write_text(patched)
        return True

    def _molecule_name_from_input(self, input_path: Path) -> Optional[str]:
        try:
            for line in input_path.read_text(errors="ignore").splitlines():
                parts = line.split()
                lowered = [part.lower() for part in parts]
                if "moleculename" in lowered:
                    idx = lowered.index("moleculename")
                    if idx + 1 < len(parts):
                        return parts[idx + 1].strip()
        except Exception:
            return None
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

    @staticmethod
    def _no_change_is_process_safe(context: Dict[str, Any]) -> bool:
        observations = (
            (context.get("results") or {}).get("raspa_runtime_observations")
            or []
        )
        if not observations:
            return False
        result = observations[-1].get("result") or {}
        return bool(
            result.get("done_marker_exists")
            and not result.get("failed_marker_exists")
            and result.get("output_exists")
            and result.get("submission_status") != "submit_failed"
        )

    @staticmethod
    def _parse_llm_type_mappings(value: Optional[str]) -> Dict[str, Tuple[str, str]]:
        mappings: Dict[str, Tuple[str, str]] = {}
        for raw in (value or "").split(";"):
            item = raw.strip()
            if not item or "=" not in item or ":" not in item:
                continue
            target, source = item.split("=", 1)
            source_forcefield, source_type = source.split(":", 1)
            target = target.strip()
            source_forcefield = source_forcefield.strip()
            source_type = source_type.strip()
            safe_token = r"[A-Za-z0-9_.+\-]+"
            if not all(
                re.fullmatch(safe_token, token)
                for token in (target, source_forcefield, source_type)
            ):
                continue
            mappings[target] = (source_forcefield, source_type)
        return mappings

    def _autofill_missing_vdw_from_llm(
        self,
        context: Dict[str, Any],
        input_path: Path,
        error_text: str,
        mapping_text: Optional[str],
    ) -> Dict[str, Any]:
        mappings = self._parse_llm_type_mappings(mapping_text)
        pairs = self._parse_missing_vdw_pairs(error_text)
        reported_types = {atom_type for pair in pairs for atom_type in pair}
        selected_ff = self._selected_forcefield_from_input(input_path) or context.get("forcefield")
        result: Dict[str, Any] = {
            "status": "skipped",
            "action": "llm_selected_file_backed_missing_vdw_autofill",
            "original_forcefield": selected_ff,
            "pairs": pairs,
            "requested_mappings": mappings,
        }
        if not selected_ff or not pairs or not mappings:
            result["reason"] = "missing selected force field, reported pairs, or LLM mappings"
            return result

        ff_root = RASPA_DIR / "share" / "raspa" / "forcefield"
        selected_dir = ff_root / str(selected_ff)
        if not selected_dir.is_dir():
            result["reason"] = "selected force-field directory does not exist"
            return result

        mr_header, mr_entries, mr_footer = self._parse_forcefield_file_entries(
            selected_dir / "force_field_mixing_rules.def"
        )
        pa_header, pa_entries = self._parse_pseudo_atom_entries(selected_dir / "pseudo_atoms.def")
        missing_reported = {atom_type for atom_type in reported_types if atom_type not in mr_entries}
        if set(mappings) != missing_reported:
            result["reason"] = "LLM mappings must cover exactly all reported atom types missing from the selected force field"
            result["missing_reported_types"] = sorted(missing_reported)
            return result

        validated: Dict[str, Dict[str, str]] = {}
        for target_type, (source_ff, source_type) in mappings.items():
            source_dir = ff_root / source_ff
            _, source_rules, _ = self._parse_forcefield_file_entries(
                source_dir / "force_field_mixing_rules.def"
            )
            _, source_atoms = self._parse_pseudo_atom_entries(source_dir / "pseudo_atoms.def")
            source_rule = source_rules.get(source_type)
            source_atom = source_atoms.get(source_type)
            if not source_rule:
                result["reason"] = f"LLM-selected mixing entry not found: {source_ff}:{source_type}"
                return result
            if target_type not in pa_entries and not source_atom:
                result["reason"] = f"LLM-selected pseudo-atom entry not found: {source_ff}:{source_type}"
                return result
            validated[target_type] = {
                "source_forcefield": source_ff,
                "source_type": source_type,
                "mixing_rule": source_rule,
                "pseudo_atom": source_atom or "",
            }

        new_ff = (
            str(selected_ff)
            if str(selected_ff).endswith("_llm_autofill")
            else f"{selected_ff}_llm_autofill"
        )
        destination_dir = ff_root / new_ff
        if destination_dir != selected_dir:
            shutil.copytree(selected_dir, destination_dir, dirs_exist_ok=True)

        added_rules = []
        added_pseudo_atoms = []
        for target_type, source in validated.items():
            mr_entries[target_type] = re.sub(
                r"^\S+",
                target_type,
                source["mixing_rule"],
                count=1,
            )
            added_rules.append({
                "target_type": target_type,
                "source_forcefield": source["source_forcefield"],
                "source_type": source["source_type"],
            })
            if target_type not in pa_entries and source["pseudo_atom"]:
                pa_entries[target_type] = re.sub(
                    r"^\S+",
                    target_type,
                    source["pseudo_atom"],
                    count=1,
                )
                added_pseudo_atoms.append({
                    "target_type": target_type,
                    "source_forcefield": source["source_forcefield"],
                    "source_type": source["source_type"],
                })

        self._write_forcefield_file_entries(
            destination_dir / "force_field_mixing_rules.def",
            mr_header,
            mr_entries,
            mr_footer,
        )
        self._write_pseudo_atom_entries(
            destination_dir / "pseudo_atoms.def",
            pa_header,
            pa_entries,
        )
        self._replace_forcefield_in_input(input_path, new_ff)

        context["forcefield"] = new_ff
        context["raspa_status"] = "patched"
        context["raspa_fixed_once"] = True
        result.update({
            "status": "patched",
            "new_forcefield": new_ff,
            "added_mixing_rules": added_rules,
            "added_pseudo_atoms": added_pseudo_atoms,
        })
        context.setdefault("results", {})["raspa_forcefield_autofill"] = result
        return result

    def _apply_llm_recovery_response(
        self,
        context: Dict[str, Any],
        input_path: Path,
        error_text: str,
        response: str,
    ) -> Dict[str, Any]:
        work_dir = Path(context["work_dir"])
        applied: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []
        no_change_reports: List[Dict[str, Any]] = []
        saw_non_no_change = False

        for raw in re.split(r"(?m)^\s*----\s*$", response):
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

            if decision == "NO_CHANGE":
                if self._no_change_is_process_safe(context):
                    no_change_reports.append(report)
                else:
                    saw_non_no_change = True
                    skipped.append({
                        **report,
                        "raw": block,
                        "reason": "no_change_not_allowed_by_process_state",
                    })
                continue

            saw_non_no_change = True
            if decision == "TEXT_PATCH":
                fname = self._safe_root_filename(self._response_field(block, "FILE"))
                if not fname:
                    skipped.append({**report, "raw": block, "reason": "unsafe_or_missing_file"})
                    continue
                target = input_path if fname == input_path.name else work_dir / fname
                if target.suffix.lower() == ".cif":
                    reason = (
                        "Direct LLM text edits to CIF structure data are forbidden; "
                        "use REFETCH_CIF or provide a validated replacement CIF."
                    )
                    self._request_user_structure(context, reason)
                    skipped.append({
                        **report,
                        "file": fname,
                        "raw": block,
                        "reason": "direct_cif_text_patch_forbidden",
                    })
                    continue
                model_fields = (
                    self._model_selection_fields_in_patch(block)
                    if target == input_path
                    else []
                )
                model_review = None
                if model_fields:
                    if not self._format_model_selection_evidence(context):
                        consultation = self._consult_rag_agent_for_models(
                            context
                        )
                        skipped.append({
                            **report,
                            "file": fname,
                            "raw": block,
                            "reason": "model_patch_deferred_pending_rag_agent",
                            "rag_consultation": consultation,
                        })
                        continue
                    model_review = self._review_model_change_with_llm(
                        context,
                        input_path,
                        block,
                        model_fields,
                    )
                    context.setdefault("results", {}).setdefault(
                        "raspa_model_change_reviews",
                        [],
                    ).append(model_review)
                    if not model_review.get("supported"):
                        revision_attempts = int(
                            context.get("raspa_model_revision_attempts", 0)
                        )
                        if (
                            self._format_model_selection_evidence(context)
                            and revision_attempts < 1
                        ):
                            context["raspa_model_revision_attempts"] = (
                                revision_attempts + 1
                            )
                            context["raspa_model_revision_feedback"] = {
                                "rejected_patch": block,
                                "review": model_review,
                            }
                            context["raspa_status"] = (
                                "model_change_revision_needed"
                            )
                        else:
                            self._request_user_model_selection(
                                context,
                                model_review.get("reason")
                                or "the proposed model change lacks verified compatibility evidence",
                                evidence,
                            )
                        skipped.append({
                            **report,
                            "file": fname,
                            "raw": block,
                            "reason": "unsupported_model_change",
                            "model_review": model_review,
                        })
                        continue
                before = target.read_text(errors="ignore") if target.exists() else None
                self.patch_file(str(target), block)
                after = target.read_text(errors="ignore") if target.exists() else None
                if before != after:
                    applied.append({
                        **report,
                        "decision": decision,
                        "file": fname,
                        "raw": block,
                        "model_review": model_review,
                    })
                else:
                    skipped.append({**report, "file": fname, "raw": block, "reason": "no_change"})
                continue

            if decision != "TOOL_ACTION":
                skipped.append({**report, "raw": block, "reason": "unsupported_decision"})
                continue

            tool_action = (self._response_field(block, "TOOL ACTION") or "").upper()
            before_status = context.get("raspa_status")
            if tool_action == "REFETCH_CIF":
                self._refetch_structure_for_missing_cif_info(context, input_path)
            elif tool_action == "CONVERT_CIF_TO_P1":
                self._convert_structure_to_p1(context, input_path)
            elif tool_action == "REQUEST_USER_CIF":
                if not context.get("raspa_structure_recovery_attempted"):
                    self._refetch_structure_for_missing_cif_info(
                        context,
                        input_path,
                    )
                else:
                    self._request_user_structure(context, rationale)
            elif tool_action == "CONSULT_RAG_AGENT_FOR_MODEL":
                consultation = self._consult_rag_agent_for_models(context)
                if consultation.get("status") not in {
                    "evidence_retrieved",
                    "evidence_already_available",
                }:
                    skipped.append({
                        **report,
                        "raw": block,
                        "reason": "rag_agent_found_no_model_evidence",
                        "tool_result": consultation,
                    })
                    continue
            elif tool_action == "REQUEST_USER_MODEL_SELECTION":
                self._request_user_model_selection(
                    context,
                    rationale,
                    evidence,
                )
            elif tool_action == "AUTOFILL_MISSING_VDW":
                autofill = self._autofill_missing_vdw_from_llm(
                    context,
                    input_path,
                    error_text,
                    self._response_field(block, "TYPE MAPPINGS"),
                )
                if autofill.get("status") != "patched":
                    skipped.append({**report, "raw": block, "reason": autofill.get("reason"), "tool_result": autofill})
                    continue
            else:
                skipped.append({**report, "raw": block, "reason": "unsupported_tool_action"})
                continue

            applied.append({
                **report,
                "decision": decision,
                "tool_action": tool_action,
                "status_before": before_status,
                "status_after": context.get("raspa_status"),
            })

        no_change = bool(no_change_reports) and not applied and not saw_non_no_change
        if no_change:
            context["raspa_status"] = "done_ok"
            context["raspa_fixed_once"] = False
        elif applied and context.get("raspa_status") in {"failed", "needs_review"}:
            context["raspa_status"] = "patched"
            context["raspa_fixed_once"] = True

        return {
            "applied": applied,
            "skipped": skipped,
            "no_change": no_change,
            "no_change_reports": no_change_reports,
        }

    def _run_core_once(self, context: Dict[str, Any]) -> Dict[str, Any]:
        status = context.get("raspa_status")
        if status not in {"needs_review", "failed"}:
            print(f"[RASPAErrorAgent] _run_core_once: raspa_status={status} -> nothing to do.")
            return context

        work_dir_str = context.get("work_dir")
        input_file_str = context.get("input_file")
        if not work_dir_str or not input_file_str:
            raise RuntimeError("[RASPAErrorAgent] work_dir or input_file is missing from context.")

        work_dir = Path(work_dir_str)
        input_path = Path(input_file_str)

        print(f"\n=== RASPAErrorAgent: reviewing completed job in {work_dir} ===")

        error_text = self._gather_error_text(context)
        print("\n[RUNTIME OUTPUT]\n", error_text[:2000], "\n")

        
        file_dict: Dict[str, str] = {}
        file_dict["simulation.input"] = self.read_file(str(input_path))

        for fname in (
            "force_field.def",
            "force_field_mixing_rules.def",
            "pseudo_atoms.def",
        ):
            candidate = work_dir / fname
            if candidate.is_file():
                file_dict[fname] = self.read_file(str(candidate))

        for mol_def in sorted(work_dir.glob("*.def"))[:8]:
            if mol_def.name not in file_dict:
                file_dict[mol_def.name] = self.read_file(str(mol_def))

        fw_name = self._find_framework_name_from_input(input_path)
        if fw_name:
            candidates = [
                work_dir / f"{fw_name}.cif",
                RASPA_DIR / "share" / "raspa" / "structures" / "cif" / f"{fw_name}.cif",
            ]
            for c in candidates:
                if c.is_file():
                    file_dict[c.name] = self.read_file(str(c))
                    break

        output_root = work_dir / "Output"
        if output_root.is_dir():
            for result_path in sorted(output_root.rglob("*.data"))[:4]:
                label = f"result:{result_path.relative_to(work_dir)}"
                file_dict[label] = self.read_file(str(result_path))

        
        rag_hits = self._retrieve_error_knowledge_hits(error_text, file_dict)
        rag_evidence = self._format_error_knowledge(rag_hits)
        if rag_evidence:
            print("\n[RASPA ERROR KNOWLEDGE]\n", rag_evidence[:3000])
            context.setdefault("results", {})["raspa_error_knowledge_hits"] = rag_hits

        selected_forcefield = self._selected_forcefield_from_input(input_path) or context.get("forcefield")
        molecule_name = self._molecule_name_from_input(input_path)
        try:
            from rag.raspa_forcefield_reference import format_raspa_local_reference_evidence

            local_reference_evidence = format_raspa_local_reference_evidence(
                error_msg=error_text,
                selected_forcefield=selected_forcefield,
                molecule_name=molecule_name,
            )
        except Exception as exc:
            local_reference_evidence = f"<< installed RASPA reference retrieval failed: {exc} >>"

        model_selection_evidence = self._format_model_selection_evidence(context)
        if model_selection_evidence:
            context.setdefault("results", {})[
                "raspa_runtime_model_selection_evidence"
            ] = model_selection_evidence

        cif_path = self._find_raspa_cif_path(context, input_path)
        runtime_facts = json.dumps(
            {
                "runtime_observations": (
                    context.get("results") or {}
                ).get("raspa_runtime_observations", []),
                "interaction_mode": INTERACTION_MODE,
                "selected_forcefield": selected_forcefield,
                "molecule_name": molecule_name,
                "cif_path": str(cif_path) if cif_path else None,
                "cif_exists": bool(cif_path and cif_path.is_file()),
                "user_query": (
                    context.get("user_query")
                    or context.get("query_text")
                    or ""
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
        fix_text = self.call_llm_for_fix(
            error_text,
            file_dict,
            rag_evidence=rag_evidence,
            local_reference_evidence=local_reference_evidence,
            model_selection_evidence=model_selection_evidence,
            runtime_facts=runtime_facts,
        )
        print("\nLLM SUGGESTION:\n", fix_text)

        if INTERACTION_MODE == "interactive":
            action, fix_text = self._ask_user_confirmation(
                "RASPAErrorAgent",
                fix_text,
            )
            if action == "skip":
                context["raspa_status"] = "giveup"
                return context

        recovery_result = self._apply_llm_recovery_response(
            context,
            input_path,
            error_text,
            fix_text,
        )
        recovery_rounds = [recovery_result]

        reasoning_round = 0
        while (
            context.get("raspa_status")
            in {"model_evidence_retrieved", "model_change_revision_needed"}
            and reasoning_round < 2
        ):
            reasoning_round += 1
            model_selection_evidence = self._format_model_selection_evidence(
                context
            )
            context.setdefault("results", {})[
                "raspa_runtime_model_selection_evidence"
            ] = model_selection_evidence
            context["raspa_status"] = "needs_review"

            revised_runtime_facts = runtime_facts
            revision_feedback = context.get("raspa_model_revision_feedback")
            if revision_feedback:
                revised_runtime_facts += (
                    "\n\nIndependent model-review rejection to address:\n"
                    + json.dumps(
                        revision_feedback,
                        indent=2,
                        ensure_ascii=False,
                    )
                )

            fix_text = self.call_llm_for_fix(
                error_text,
                file_dict,
                rag_evidence=rag_evidence,
                local_reference_evidence=local_reference_evidence,
                model_selection_evidence=model_selection_evidence,
                runtime_facts=revised_runtime_facts,
            )
            print(
                "\nLLM SUGGESTION AFTER MODEL-EVIDENCE REVIEW:\n",
                fix_text,
            )
            recovery_result = self._apply_llm_recovery_response(
                context,
                input_path,
                error_text,
                fix_text,
            )
            recovery_rounds.append(recovery_result)

        if len(recovery_rounds) > 1:
            recovery_result = {
                "applied": [
                    item
                    for round_result in recovery_rounds
                    for item in round_result.get("applied", [])
                ],
                "skipped": [
                    item
                    for round_result in recovery_rounds
                    for item in round_result.get("skipped", [])
                ],
                "no_change": bool(recovery_rounds[-1].get("no_change")),
                "no_change_reports": recovery_rounds[-1].get(
                    "no_change_reports",
                    [],
                ),
                "rounds": recovery_rounds,
            }
        context.setdefault("results", {})["raspa_error_recovery_mode"] = "rag_llm"
        context["results"]["raspa_llm_recovery"] = recovery_result
        if (
            not recovery_result.get("applied")
            and not recovery_result.get("no_change")
            and context.get("raspa_status") in {"failed", "needs_review"}
        ):
            context["raspa_status"] = "giveup"

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


        
        interval_sec = int(context.get("raspa_poll_interval_sec", 60))

        
        
        for item in batch:
            item.setdefault("raspa_retry", 0)
            item.setdefault("raspa_state", "pending")  

        print(f"[RASPAErrorAgent] batch polling start: n={len(batch)}, interval={interval_sec}s")
        record_job_event(
            context,
            "polling",
            message="RASPA marker polling started",
            metadata={"batch_size": len(batch), "poll_interval_sec": interval_sec},
        )

        while True:
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
                    record_job_event(it, "giveup", message="RASPA missing work_dir or input_file")
                    progressed = True
                    continue

                work_dir = Path(work_dir_str)

                retry = int(it.get("raspa_retry", 0))
                observation = self._observe_job_completion(it, work_dir)

                if not observation["result"]["complete"]:
                    record_scheduler_status(it)
                    continue

                progressed = True

                local_ctx = dict(it)
                local_ctx["results"] = dict(it.get("results") or {})
                local_ctx["results"]["raspa_runtime_observations"] = [
                    observation
                ]
                local_ctx["raspa_status"] = "needs_review"
                record_job_event(
                    local_ctx,
                    "completed_for_review",
                    message="RASPA completion evidence sent to the RAG-grounded LLM",
                    metadata={
                        "retry": retry,
                        "runtime_observation": observation,
                    },
                )

                local_ctx = self._run_core_once(local_ctx)

                if local_ctx.get("raspa_status") == "needs_structure_from_user":
                    it["results"] = local_ctx.get("results", {})
                    it["raspa_status"] = local_ctx.get("raspa_status")
                    it["raspa_state"] = "giveup"
                    it["raspa_structure_request"] = local_ctx.get("raspa_structure_request")
                    record_job_event(
                        it,
                        "giveup",
                        message="RASPA needs a corrected user-provided CIF before retry",
                        metadata=local_ctx.get("raspa_structure_request") or {},
                    )
                    continue

                if local_ctx.get("raspa_status") == "needs_model_selection_from_user":
                    it["results"] = local_ctx.get("results", {})
                    it["raspa_status"] = local_ctx.get("raspa_status")
                    it["raspa_state"] = "giveup"
                    it["raspa_model_selection_request"] = local_ctx.get(
                        "raspa_model_selection_request"
                    )
                    record_job_event(
                        it,
                        "giveup",
                        message="RASPA needs a user-confirmed force field or molecule model before retry",
                        metadata=local_ctx.get("raspa_model_selection_request") or {},
                    )
                    continue

                if local_ctx.get("raspa_status") == "done_ok":
                    it["results"] = local_ctx.get("results", {})
                    it["raspa_status"] = "done_ok"
                    it["raspa_state"] = "done_ok"
                    record_job_event(
                        it,
                        "done_ok",
                        message="RASPA RAG-grounded LLM accepted the output without a patch",
                        metadata=local_ctx.get("results", {}).get("raspa_llm_recovery", {}),
                    )
                    continue

                if local_ctx.get("raspa_status") != "patched":
                    it["results"] = local_ctx.get("results", {})
                    it["raspa_status"] = local_ctx.get("raspa_status") or "giveup"
                    it["raspa_state"] = "giveup"
                    record_job_event(
                        it,
                        "giveup",
                        message="RASPA RAG-grounded LLM produced no applicable recovery",
                        metadata=local_ctx.get("results", {}).get("raspa_llm_recovery", {}),
                    )
                    continue

                max_trials = max(
                    0,
                    int(
                        it.get(
                            "raspa_max_trials",
                            context.get("raspa_max_trials", self.max_trials),
                        )
                    ),
                )
                if retry >= max_trials:
                    it["results"] = local_ctx.get("results", {})
                    it["raspa_state"] = "giveup"
                    it["raspa_status"] = "giveup"
                    record_job_event(
                        it,
                        "giveup",
                        message="RASPA configured maximum retry count reached",
                        metadata={
                            "retry": retry,
                            "max_trials": max_trials,
                        },
                    )
                    continue

                self._clear_flags(work_dir)

                
                raspa_runner = RASPARunner()
                local_ctx = raspa_runner.run(local_ctx)

                
                it["results"] = local_ctx.get("results", {})
                it["pbs_job_name"] = local_ctx.get("pbs_job_name")
                it["raspa_status"] = local_ctx.get("raspa_status")
                it["raspa_job_id"] = local_ctx.get("raspa_job_id")
                it["scheduler_job_id"] = local_ctx.get("scheduler_job_id")

                it["raspa_retry"] = retry + 1
                it["raspa_state"] = "pending"
                record_job_event(
                    it,
                    "retrying",
                    message="RASPA auto-patch applied; retry submitted",
                    metadata={"retry": it["raspa_retry"]},
                )

            if not progressed:
                time.sleep(interval_sec)

        
        context.setdefault("results", {})
        n_ok = sum(1 for it in batch if it.get("raspa_state") == "done_ok")
        n_fail = sum(1 for it in batch if it.get("raspa_state") == "giveup")
        context["results"]["raspa_error_summary"] = {"done_ok": n_ok, "giveup": n_fail, "total": len(batch)}
        final_status = "done_ok" if n_ok == len(batch) else "partial_or_failed"
        record_job_event(
            context,
            final_status,
            message="RASPA polling completed",
            metadata=context["results"]["raspa_error_summary"],
        )

        
        if created_batch_wrapper:
            sub = batch[0]
            context.setdefault("results", {})
            context["results"].update(sub.get("results", {}))
            context["raspa_status"] = sub.get("raspa_status", context.get("raspa_status"))
            context["raspa_retry"] = sub.get("raspa_retry", context.get("raspa_retry", 0))
            context["raspa_state"] = sub.get("raspa_state", context.get("raspa_state"))
            if sub.get("raspa_model_selection_request"):
                context["raspa_model_selection_request"] = sub[
                    "raspa_model_selection_request"
                ]
            context.pop("batch", None)  
        else:
            
            context["batch"] = batch

        return context
