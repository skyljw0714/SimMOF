import os
import re
import subprocess
import time
import math

from collections import deque
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from config import AGENT_LLM_MAP, LLM_DEFAULT, INTERACTION_MODE, ask_user_confirmation
from core.job_manager import record_job_event, record_scheduler_status
from core.llm_logging import log_llm_decision

from .agent import ErrorAgent
from .structure_regeneration import request_structure_regeneration


class LAMMPSErrorAgent(ErrorAgent):
    DEFAULT_FORCEFIELD_RECOVERY_POLICY = "generic_uff_baseline"

    UFF_LJ_BY_ELEMENT = {
        "H": (0.044, 2.571),
        "B": (0.095, 3.638),
        "C": (0.105, 3.431),
        "N": (0.069, 3.261),
        "O": (0.060, 3.118),
        "F": (0.050, 2.997),
        "Na": (0.030, 2.983),
        "Mg": (0.111, 3.021),
        "Al": (0.505, 4.008),
        "Si": (0.402, 3.826),
        "P": (0.200, 3.694),
        "S": (0.274, 3.594),
        "Cl": (0.227, 3.516),
        "K": (0.035, 3.812),
        "Ca": (0.238, 3.399),
        "Ti": (0.017, 2.829),
        "Cr": (0.015, 2.694),
        "Mn": (0.013, 2.638),
        "Fe": (0.013, 2.594),
        "Co": (0.014, 2.558),
        "Ni": (0.015, 2.525),
        "Cu": (0.005, 3.114),
        "Zn": (0.124, 2.763),
        "Br": (0.251, 3.732),
        "Zr": (0.069, 2.783),
        "I": (0.339, 4.009),
    }

    ELEMENT_BY_APPROX_MASS = (
        ("H", 1.008),
        ("B", 10.81),
        ("C", 12.011),
        ("N", 14.007),
        ("O", 15.999),
        ("F", 18.998),
        ("Na", 22.990),
        ("Mg", 24.305),
        ("Al", 26.982),
        ("Si", 28.085),
        ("P", 30.974),
        ("S", 32.060),
        ("Cl", 35.450),
        ("K", 39.098),
        ("Ca", 40.078),
        ("Ti", 47.867),
        ("Cr", 51.996),
        ("Mn", 54.938),
        ("Fe", 55.845),
        ("Co", 58.933),
        ("Ni", 58.693),
        ("Cu", 63.546),
        ("Zn", 65.380),
        ("Br", 79.904),
        ("Zr", 91.224),
        ("I", 126.904),
    )

    def __init__(self, llm=None, log_file="log.lammps", input_files=None, max_lines=200):
        self._init_error_agent(
            llm=llm,
            default_llm=AGENT_LLM_MAP.get("LAMMPSErrorAgent", LLM_DEFAULT),
            max_lines=max_lines,
        )
        self.log_file = log_file
        self.input_files = input_files or ["system.in", "system.in.settings", "system.in.init", "system.data"]

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

    def _error_query_text(self, error_msg: str) -> str:
        error_lines = []
        warning_lines = []
        last_command = []
        for line in (error_msg or "").splitlines():
            stripped = line.strip()
            if re.search(r"\bERROR\b", line, flags=re.IGNORECASE):
                error_lines.append(stripped)
            elif re.search(r"\bWARNING\b", line, flags=re.IGNORECASE):
                warning_lines.append(stripped)
            elif re.search(r"Last command", line, flags=re.IGNORECASE):
                last_command.append(stripped)
        lines = error_lines or warning_lines
        lines.extend(last_command)
        return "\n".join(lines) if lines else (error_msg or "")

    def _retrieve_error_knowledge_hits(self, error_msg: str, file_dict: Dict[str, str]):
        try:
            from rag.lammps_error_knowledge import LAMMPSErrorKnowledgeBase

            hits = LAMMPSErrorKnowledgeBase().search(self._error_query_text(error_msg), top_k=5)
            return [hit for hit in hits if float(hit.get("score") or 0.0) >= 7.5]
        except Exception as exc:
            print(f"[LAMMPSErrorAgent] LAMMPS error knowledge retrieval disabled: {exc}")
            return []

    def _format_error_knowledge(self, hits) -> str:
        if not hits:
            return ""
        try:
            from rag.lammps_error_knowledge import LAMMPSErrorKnowledgeBase

            return LAMMPSErrorKnowledgeBase().format_hits(hits, max_chars=4500)
        except Exception as exc:
            print(f"[LAMMPSErrorAgent] LAMMPS error knowledge formatting disabled: {exc}")
            return ""

    def _format_forcefield_reference(self, error_msg: str, file_dict: Dict[str, str]) -> str:
        try:
            from rag.lammps_forcefield_reference import format_lammps_forcefield_reference_evidence

            return format_lammps_forcefield_reference_evidence(error_msg=error_msg, file_dict=file_dict)
        except Exception as exc:
            print(f"[LAMMPSErrorAgent] force-field reference evidence disabled: {exc}")
            return ""

    def call_llm_for_fix(
        self,
        error_msg,
        file_dict,
        rag_evidence: str = "",
        forcefield_literature_evidence: str = "",
        allow_forcefield_parameter_reference: bool = True,
        forcefield_recovery_policy: str = "",
    ):
        system_prompt = (
            "You are a LAMMPS simulation troubleshooting assistant. This simulation uses LAMMPS (3 Mar 2020).\n"
            "Given an ERROR message, retrieved source/manual evidence, and input files, decide the safest recovery.\n"
            "Rules for your response:\n"
            "- Always provide the smallest number of changes necessary to resolve the ERROR.\n"
            "- Never suggest contradictory changes (e.g., both removing and re-adding the same line).\n"
            "- Never duplicate the same command (e.g., do not add multiple identical `kspace_style` lines).\n"
            "- Do not propose cosmetic changes unless they are required for correctness.\n"
            "- Use the retrieved official LAMMPS error documentation when it matches the log.\n"
            "- Prefer fixes supported by the current error message, source file hint, and input files.\n"
            "- Preserve the user's intended physical model. Do not make the input merely runnable by removing "
            "interactions, charges, atom types, molecules, or force-field terms that the original input intended to use.\n"
            "- Do not replace an interacting model with `pair_style none` unless the user explicitly requested a "
            "topology-only/no-interaction check. A `run 0` command is still a model validation step and should preserve "
            "the intended force field.\n"
            "- Report the physical or simulation-engine rationale for every suggested fix using `RATIONALE:`.\n"
            "- Report the concrete evidence used for every suggested fix using `EVIDENCE:`.\n"
            "- Do not invent generic force-field constants. Never propose wildcard coefficients such as "
            "`pair_coeff * * 0.1 3.0` or arbitrary epsilon/sigma values.\n"
            "- The proposed patch must be runnable as-is for the provided input. Do not defer required "
            "force-field definitions to a later stage inside an LLM patch.\n"
            "- If pair coefficients are missing, prefer returning to the force-field/input-generation stage. "
            "If you must write coefficients, infer atom elements from `system.data` Masses labels/masses, existing "
            "`pair_coeff` values, or atom-type comments, and state the parameter source and mixing rule.\n"
            "- If replacing an invalid `pair_style` with an LJ style, also check whether all required `pair_coeff` "
            "entries are present. If they are missing and elements can be inferred, include the missing coefficients "
            "in the same answer so the patch reruns successfully.\n"
            "- If masses are missing, edit `system.data` Masses entries; do not add `mass` commands after `read_data` "
            "unless the data file cannot be edited.\n"
            "- Never use placeholder masses such as 1.0 for non-hydrogen atom types. If element identity cannot be "
            "inferred from atom-type labels, existing LJ coefficients, masses, or supplied files, state that safe automatic "
            "correction is not possible.\n"
            "- When masses or LJ constants are required, use only file-backed force-field reference evidence supplied "
            "in the user prompt. Do not rely on memorized element tables. If the evidence does not cover the required "
            "atom type or element, state that safe automatic correction is not possible.\n"
            "- A missing force-field assignment is not enough evidence to select a specialized scientific model. "
            "If choosing a framework force-field family or guest model is required and no literature model-selection "
            "evidence is supplied, use CONSULT_RAG_AGENT_FOR_FORCEFIELD instead of writing coefficients.\n"
            "- Do not use REQUEST_USER_MODEL_SELECTION before the RagAgent consultation. Use it only after literature "
            "model-selection evidence has been supplied and remains ambiguous/incomplete, or when the supplied "
            "consultation result explicitly reports that retrieval failed.\n"
            "- SimMOF may explicitly supply the recovery policy `generic_uff_baseline`. Under that policy, if no "
            "force-field family was explicitly requested in the user context or existing input, use generic elemental "
            "UFF Lennard-Jones parameters as the automatic screening baseline when every atom type is covered by the "
            "file-backed UFF reference and the current pair style is compatible. Use Lorentz-Berthelot mixing and label "
            "the inserted terms as a generic fallback. The presence of other literature force-field families does not "
            "by itself require user selection because this workflow policy has already selected the generic baseline.\n"
            "- The generic fallback does not override an explicitly requested force field, a literature statement that "
            "UFF is incompatible with the target, missing element coverage, unresolved atom typing, charges or bonded "
            "terms required by the requested model, or an incompatible pair style. In those cases, use "
            "REGENERATE_LAMMPS_INPUTS or REQUEST_USER_MODEL_SELECTION as appropriate.\n"
            "- Literature evidence can justify a specialized model family, but it is not a numerical parameter file. "
            "When the generic fallback policy is not active and the literature presents multiple incompatible model "
            "families or lacks a complete compatible parameterization, use REQUEST_USER_MODEL_SELECTION.\n"
            "- Use REGENERATE_LAMMPS_INPUTS when the intended model is sufficiently supported but the atom typing, "
            "topology, or coefficients must be rebuilt consistently by the input-generation stage.\n"
            "- A missing molecule file may be corrected as a text patch only when an exact existing molecule-template "
            "file for the same named guest is present in the supplied files. Do not synthesize molecule topology or "
            "charges from memory.\n"
            "- If the docs indicate a likely unstable trajectory/geometry problem, prefer safer run-protocol changes "
            "such as smaller timestep, neighbor-list updates, minimization, or structure/packing review over arbitrary coefficients.\n"
            "- For `Lost atoms` caused by severe initial overlaps or extremely steep forces, do not rely on minimization alone. "
            "The patch must include a combined stabilization protocol: reduce timestep substantially, insert a conservative "
            "minimize before dynamics, and replace plain `fix ... nve` with `fix ... nve/limit <small displacement>` "
            "when a short rescue run is needed. "
            "If the geometry is physically impossible, state that structure regeneration is required.\n"
            "- Treat `kspace_style` as a solver-initialization command dependent on final box geometry.\n"
            "- Never suggest adding or redefining `kspace_style` after a `minimize` or `run` command.\n"
            "- If fixing `kspace_style`, place it AFTER box geometry is finalized and STRICTLY BEFORE the first `minimize` or `run`.\n"
            "- Reason about the command graph before patching: every `unfix`/`uncompute` must reference a live ID, "
            "a patch must not create duplicate commands, and an exact ID must never be confused with a longer ID sharing its prefix.\n"
            "- After a state-changing `run`, `minimize`, or box operation, verify that computes consumed by variables, "
            "prints, or fixes are current at the point of use. Add an evaluation step only when it is required by "
            "the command lifecycle and does not change the requested physical protocol.\n"
            "- Check whether thermostat and velocity operations have usable translational or rotational degrees of "
            "freedom for their target group. Infer the remedy from the group size, constraints, and intended ensemble; "
            "do not rely on a task-name-specific exception.\n"
            "- For cell-instability errors, diagnose whether the initial geometry, timestep, pressure coupling, or "
            "active cell degrees of freedom caused the divergence. Preserve only the cell degrees of freedom required "
            "by the requested observable and do not hide an unstable physical protocol with an unrelated output change.\n"
            "\n"
            "Output format (strict): choose exactly one decision.\n"
            "\n"
            "For a safe text-only input correction:\n"
            "RATIONALE: <why this recovery is justified>\n"
            "EVIDENCE: <specific log, retrieved evidence, and input-file facts>\n"
            "DECISION: TEXT_PATCH\n"
            "FILE: <filename>\n"
            "ACTION: <copy exactly one action header from the list below; do not paraphrase it>\n"
            "<payload, following exactly one action pattern below>\n"
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
            "\n"
            "For a recovery requiring another agent or user decision, return exactly:\n"
            "RATIONALE: <why the action is required>\n"
            "EVIDENCE: <specific supporting facts>\n"
            "DECISION: TOOL_ACTION\n"
            "TOOL ACTION: <one of CONSULT_RAG_AGENT_FOR_FORCEFIELD, REGENERATE_LAMMPS_INPUTS, "
            "REQUEST_USER_MODEL_SELECTION>\n"
            "Do not include FILE or ACTION for a tool action. Return your response strictly in one of these formats."
        )

        user_prompt = f"ERROR message from LAMMPS log:\n{error_msg}\n\n"
        if forcefield_recovery_policy:
            user_prompt += (
                "SimMOF force-field recovery policy:\n"
                f"{forcefield_recovery_policy}\n\n"
            )
        if rag_evidence:
            user_prompt += "Retrieved official LAMMPS error documentation:\n"
            user_prompt += rag_evidence
            user_prompt += "\n\n"
        if forcefield_literature_evidence:
            user_prompt += "RagAgent literature evidence for force-field/model selection:\n"
            user_prompt += forcefield_literature_evidence
            user_prompt += "\n\n"
        forcefield_reference = (
            self._format_forcefield_reference(error_msg, file_dict)
            if allow_forcefield_parameter_reference
            else ""
        )
        if forcefield_reference:
            user_prompt += "Retrieved local force-field reference evidence:\n"
            user_prompt += forcefield_reference
            user_prompt += "\n\n"
        for fname, content in file_dict.items():
            user_prompt += f"\n----- {fname} -----\n{content}\n"

        result = self._invoke_llm(system_prompt, user_prompt,
                                  agent="LAMMPSErrorAgent", label="runtime_error_fix")
        try:
            log_llm_decision(
                "LAMMPSErrorAgent",
                "runtime_error_fix",
                {
                    "error_preview": error_msg[:300],
                    "rag_evidence": rag_evidence[:2000],
                    "forcefield_literature_evidence": forcefield_literature_evidence[:3000],
                    "forcefield_reference": forcefield_reference[:2000],
                    "patch": result[:2000],
                },
            )
        except Exception:
            pass
        return result

    @staticmethod
    def _response_field(response: str, name: str) -> Optional[str]:
        match = re.search(
            rf"(?ms)^\s*{re.escape(name)}\s*:\s*(.*?)"
            rf"(?=^\s*(?:RATIONALE|EVIDENCE|DECISION|FILE|ACTION|TOOL ACTION)\s*:|\Z)",
            response or "",
        )
        value = match.group(1).strip() if match else ""
        return value or None

    def _consult_rag_agent_for_forcefield(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        cached = context.get("lammps_rag_hints")
        if isinstance(cached, dict) and (cached.get("ff_hints") or "").strip():
            return {
                "status": "ok",
                "source": "context_cache",
                **cached,
            }

        from rag.agent import RagAgent

        rag_context = {
            "job_name": context.get("job_name") or "",
            "mof": context.get("mof") or "",
            "guest": context.get("guest") or "",
            "property": context.get("property") or "",
            "query_text": context.get("query_text") or context.get("QueryText") or "",
        }
        result = RagAgent(agent_name="RagAgent").run_for_lammps_ff(
            rag_context,
            top_files=5,
        )
        hints = {
            "ff_hints": (result.get("ff_hints") or "").strip(),
            "charge_hints": (result.get("charge_hints") or "").strip(),
        }
        context["lammps_rag_hints"] = hints
        return {
            "status": "ok" if hints["ff_hints"] else "no_evidence",
            "source": "RagAgent.run_for_lammps_ff",
            **hints,
            "queries": result.get("lammps_ff_queries") or [],
            "top_file_hits": result.get("top_file_hits") or [],
        }

    def _review_forcefield_recovery(
        self,
        *,
        error_msg: str,
        proposed_response: str,
        literature_evidence: str,
        recovery_policy: str,
    ) -> Dict[str, Any]:
        system_prompt = (
            "You are an independent scientific reviewer of a proposed LAMMPS force-field recovery.\n"
            "Decide whether the proposed automatic action is supported by the supplied literature evidence and "
            "preserves the intended model. A LAMMPS pair_style alone does not identify a force-field family. "
            "A local parameter file for one family is a numerical source, not by itself evidence that this family was "
            "intended.\n"
            "When the supplied recovery policy is `generic_uff_baseline`, however, the workflow has explicitly selected "
            "generic elemental UFF Lennard-Jones parameters as the default screening model whenever the user/input did "
            "not name another force field. For that policy, do not reject merely because the literature contains CVFF, "
            "DREIDING, modified-UFF, or other alternatives. Instead, accept only if the proposal uses file-backed UFF "
            "values for every inferred atom type, uses a compatible plain-LJ pair style and a stated mixing rule, and "
            "does not silently discard required charges or bonded terms. Reject if UFF is explicitly contraindicated, "
            "coverage is incomplete, atom typing is unresolved, or another model was explicitly requested.\n"
            "When that generic policy is not active, reject when the literature offers multiple incompatible families, "
            "covers a different MOF, describes a modified/reparameterized variant while the proposal uses generic "
            "parameters, or lacks a complete compatible parameterization. Return exactly:\n"
            "VERDICT: ACCEPT or REJECT\n"
            "RATIONALE: <one paragraph>\n"
            "EVIDENCE: <specific supplied facts>"
        )
        user_prompt = (
            f"LAMMPS error:\n{error_msg}\n\n"
            f"SimMOF recovery policy:\n{recovery_policy}\n\n"
            f"RagAgent literature evidence:\n{literature_evidence}\n\n"
            f"Proposed recovery:\n{proposed_response}"
        )
        raw = self._invoke_llm(
            system_prompt,
            user_prompt,
            agent="LAMMPSErrorAgent",
            label="forcefield_recovery_review",
        )
        verdict = (self._response_field(raw, "VERDICT") or "").upper()
        return {
            "verdict": verdict,
            "rationale": self._response_field(raw, "RATIONALE") or "",
            "evidence": self._response_field(raw, "EVIDENCE") or "",
            "raw": raw,
        }

    def decide_runtime_recovery(
        self,
        error_msg: str,
        file_dict: Dict[str, str],
        *,
        context: Optional[Dict[str, Any]] = None,
        rag_evidence: str = "",
    ) -> Dict[str, Any]:
        context = context if context is not None else {}
        recovery_policy = str(
            context.get("lammps_forcefield_recovery_policy")
            or self.DEFAULT_FORCEFIELD_RECOVERY_POLICY
        )
        context["lammps_forcefield_recovery_policy"] = recovery_policy
        response = self.call_llm_for_fix(
            error_msg,
            file_dict,
            rag_evidence=rag_evidence,
            allow_forcefield_parameter_reference=False,
            forcefield_recovery_policy=recovery_policy,
        )
        consultations = []
        model_change_reviews = []
        decision = (self._response_field(response, "DECISION") or "").upper()
        tool_action = (self._response_field(response, "TOOL ACTION") or "").upper()

        if decision == "TOOL_ACTION" and tool_action == "CONSULT_RAG_AGENT_FOR_FORCEFIELD":
            try:
                consultation = self._consult_rag_agent_for_forcefield(context)
            except Exception as exc:
                consultation = {
                    "status": "failed",
                    "source": "RagAgent.run_for_lammps_ff",
                    "error": str(exc),
                    "ff_hints": "",
                }
            consultations.append(consultation)
            response = self.call_llm_for_fix(
                error_msg,
                file_dict,
                rag_evidence=rag_evidence,
                forcefield_literature_evidence=consultation.get("ff_hints") or "",
                allow_forcefield_parameter_reference=True,
                forcefield_recovery_policy=recovery_policy,
            )
            decision = (self._response_field(response, "DECISION") or "").upper()
            tool_action = (self._response_field(response, "TOOL ACTION") or "").upper()
            if consultation.get("ff_hints") and (
                decision == "TEXT_PATCH"
                or tool_action == "REGENERATE_LAMMPS_INPUTS"
            ):
                review = self._review_forcefield_recovery(
                    error_msg=error_msg,
                    proposed_response=response,
                    literature_evidence=consultation.get("ff_hints") or "",
                    recovery_policy=recovery_policy,
                )
                model_change_reviews.append(review)
                if review.get("verdict") != "ACCEPT":
                    response = (
                        f"RATIONALE: {review.get('rationale') or 'The proposed force-field recovery is not uniquely supported.'}\n"
                        f"EVIDENCE: {review.get('evidence') or 'RagAgent evidence did not establish one compatible model and parameter source.'}\n"
                        "DECISION: TOOL_ACTION\n"
                        "TOOL ACTION: REQUEST_USER_MODEL_SELECTION"
                    )
                    decision = "TOOL_ACTION"
                    tool_action = "REQUEST_USER_MODEL_SELECTION"

        return {
            "response": response,
            "decision": decision or ("TEXT_PATCH" if "FILE:" in response else "UNKNOWN"),
            "tool_action": tool_action,
            "forcefield_recovery_policy": recovery_policy,
            "rag_agent_consultations": consultations,
            "model_change_reviews": model_change_reviews,
        }

    def _append_line_if_missing(self, path: Path, line: str) -> bool:
        text = path.read_text(errors="ignore") if path.exists() else ""
        if line.strip() in text:
            return False
        if text and not text.endswith("\n"):
            text += "\n"
        text += line.rstrip() + "\n"
        path.write_text(text)
        return True

    def _replace_first_matching_line(self, path: Path, pattern: str, replacement: str) -> bool:
        if not path.exists():
            return False
        lines = path.read_text(errors="ignore").splitlines()
        changed = False
        out = []
        for line in lines:
            if not changed and re.search(pattern, line):
                out.append(replacement)
                changed = True
            else:
                out.append(line)
        if changed:
            path.write_text("\n".join(out).rstrip() + "\n")
        return changed

    def _remove_lines_matching(self, path: Path, pattern: str) -> int:
        if not path.exists():
            return 0
        lines = path.read_text(errors="ignore").splitlines()
        kept = [line for line in lines if not re.search(pattern, line)]
        removed = len(lines) - len(kept)
        if removed:
            path.write_text("\n".join(kept).rstrip() + "\n")
        return removed

    def _parse_system_data_atoms(self, data_path: Path):
        if not data_path.exists():
            return []
        lines = data_path.read_text(errors="ignore").splitlines()
        atoms = []
        in_atoms = False
        for raw in lines:
            stripped = raw.strip()
            if not stripped:
                if in_atoms and atoms:
                    break
                continue
            if re.match(r"^Atoms\b", stripped, flags=re.IGNORECASE):
                in_atoms = True
                continue
            if not in_atoms or stripped.startswith("#"):
                continue
            if re.match(r"^[A-Za-z]", stripped):
                break
            body = raw.split("#", 1)[0].split()
            if len(body) < 5 or not body[0].isdigit():
                continue
            try:
                atom_id = int(body[0])
                atom_type = int(body[2]) if len(body) >= 7 else int(body[1])
                charge = float(body[3]) if len(body) >= 7 else 0.0
                x, y, z = (float(v) for v in body[-3:])
            except (ValueError, IndexError):
                continue
            atoms.append(
                {
                    "id": atom_id,
                    "type": atom_type,
                    "charge": charge,
                    "x": x,
                    "y": y,
                    "z": z,
                }
            )
        return atoms

    def _system_data_has_charges(self, data_path: Path, tol: float = 1.0e-12) -> bool:
        return any(abs(atom.get("charge", 0.0)) > tol for atom in self._parse_system_data_atoms(data_path))

    def _closest_atom_pair(self, data_path: Path, max_atoms: int = 12000) -> Optional[Dict[str, Any]]:
        atoms = self._parse_system_data_atoms(data_path)
        if len(atoms) < 2 or len(atoms) > max_atoms:
            return None
        best = None
        best_r2 = float("inf")
        for i, left in enumerate(atoms):
            for right in atoms[i + 1:]:
                dx = left["x"] - right["x"]
                dy = left["y"] - right["y"]
                dz = left["z"] - right["z"]
                r2 = dx * dx + dy * dy + dz * dz
                if r2 < best_r2:
                    best_r2 = r2
                    best = (left, right)
        if best is None:
            return None
        return {
            "distance": math.sqrt(best_r2),
            "atom_i": best[0],
            "atom_j": best[1],
        }

    def _structure_overlap_precheck(self, context: Dict[str, Any], threshold: float = 0.60) -> Dict[str, Any]:
        work_dir = Path(context.get("work_dir") or "")
        data_path = work_dir / "system.data"
        closest = self._closest_atom_pair(data_path)
        if not closest or closest["distance"] >= threshold:
            return {"status": "ok", "closest_pair": closest}
        result = {
            "status": "needs_structure_regeneration",
            "reason": "atom overlap below LAMMPS pre-run threshold",
            "threshold_angstrom": threshold,
            "closest_pair": closest,
            "suggested_action": "regenerate guest packing or structure before submitting LAMMPS",
        }
        context["lammps_status"] = "needs_structure_regeneration"
        context["lammps_needs_structure_regeneration"] = True
        context.setdefault("results", {})["lammps_structure_precheck"] = result
        request_structure_regeneration(
            context,
            software="lammps",
            reason=result["reason"],
            action="regenerate_packing_and_lammps_inputs",
            metadata=result,
        )
        return result

    def _patch_charge_kspace_protocol(self, work_dir: Path) -> Optional[Dict[str, Any]]:
        system_in = work_dir / "system.in"
        data_path = work_dir / "system.data"
        if not system_in.exists() or not data_path.exists():
            return None

        text = system_in.read_text(errors="ignore")
        has_long_coulomb = bool(re.search(r"pair_style\s+.*coul/long", text, flags=re.IGNORECASE))
        charged = self._system_data_has_charges(data_path)
        if not charged and not has_long_coulomb:
            return None

        pair_style = "pair_style lj/cut/coul/long 10.0"
        kspace_style = "kspace_style pppm 1.0e-4"
        lines = text.splitlines()
        out = []
        inserted = False
        changed = False
        for line in lines:
            stripped = line.strip()
            keyword = stripped.split(None, 1)[0].lower() if stripped else ""
            if keyword in {"pair_style", "kspace_style", "kspace_modify"}:
                if stripped not in {pair_style, kspace_style}:
                    changed = True
                continue
            out.append(line)
            if not inserted and stripped in {'read_data "system.data"', "read_data system.data"}:
                out.append(pair_style)
                out.append(kspace_style)
                inserted = True
                changed = True

        if not inserted:
            out.insert(0, kspace_style)
            out.insert(0, pair_style)
            changed = True

        if changed:
            system_in.write_text("\n".join(out).rstrip() + "\n")
            return {
                "status": "patched",
                "action": "ensure_charge_kspace_protocol",
                "charged_system_data": charged,
                "files_changed": ["system.in"],
            }
        return None

    def _apply_lammps_stability_protocol(self, work_dir: Path) -> Optional[Dict[str, Any]]:
        system_in = work_dir / "system.in"
        if not system_in.exists():
            return None
        lines = system_in.read_text(errors="ignore").splitlines()
        if any("simmof_lammps_stability_protocol" in line for line in lines):
            return None

        changed = False
        has_neighbor = any(re.match(r"^\s*neighbor\b", line) for line in lines)
        has_neigh_modify = any(re.match(r"^\s*neigh_modify\b", line) for line in lines)
        has_minimize = any(re.match(r"^\s*minimize\b", line) for line in lines)
        out = []
        inserted_before_run = False

        for line in lines:
            m = re.match(r"^(\s*timestep\s+)([-+0-9.eE]+)(.*)$", line)
            if m:
                try:
                    old_dt = float(m.group(2))
                    new_dt = min(old_dt, 0.25)
                    if new_dt != old_dt:
                        out.append(f"{m.group(1)}{new_dt:g}{m.group(3)}  # simmof_lammps_stability_protocol")
                        changed = True
                        continue
                except ValueError:
                    pass

            if not inserted_before_run and re.match(r"^\s*run\s+", line):
                out.append("# simmof_lammps_stability_protocol: conservative recovery for unstable dynamics")
                if not has_neighbor:
                    out.append("neighbor 2.0 bin")
                if not has_neigh_modify:
                    out.append("neigh_modify delay 0 every 1 check yes")
                if not has_minimize:
                    out.append("min_style cg")
                    out.append("minimize 1.0e-6 1.0e-8 1000 10000")
                inserted_before_run = True
                changed = True
            out.append(line)

        if changed:
            system_in.write_text("\n".join(out).rstrip() + "\n")
            return {
                "status": "patched",
                "action": "apply_stability_protocol",
                "files_changed": ["system.in"],
                "changes": [
                    "reduced timestep to at most 0.25 fs",
                    "ensured neighbor/neigh_modify before first run",
                    "inserted conservative minimize before first run when missing",
                ],
            }
        return None

    def _request_lammps_structure_regeneration(self, context: Dict[str, Any], reason: str) -> Dict[str, Any]:
        result = {
            "status": "needs_structure_regeneration",
            "method": "structure_regeneration_request",
            "reason": reason,
            "actions": ["regenerate structure or guest packing before retrying LAMMPS"],
        }
        context["lammps_status"] = "needs_structure_regeneration"
        context["lammps_needs_structure_regeneration"] = True
        context.setdefault("results", {})["lammps_structure_recovery"] = result
        request_structure_regeneration(
            context,
            software="lammps",
            reason=reason,
            action="regenerate_packing_and_lammps_inputs",
            metadata=result,
        )
        return result

    def _has_any_pair_coeff(self, work_dir: Path) -> bool:
        settings = work_dir / "system.in.settings"
        system_in = work_dir / "system.in"
        text = ""
        if settings.exists():
            text += settings.read_text(errors="ignore")
        if system_in.exists():
            text += "\n" + system_in.read_text(errors="ignore")
        return bool(re.search(r"(?m)^\s*pair_coeff\b", text))

    def _current_pair_style_supports_lj_pair_table(self, work_dir: Path) -> bool:
        system_in = work_dir / "system.in"
        if not system_in.exists():
            return True
        text = system_in.read_text(errors="ignore")
        styles = [
            line.split("#", 1)[0].strip()
            for line in text.splitlines()
            if re.match(r"^\s*pair_style\b", line)
        ]
        if not styles:
            return True
        joined = "\n".join(styles).lower()
        if "hybrid" in joined:
            return False
        return ("lj/cut" in joined) or ("lj/charmm" in joined)

    def _forcefield_lj_by_element(self) -> Dict[str, Dict[str, Any]]:
        if hasattr(self, "_cached_forcefield_lj_by_element"):
            return self._cached_forcefield_lj_by_element

        values: Dict[str, Dict[str, Any]] = {}
        try:
            from rag.lammps_forcefield_reference import load_raspa_uff_reference

            for element, item in load_raspa_uff_reference().get("by_element", {}).items():
                values[element] = {
                    "epsilon": float(item["epsilon_kcal_per_mol"]),
                    "sigma": float(item["sigma_angstrom"]),
                    "mass": float(item["mass"]),
                    "source": "RASPA_UFF",
                    "pseudo_atoms_path": item.get("pseudo_atoms_path"),
                    "pseudo_atoms_line": item.get("pseudo_atoms_line"),
                    "mixing_rules_path": item.get("mixing_rules_path"),
                    "mixing_rules_line": item.get("mixing_rules_line"),
                }
        except Exception as exc:
            print(f"[LAMMPSErrorAgent] file-backed UFF reference unavailable; using built-in fallback: {exc}")

        if not values:
            values = {
                element: {"epsilon": epsilon, "sigma": sigma, "source": "built_in_fallback"}
                for element, (epsilon, sigma) in self.UFF_LJ_BY_ELEMENT.items()
            }

        self._cached_forcefield_lj_by_element = values
        return values

    def _infer_element_from_label_or_mass(self, label: str, mass: Optional[float]) -> Optional[str]:
        lj_by_element = self._forcefield_lj_by_element()
        cleaned = re.sub(r"[^A-Za-z]", " ", label or "").strip()
        tokens = cleaned.split()
        for token in tokens:
            token = token.strip("_")
            if not token:
                continue
            candidates = [
                token[:2].title(),
                token[:1].upper(),
            ]
            for cand in candidates:
                if cand in lj_by_element:
                    return cand

        if mass is not None:
            element, nearest = min(
                self.ELEMENT_BY_APPROX_MASS,
                key=lambda item: abs(item[1] - mass),
            )
            if abs(nearest - mass) <= max(0.75, nearest * 0.04):
                return element
        return None

    def _parse_atom_types_from_masses(self, data_path: Path) -> Dict[int, Dict[str, Any]]:
        if not data_path.exists():
            return {}
        text = data_path.read_text(errors="ignore")
        types: Dict[int, Dict[str, Any]] = {}
        in_masses = False
        for raw in text.splitlines():
            stripped = raw.strip()
            if not stripped:
                continue
            if re.match(r"^Masses\b", stripped, flags=re.IGNORECASE):
                in_masses = True
                continue
            if in_masses and re.match(r"^[A-Za-z]", stripped):
                break
            if not in_masses:
                continue

            body, _, comment = raw.partition("#")
            parts = body.split()
            if len(parts) < 2 or not parts[0].isdigit():
                continue
            try:
                atom_type = int(parts[0])
                mass = float(parts[1])
            except ValueError:
                continue
            label = comment.strip()
            element = self._infer_element_from_label_or_mass(label, mass)
            lj_by_element = self._forcefield_lj_by_element()
            if element and element in lj_by_element:
                ff = lj_by_element[element]
                types[atom_type] = {
                    "mass": mass,
                    "label": label,
                    "element": element,
                    "epsilon": ff["epsilon"],
                    "sigma": ff["sigma"],
                    "source": ff.get("source"),
                    "pseudo_atoms_path": ff.get("pseudo_atoms_path"),
                    "pseudo_atoms_line": ff.get("pseudo_atoms_line"),
                    "mixing_rules_path": ff.get("mixing_rules_path"),
                    "mixing_rules_line": ff.get("mixing_rules_line"),
                }
        return types

    def _write_uff_like_pair_coeffs_from_atom_types(self, work_dir: Path) -> Tuple[bool, Dict[str, Any]]:
        settings = work_dir / "system.in.settings"
        data_path = work_dir / "system.data"
        if self._has_any_pair_coeff(work_dir):
            return False, {"reason": "pair_coeff already exists"}
        if not self._current_pair_style_supports_lj_pair_table(work_dir):
            return False, {"reason": "current pair_style is hybrid or not a plain LJ style"}

        atom_types = self._parse_atom_types_from_masses(data_path)
        if not atom_types:
            return False, {"reason": "could not infer atom elements from system.data Masses"}

        expected_types = None
        text = data_path.read_text(errors="ignore") if data_path.exists() else ""
        m = re.search(r"(?m)^\s*(\d+)\s+atom\s+types\b", text)
        if m:
            expected_types = int(m.group(1))
            missing = [idx for idx in range(1, expected_types + 1) if idx not in atom_types]
            if missing:
                return False, {"reason": "not all atom types could be assigned UFF-like LJ parameters", "missing_types": missing}

        lines = [
            "",
            "# UFF-like LJ fallback generated by LAMMPS error agent after input regeneration failed to set pair_coeff.",
            "# epsilon: kcal/mol, sigma: Angstrom; cross terms use Lorentz-Berthelot mixing.",
        ]
        type_ids = sorted(atom_types)
        for i in type_ids:
            for j in type_ids:
                if j < i:
                    continue
                left = atom_types[i]
                right = atom_types[j]
                epsilon = math.sqrt(float(left["epsilon"]) * float(right["epsilon"]))
                sigma = 0.5 * (float(left["sigma"]) + float(right["sigma"]))
                lines.append(
                    f"pair_coeff {i} {j} {epsilon:.6f} {sigma:.6f} "
                    f"# {left['element']}-{right['element']} UFF_like"
                )

        if settings.exists():
            current = settings.read_text(errors="ignore").rstrip()
        else:
            current = ""
        settings.write_text((current + "\n" if current else "") + "\n".join(lines).strip() + "\n")
        return True, {
            "source": "UFF_like_element_parameters",
            "mixing_rule": "Lorentz-Berthelot",
            "atom_types": atom_types,
            "n_atom_types": expected_types or len(atom_types),
            "n_pair_coeffs": len(lines) - 3,
        }

    def _regenerate_lammps_inputs_from_context(self, context: Dict[str, Any], error_msg: str) -> Optional[Dict[str, Any]]:
        if context.get("lammps_input_regenerated_after_pair_coeff_error"):
            return None

        required = ("work_dir", "mof", "property")
        missing = [key for key in required if not context.get(key)]
        if missing:
            return {
                "status": "skipped",
                "method": "regenerate_lammps_inputs",
                "reason": f"missing context keys: {', '.join(missing)}",
                "rationale": [
                    {
                        "action": "skipped LAMMPS input regeneration",
                        "basis": (
                            "The preferred recovery for missing pair coefficients is to rerun force-field assignment, "
                            "but that requires the original MOF/property context."
                        ),
                        "evidence": {"missing_context_keys": missing},
                        "risk": "low",
                    }
                ],
            }

        proposed = (
            "LAMMPS reported missing pair coefficients. The safest first recovery is to rerun "
            "the LAMMPS input generation / force-field assignment stage, then retry the job. "
            "Only if this still fails will the error agent fill LJ parameters from atom types/elements."
        )
        if INTERACTION_MODE == "interactive":
            action, _ = ask_user_confirmation("LAMMPSErrorAgent", proposed)
            if action == "skip":
                return {
                    "status": "skipped",
                    "method": "regenerate_lammps_inputs",
                    "reason": "skipped by user in interactive mode",
                    "rationale": [
                        {
                            "action": "skipped LAMMPS input regeneration",
                            "basis": (
                                "In interactive mode, potentially consequential force-field regeneration is gated "
                                "by user confirmation."
                            ),
                            "evidence": {"user_action": "skip"},
                            "risk": "low",
                        }
                    ],
                }
        else:
            print(f"[LAMMPSErrorAgent] {proposed}")

        try:
            from input.lammps_input import LAMMPSInputAgent

            input_agent = LAMMPSInputAgent(llm=None)
            result = input_agent._run_generate_lammps_inputs(
                working_dir=context["work_dir"],
                mof_name=context["mof"],
                guest_name=context.get("guest"),
                prop=context["property"],
                query_text=context.get("query_text", ""),
                num_guest=context.get("num_guest", 1),
                job_name=context.get("job_name", ""),
                simulation_input=context.get("simulation_input") or {"present": False, "snippets": []},
                charge_method=context.get("charge_method", "auto"),
                context=context,
            )
        except Exception as exc:
            context["lammps_input_regenerated_after_pair_coeff_error"] = True
            return {
                "status": "failed",
                "method": "regenerate_lammps_inputs",
                "reason": str(exc),
                "rationale": [
                    {
                        "action": "attempted LAMMPS input regeneration / force-field reassignment",
                        "basis": (
                            "Missing pair coefficients are best fixed by returning to the input-generation stage so "
                            "the force-field assignment can be rebuilt consistently before any fallback parameters are used."
                        ),
                        "evidence": {"exception": str(exc), "trigger_error": error_msg[:1000]},
                        "risk": "low",
                    }
                ],
            }

        context["lammps_input_regenerated_after_pair_coeff_error"] = True
        results = context.setdefault("results", {})
        results["lammps_input_regeneration_after_error"] = {
            "returncode": result.returncode,
            "stdout": result.stdout[-4000:] if result.stdout else "",
            "stderr": result.stderr[-4000:] if result.stderr else "",
            "trigger_error": error_msg[:1000],
        }

        if result.returncode == 0:
            return {
                "status": "patched",
                "method": "regenerate_lammps_inputs",
                "actions": ["reran LAMMPS input generation / force-field assignment"],
                "rationale": [
                    {
                        "action": "reran LAMMPS input generation / force-field assignment",
                        "basis": (
                            "The physically safest response to missing pair coefficients is to regenerate the LAMMPS "
                            "input from the force-field assignment pipeline, preserving internally consistent atom types, "
                            "masses, charges, pair styles, and coefficients."
                        ),
                        "evidence": {
                            "error_pattern": "All pair coeffs are not set",
                            "returncode": result.returncode,
                        },
                        "risk": "low",
                    }
                ],
                "files_changed": sorted(
                    f.name for f in Path(context["work_dir"]).glob("system.in*")
                ) + (["system.data"] if (Path(context["work_dir"]) / "system.data").exists() else []),
            }

        return {
            "status": "failed",
            "method": "regenerate_lammps_inputs",
            "reason": "LAMMPS input regeneration returned nonzero",
            "returncode": result.returncode,
            "rationale": [
                {
                    "action": "attempted LAMMPS input regeneration / force-field reassignment",
                    "basis": (
                        "The agent tried the physically preferred repair path first. Because regeneration failed, "
                        "the caller may either stop or proceed to the explicit element-based LJ fallback."
                    ),
                    "evidence": {
                        "error_pattern": "All pair coeffs are not set",
                        "returncode": result.returncode,
                    },
                    "risk": "medium",
                }
            ],
        }

    def _pair_coeff_self_terms(self, work_dir: Path) -> Dict[int, Dict[str, float]]:
        coeffs: Dict[int, Dict[str, float]] = {}
        for path in (work_dir / "system.in", work_dir / "system.in.settings"):
            if not path.exists():
                continue
            for raw in path.read_text(errors="ignore").splitlines():
                line = raw.split("#", 1)[0].strip()
                parts = line.split()
                if len(parts) < 5 or parts[0].lower() != "pair_coeff":
                    continue
                if not parts[1].isdigit() or parts[1] != parts[2]:
                    continue
                try:
                    coeffs[int(parts[1])] = {
                        "epsilon": float(parts[3]),
                        "sigma": float(parts[4]),
                    }
                except ValueError:
                    continue
        return coeffs

    def _infer_element_from_lj(self, epsilon: float, sigma: float) -> Optional[str]:
        candidates = []
        for element, ff in self._forcefield_lj_by_element().items():
            ref_epsilon = float(ff.get("epsilon") or 0.0)
            ref_sigma = float(ff.get("sigma") or 0.0)
            if ref_epsilon <= 0.0 or ref_sigma <= 0.0:
                continue
            eps_rel = abs(epsilon - ref_epsilon) / ref_epsilon
            sig_rel = abs(sigma - ref_sigma) / ref_sigma
            if eps_rel <= 0.10 and sig_rel <= 0.05:
                candidates.append((eps_rel + sig_rel, element))
        if not candidates:
            return None
        return min(candidates)[1]

    def _ensure_file_backed_masses(self, work_dir: Path) -> Tuple[bool, Dict[str, Any]]:
        data_path = work_dir / "system.data"
        if not data_path.exists():
            return False, {"reason": "system.data not found"}
        text = data_path.read_text(errors="ignore")
        m = re.search(r"(?m)^\s*(\d+)\s+atom\s+types\b", text)
        if not m:
            return False, {"reason": "could not determine declared atom type count"}
        n_types = int(m.group(1))

        ff_by_element = self._forcefield_lj_by_element()
        self_coeffs = self._pair_coeff_self_terms(work_dir)
        lines = text.splitlines()
        existing: Dict[int, Dict[str, Any]] = {}
        in_masses = False
        masses_start = None
        masses_end = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if re.match(r"^Masses\b", stripped, flags=re.IGNORECASE):
                in_masses = True
                masses_start = i
                continue
            if in_masses and stripped and re.match(r"^[A-Za-z]", stripped):
                masses_end = i
                break
            if not in_masses:
                continue
            body, _, comment = line.partition("#")
            parts = body.split()
            if len(parts) < 2 or not parts[0].isdigit():
                continue
            try:
                atom_type = int(parts[0])
                mass = float(parts[1])
            except ValueError:
                continue
            existing[atom_type] = {
                "line_index": i,
                "mass": mass,
                "label": comment.strip(),
            }
        if in_masses and masses_end is None:
            masses_end = len(lines)

        assignments: Dict[int, Dict[str, Any]] = {}
        unresolved = []
        for atom_type in range(1, n_types + 1):
            current = existing.get(atom_type)
            if current and float(current.get("mass") or 0.0) > 0.0:
                continue

            element = None
            evidence = {}
            if current and current.get("label"):
                element = self._infer_element_from_label_or_mass(current["label"], None)
                if element:
                    evidence["inferred_from"] = "Masses comment"
                    evidence["label"] = current["label"]

            if not element and atom_type in self_coeffs:
                coeff = self_coeffs[atom_type]
                element = self._infer_element_from_lj(coeff["epsilon"], coeff["sigma"])
                if element:
                    evidence["inferred_from"] = "self pair_coeff"
                    evidence["pair_coeff"] = coeff

            if not element or element not in ff_by_element or "mass" not in ff_by_element[element]:
                unresolved.append(atom_type)
                continue

            ff = ff_by_element[element]
            assignments[atom_type] = {
                "element": element,
                "mass": float(ff["mass"]),
                "source": ff.get("source"),
                "pseudo_atoms_path": ff.get("pseudo_atoms_path"),
                "pseudo_atoms_line": ff.get("pseudo_atoms_line"),
                **evidence,
            }

        if unresolved:
            return False, {
                "reason": "safe automatic mass correction is not possible",
                "unresolved_atom_types": unresolved,
                "available_self_pair_coeffs": self_coeffs,
            }
        if not assignments:
            return False, {"reason": "all Masses entries are already set"}

        if masses_start is not None:
            for atom_type, item in sorted(assignments.items(), reverse=True):
                entry = (
                    f"{atom_type} {item['mass']:.6g} # {item['element']} "
                    f"mass_from_{item.get('source') or 'forcefield_file'}"
                )
                if atom_type in existing:
                    lines[existing[atom_type]["line_index"]] = entry
                else:
                    lines.insert(masses_end, entry)
            data_path.write_text("\n".join(lines).rstrip() + "\n")
            return True, {
                "source": "file_backed_forcefield_mass",
                "assignments": assignments,
                "files_changed": ["system.data"],
            }

        masses = ["Masses", ""]
        for atom_type in range(1, n_types + 1):
            item = assignments[atom_type]
            masses.append(
                f"{atom_type} {item['mass']:.6g} # {item['element']} "
                f"mass_from_{item.get('source') or 'forcefield_file'}"
            )
        masses.append("")
        atom_match = re.search(r"(?m)^Atoms\b", text)
        if atom_match:
            text = text[:atom_match.start()] + "\n".join(masses) + "\n" + text[atom_match.start():]
        else:
            text = text.rstrip() + "\n\n" + "\n".join(masses) + "\n"
        data_path.write_text(text)
        return True, {
            "source": "file_backed_forcefield_mass",
            "assignments": assignments,
            "files_changed": ["system.data"],
        }

    def apply_known_fix_once(
        self,
        error_msg: str,
        work_dir: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        context = context or {}
        wd = Path(work_dir)
        system_in = wd / "system.in"
        data_path = wd / "system.data"
        changed_files = []
        actions = []
        rationales = []
        err = error_msg or ""

        if re.search(r"All pair coeffs are not set", err, flags=re.IGNORECASE):
            regen = self._regenerate_lammps_inputs_from_context(context, err)
            if regen and regen.get("status") == "patched":
                return regen
            if regen and "skipped by user" in str(regen.get("reason", "")):
                return regen

            ok, details = self._write_uff_like_pair_coeffs_from_atom_types(wd)
            if ok:
                changed_files.append("system.in.settings")
                actions.append("added UFF-like LJ pair_coeff table inferred from atom type labels/elements")
                rationales.append(
                    {
                        "action": "added UFF-like LJ pair_coeff table inferred from atom type labels/elements",
                        "basis": (
                            "LAMMPS requires all pair coefficients before a run can start. "
                            "The fallback uses atom-type labels/masses to infer elements, assigns UFF-like "
                            "element Lennard-Jones parameters, and uses Lorentz-Berthelot mixing for cross terms."
                        ),
                        "evidence": {
                            "error_pattern": "All pair coeffs are not set",
                            "parameter_source": details.get("source"),
                            "mixing_rule": details.get("mixing_rule"),
                            "atom_types": details.get("atom_types"),
                        },
                        "risk": "medium",
                    }
                )
                context.setdefault("results", {})["lammps_pair_coeff_fallback"] = {
                    "status": "patched",
                    "fallback_after_regeneration": bool(context.get("lammps_input_regenerated_after_pair_coeff_error")),
                    "regeneration_result": regen,
                    **details,
                }
            elif regen:
                return regen

        elif re.search(r"(All masses are not set|Not all per-type masses are set)", err, flags=re.IGNORECASE):
            ok, details = self._ensure_file_backed_masses(wd)
            if ok:
                changed_files.append("system.data")
                actions.append("added file-backed Masses entries inferred from force-field evidence")
                rationales.append(
                    {
                        "action": "added file-backed Masses entries inferred from force-field evidence",
                        "basis": (
                            "LAMMPS cannot initialize atom types without a Masses section. "
                            "This recovery infers each missing atom type from Masses comments or existing self "
                            "pair_coeff values, then reads the corresponding mass from the local force-field file. "
                            "It refuses to patch unresolved atom types instead of inserting placeholder masses."
                        ),
                        "evidence": {
                            "error_pattern": "Not all per-type masses are set",
                            **details,
                        },
                        "risk": "medium",
                    }
                )
                context.setdefault("results", {})["lammps_mass_recovery"] = {
                    "status": "patched",
                    **details,
                }
            else:
                context.setdefault("results", {})["lammps_mass_recovery"] = {
                    "status": "not_patched",
                    **details,
                }

        elif re.search(r"Unrecognized pair style", err, flags=re.IGNORECASE):
            if self._replace_first_matching_line(
                system_in,
                r"^\s*pair_style\s+\S+",
                "pair_style lj/cut 10.0",
            ):
                changed_files.append("system.in")
                actions.append("replaced unrecognized pair_style with lj/cut")
                rationales.append(
                    {
                        "action": "replaced unrecognized pair_style with lj/cut",
                        "basis": (
                            "The requested pair style is not available in the current LAMMPS build. "
                            "For non-bonded Lennard-Jones-only fixtures, lj/cut is the minimal compatible pair style."
                        ),
                        "evidence": {"error_pattern": "Unrecognized pair style"},
                        "risk": "medium",
                    }
                )
            ok, details = self._write_uff_like_pair_coeffs_from_atom_types(wd)
            if ok:
                changed_files.append("system.in.settings")
                actions.append("added UFF-like LJ pair_coeff table inferred from atom type labels/elements")
                rationales.append(
                    {
                        "action": "added UFF-like LJ pair_coeff table inferred from atom type labels/elements",
                        "basis": (
                            "After replacing the unavailable pair style with lj/cut, explicit LJ coefficients "
                            "are still required. The fallback infers element types from system.data and applies "
                            "UFF-like Lennard-Jones parameters with Lorentz-Berthelot mixing."
                        ),
                        "evidence": {
                            "error_pattern": "Unrecognized pair style",
                            "parameter_source": details.get("source"),
                            "mixing_rule": details.get("mixing_rule"),
                            "atom_types": details.get("atom_types"),
                        },
                        "risk": "medium",
                    }
                )
                context.setdefault("results", {})["lammps_pair_coeff_fallback"] = {
                    "status": "patched",
                    "fallback_after_regeneration": False,
                    **details,
                }

        elif re.search(r"Attempting to rescale a 0\.0 temperature", err, flags=re.IGNORECASE):
            removed = self._remove_lines_matching(system_in, r"^\s*velocity\s+\S+\s+create\b")
            if removed:
                changed_files.append("system.in")
                actions.append("removed velocity create command for zero-temperature group")
                rationales.append(
                    {
                        "action": "removed velocity create command for zero-temperature group",
                        "basis": (
                            "LAMMPS cannot rescale or initialize a group with zero instantaneous temperature. "
                            "Removing the velocity creation command avoids applying a thermostat initialization "
                            "to a group with no thermal degrees of freedom."
                        ),
                        "evidence": {"error_pattern": "Attempting to rescale a 0.0 temperature"},
                        "risk": "low",
                    }
                )

        elif re.search(
            r"(KSpace style requires atom attribute q|Pair style requires a KSpace style|Must use kspace_style|coul/long|pppm)",
            err,
            flags=re.IGNORECASE,
        ):
            patched = self._patch_charge_kspace_protocol(wd)
            if patched:
                changed_files.extend(patched.get("files_changed", []))
                actions.append("ensured charge-aware pair_style and kspace_style after read_data")
                rationales.append(
                    {
                        "action": "ensured charge-aware pair_style and kspace_style after read_data",
                        "basis": (
                            "Long-range Coulomb pair styles require a compatible kspace solver. "
                            "When system.data contains nonzero charges or the input uses coul/long, "
                            "pppm with a conservative precision is inserted after read_data."
                        ),
                        "evidence": {
                            "error_pattern": "Pair style requires a KSpace style / kspace-related LAMMPS error",
                            "charged_system_data": patched.get("charged_system_data"),
                            "inserted": ["pair_style lj/cut/coul/long 10.0", "kspace_style pppm 1.0e-4"],
                        },
                        "risk": "medium",
                    }
                )
                context.setdefault("results", {})["lammps_charge_kspace_recovery"] = patched

        elif re.search(
            r"(Non-numeric|Lost atoms|Bond atoms missing|Angle atoms missing|Dihedral atoms missing|Out of range atoms)",
            err,
            flags=re.IGNORECASE,
        ):
            if context.get("lammps_stability_protocol_applied"):
                return self._request_lammps_structure_regeneration(
                    context,
                    "LAMMPS remained unstable after applying the conservative timestep/minimize protocol",
                )
            patched = self._apply_lammps_stability_protocol(wd)
            if patched:
                context["lammps_stability_protocol_applied"] = True
                changed_files.extend(patched.get("files_changed", []))
                actions.append("applied conservative stability protocol for non-numeric/unstable dynamics")
                rationales.append(
                    {
                        "action": "applied conservative stability protocol for non-numeric/unstable dynamics",
                        "basis": (
                            "Lost atoms, non-numeric coordinates, and missing bonded atoms usually indicate an unstable "
                            "initial geometry or an integration step that is too aggressive. The first response reduces "
                            "the timestep, enforces neighbor-list updates, and minimizes before dynamics; if the failure "
                            "recurs, the flow escalates to structure regeneration."
                        ),
                        "evidence": {
                            "error_pattern": "Non-numeric / Lost atoms / bonded atoms missing / out-of-range atoms",
                            "protocol": patched,
                        },
                        "risk": "low_to_medium",
                    }
                )
                context.setdefault("results", {})["lammps_stability_recovery"] = patched

        if not changed_files:
            return None

        return {
            "status": "patched",
            "method": "rule_based",
            "actions": actions,
            "rationale": rationales,
            "files_changed": sorted(set(changed_files)),
        }

    def pre_run_review(self, context: dict) -> dict:
        work_dir = context.get("work_dir")
        if not work_dir:
            return context

        overlap = self._structure_overlap_precheck(context)
        if overlap.get("status") == "needs_structure_regeneration":
            record_job_event(
                context,
                "blocked",
                message="LAMMPS pre-run structure overlap detected",
                metadata=overlap,
            )
            return context

        abs_files = [os.path.join(work_dir, f) for f in self.input_files]
        file_dict = {f: self.read_file(f) for f in abs_files if os.path.exists(f)}
        if not file_dict:
            return context

        mof   = context.get("mof", "unknown MOF")
        guest = context.get("guest", "unknown guest")

        system_prompt = (
            "You are a LAMMPS input file reviewer for MOF simulations. This simulation uses LAMMPS (3 Mar 2020).\n"
            "Check the provided LAMMPS input files for physics-level errors BEFORE running.\n"
            "Focus on: missing electrostatics (kspace/coul) for polar guests, "
            "wrong pair_style for the guest molecule, missing rigid constraints for linear molecules, "
            "MSD computed per-atom instead of per-molecule COM, missing bond/angle coefficients.\n"
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
            "Only propose a fix if you are certain the replacement is syntactically correct "
            "for LAMMPS (3 Mar 2020) and will not introduce a new error. "
            "If you are not fully confident in the exact corrected syntax, omit that fix entirely.\n"
            "The ACTION line must start with one of: Replace:, After the line:, Before the line:, "
            "Remove the line:, Append at end:, Overwrite entire file with:"
        )

        user_prompt = f"MOF: {mof}\nGuest: {guest}\n\n"
        for fname, content in file_dict.items():
            user_prompt += f"\n----- {os.path.basename(fname)} -----\n{content}\n"

        print(f"\n[LAMMPSErrorAgent] Pre-run review for {mof} / {guest} ...")
        response = self._invoke_llm(system_prompt, user_prompt,
                                    agent="LAMMPSErrorAgent", label="pre_run_review")

        if response.strip().upper() == "OK":
            print("[LAMMPSErrorAgent] Pre-run review: no issues found.")
            return context

        print("[LAMMPSErrorAgent] Pre-run review found issues. Proposed fixes:\n")
        print(response)

        action, response = self._ask_user_confirmation(
            "LAMMPSErrorAgent", response, system_prompt, user_prompt
        )
        if action == "skip":
            return context

        for block in response.split("----"):
            if not block.strip():
                continue
            if "FILE:" in block:
                fname_rel = block.split("FILE:")[1].split("\n")[0].strip()
                full_path = os.path.join(work_dir, fname_rel)
                self.patch_file(full_path, block)

        print("[LAMMPSErrorAgent] Pre-run fixes applied.")
        return context

    def run(self, context: dict):
        if context.get("lammps_status") == "needs_structure_from_user":
            context["lammps_success"] = False
            context.setdefault("results", {})[
                "lammps_error_status"
            ] = "not_invoked_missing_structure"
            return context

        work_dir = context.get("work_dir")
        if not work_dir:
            raise RuntimeError("LammpsErrorAgent.run: context['work_dir'] is missing.")

        print(f"\n=== LammpsErrorAgent: error loop in {work_dir} ===")

        base_files = self.input_files or ["system.in", "system.in.settings", "system.in.init"]
        abs_files = [os.path.join(work_dir, f) for f in base_files]

        log_path = os.path.join(work_dir, self.log_file)

        max_trials = 5
        success = False
        poll_interval = 60

        first_already_submitted = bool(context.get("lammps_submitted", False))
        record_job_event(
            context,
            "polling",
            message="LAMMPS marker polling started",
            metadata={"poll_interval_sec": poll_interval},
        )

        for attempt in range(1, max_trials + 1):
            context["retry_count"] = max(0, attempt - 1)
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

            while True:
                time.sleep(poll_interval)
                waited += poll_interval

                done_path = os.path.join(work_dir, "DONE")
                failed_path = os.path.join(work_dir, "FAILED")

                if os.path.exists(done_path):
                    print("\nDONE detected. LAMMPS finished successfully.")
                    success = True
                    err = ""
                    finished = True
                    record_job_event(context, "done_ok", message="LAMMPS DONE marker detected")
                    break

                if os.path.exists(failed_path):
                    print("\nFAILED detected. LAMMPS failed.")
                    success = False

                    
                    err = self.extract_error(log_path, n=80)
                    if not err:
                        err = self.read_file(log_path)

                    finished = True
                    record_job_event(
                        context,
                        "failed",
                        message="LAMMPS FAILED marker detected",
                        last_error=err[:4000],
                    )
                    break

                record_scheduler_status(context)
                print(f"[poll] waiting for DONE/FAILED... ({waited}s)")

            if success:
                break

            
            print(f"\nLAMMPS ERROR detected on attempt #{attempt}:\n{err}\n")

            file_dict = {f: self.read_file(f) for f in abs_files}
            rag_hits = self._retrieve_error_knowledge_hits(err, file_dict)
            rag_evidence = self._format_error_knowledge(rag_hits)
            if rag_evidence:
                print("\n[LAMMPS ERROR KNOWLEDGE]\n", rag_evidence[:3000])
                context.setdefault("results", {})["lammps_error_knowledge_hits"] = rag_hits

            context.setdefault("results", {})["lammps_error_recovery_mode"] = "llm_rag"

            recovery = self.decide_runtime_recovery(
                err,
                file_dict,
                context=context,
                rag_evidence=rag_evidence,
            )
            fix = recovery["response"]
            context.setdefault("results", {})["lammps_llm_recovery"] = recovery
            print("\nLLM SUGGESTION:\n", fix)

            if (
                recovery.get("decision") == "TOOL_ACTION"
                and recovery.get("tool_action") == "REQUEST_USER_MODEL_SELECTION"
            ):
                context["lammps_status"] = "needs_model_selection_from_user"
                context["lammps_model_selection_request"] = {
                    "status": "needs_user_model_selection",
                    "mof": context.get("mof"),
                    "guest": context.get("guest"),
                    "property": context.get("property"),
                    "reason": self._response_field(fix, "RATIONALE") or "",
                    "evidence": self._response_field(fix, "EVIDENCE") or "",
                }
                context.setdefault("results", {})[
                    "lammps_model_selection_request"
                ] = context["lammps_model_selection_request"]
                record_job_event(
                    context,
                    "blocked",
                    message="LAMMPS recovery needs force-field/model selection",
                    metadata=context["lammps_model_selection_request"],
                    last_error=err[:4000],
                )
                break

            if (
                recovery.get("decision") == "TOOL_ACTION"
                and recovery.get("tool_action") == "REGENERATE_LAMMPS_INPUTS"
            ):
                regeneration = self._regenerate_lammps_inputs_from_context(context, err)
                context.setdefault("results", {})[
                    "lammps_runtime_input_regeneration"
                ] = regeneration
                if not regeneration or regeneration.get("status") != "patched":
                    context["lammps_status"] = "needs_model_selection_from_user"
                    record_job_event(
                        context,
                        "blocked",
                        message="LAMMPS input/force-field regeneration could not complete safely",
                        metadata=regeneration or {},
                        last_error=err[:4000],
                    )
                    break
                first_already_submitted = False
                continue

            for block in fix.split("----"):
                if not block.strip():
                    continue
                if "FILE:" in block:
                    fname_rel = block.split("FILE:")[1].split("\n")[0].strip()
                    full_path = os.path.join(work_dir, fname_rel)
                    self.patch_file(full_path, block)

            print("\nAuto-patch applied. Proceeding to next attempt.")
            record_job_event(
                context,
                "retrying",
                message="LAMMPS auto-patch applied; retry scheduled",
                metadata={"attempt": attempt},
                last_error=err[:4000],
            )

            first_already_submitted = False

        else:
            print("\nMaximum attempts reached. Manual intervention required.")
            record_job_event(context, "giveup", message="LAMMPS maximum attempts reached")

        context["lammps_success"] = success
        if not success:
            record_job_event(context, "giveup", message="LAMMPS did not finish successfully")
        return context
