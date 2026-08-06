import os
import shutil
import json
import re

from pathlib import Path
from typing import Dict, Any, Optional

from config import LLM_DEFAULT
from langchain.schema import HumanMessage, SystemMessage
from core.timing import timed_call
from core.llm_logging import log_llm_decision
from input.interactive_review import maybe_interactive_review_input_file

from file.agent import VASPFileAgent
from core.resource_allocator import ResourceAllocator
from .vasp.prompt import (
    create_vasp_incar_prompt,
    get_vasp_system_message,
    render_vasp_format,
)


_QMOF_D_BLOCK_ATOMIC_NUMBERS = frozenset(
    atomic_number
    for first, last in ((21, 29), (39, 47), (71, 79), (103, 111))
    for atomic_number in range(first, last + 1)
)
_QMOF_F_BLOCK_ATOMIC_NUMBERS = frozenset(
    atomic_number
    for first, last in ((57, 70), (89, 102))
    for atomic_number in range(first, last + 1)
)


def _run_length_encode(values):
    runs = []
    for value in values:
        if runs and runs[-1][0] == value:
            runs[-1][1] += 1
        else:
            runs.append([value, 1])
    return runs


def _compact_vasp_magmom(moments) -> str:
    tokens = []
    for value, count in _run_length_encode(moments):
        rendered = f"{float(value):.1f}"
        tokens.append(f"{count}*{rendered}" if count > 1 else rendered)
    return " ".join(tokens)


def build_qmof_high_spin_initialization(poscar_path: str) -> Dict[str, Any]:
    import ase.io

    atoms = ase.io.read(poscar_path, format="vasp")
    symbols = atoms.get_chemical_symbols()
    atomic_numbers = atoms.get_atomic_numbers()

    moments = []
    magnetic_elements = {}
    for symbol, atomic_number in zip(symbols, atomic_numbers):
        atomic_number = int(atomic_number)
        if atomic_number in _QMOF_D_BLOCK_ATOMIC_NUMBERS:
            block, moment = "d", 5.0
        elif atomic_number in _QMOF_F_BLOCK_ATOMIC_NUMBERS:
            block, moment = "f", 7.0
        else:
            block, moment = None, 0.0
        moments.append(moment)
        if block:
            entry = magnetic_elements.setdefault(
                symbol,
                {"element": symbol, "block": block, "count": 0, "initial_magmom": moment},
            )
            entry["count"] += 1

    element_runs = [
        {"element": symbol, "count": count}
        for symbol, count in _run_length_encode(symbols)
    ]
    applicable = bool(magnetic_elements)
    return {
        "policy": "QMOF-style high-spin initial guess",
        "applicable": applicable,
        "reason": (
            "POSCAR contains at least one QMOF d-block or f-block atom"
            if applicable
            else "POSCAR contains no QMOF d-block or f-block atoms"
        ),
        "atom_order": "POSCAR",
        "atom_count": len(atoms),
        "poscar_element_runs": element_runs,
        "element_scope": {
            "d_block": "Sc-Cu, Y-Ag, Lu-Au, Lr-Rg (group 12 excluded)",
            "f_block": "La-Yb, Ac-No",
        },
        "per_atom_rule": {"d_block": 5.0, "f_block": 7.0, "other": 0.0},
        "magnetic_elements": list(magnetic_elements.values()),
        "incar_settings": {
            "ISPIN": 2,
            "MAGMOM": _compact_vasp_magmom(moments),
        } if applicable else {},
    }


def _resolve_qmof_high_spin_initialization(
    profile: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    if not profile.get("applicable"):
        return profile

    overrides = context.get("incar_overrides") or {}
    normalized_overrides = {str(key).upper(): value for key, value in overrides.items()}
    if "MAGMOM" in normalized_overrides:
        profile["applicable"] = False
        profile["reason"] = "explicit user INCAR override supplies MAGMOM"
        return profile
    if "ISPIN" in normalized_overrides and str(normalized_overrides["ISPIN"]).strip() != "2":
        profile["applicable"] = False
        profile["reason"] = "explicit user INCAR override supplies a non-default ISPIN"
        return profile

    instruction_text = "\n".join(
        str(context.get(key) or "")
        for key in ("query_text", "method_paragraph")
    )
    if re.search(r"\bMAGMOM\b", instruction_text, flags=re.IGNORECASE):
        profile["applicable"] = False
        profile["reason"] = "explicit user or method text supplies MAGMOM"
        return profile

    text_without_ispin2 = re.sub(
        r"\bISPIN\s*=\s*2\b",
        "",
        instruction_text,
        flags=re.IGNORECASE,
    )
    conflicting_spin_pattern = re.compile(
        r"\b(?:ISPIN|low[- ]?spin|antiferromagnet(?:ic|ism)?|AFM|"
        r"ferrimagnet(?:ic|ism)?|non[- ]?collinear|spin[- ]?orbit|LSORBIT|"
        r"LNONCOLLINEAR|nonmagnetic|non[- ]?magnetic|spin[- ]?unpolarized|"
        r"non[- ]?spin[- ]?polarized|closed[- ]?shell)\b",
        flags=re.IGNORECASE,
    )
    if conflicting_spin_pattern.search(text_without_ispin2):
        profile["applicable"] = False
        profile["reason"] = "explicit user or method text requests a different spin treatment"
    return profile


def _pick_snippet(simulation_input: dict, software: str) -> str:
    if not simulation_input:
        return ""
    for s in (simulation_input.get("snippets") or []):
        if (s.get("software") == software) and (s.get("text") or "").strip():
            return s["text"].strip()
    return ""

VASP_REPRO_PATCH_SYSTEM = """You are a careful text editor for VASP INCAR files.
Return ONLY the patched INCAR text. No markdown. No explanations."""

VASP_REPRO_PATCH_USER = """Patch the original VASP INCAR by applying ONLY the required replacements below.

HARD RULES:
1) MINIMAL CHANGE: do not alter any lines except where needed to apply REQUIRED REPLACEMENTS.
2) Preserve all other tags exactly as-is (ENCUT, ISMEAR, SIGMA, NSW, EDIFF, LREAL, etc).
3) If a required key is missing, insert it in a reasonable place (end is fine).
4) Output MUST be a valid INCAR.

REQUIRED REPLACEMENTS (JSON):
{replacements_json}

ORIGINAL INCAR:
<<<{original_text}>>>
"""

class VASPInputAgent:

    def __init__(self, llm=None):
        self.llm = llm or LLM_DEFAULT
        self.file_agent = VASPFileAgent  

    

    def _format_incar_value(self, v: Any) -> str:
        if isinstance(v, bool):
            return ".TRUE." if v else ".FALSE."
        return str(v)

    

    def _enforce_guest_incar_settings(self, incar_text: str) -> str:
        import re

        if re.search(r'^\s*LREAL\s*=', incar_text, re.MULTILINE):
            incar_text = re.sub(r'^\s*LREAL\s*=.*$', 'LREAL = .FALSE.', incar_text, flags=re.MULTILINE)
        else:
            incar_text += '\nLREAL = .FALSE.'

        if re.search(r'^\s*ISYM\s*=', incar_text, re.MULTILINE):
            incar_text = re.sub(r'^\s*ISYM\s*=.*$', 'ISYM = 0', incar_text, flags=re.MULTILINE)
        else:
            incar_text += '\nISYM = 0'

        return incar_text

    @staticmethod
    def _upsert_incar_settings(
        incar_text: str,
        settings: Dict[str, Any],
    ) -> str:
        lines = incar_text.splitlines()
        for key, value in settings.items():
            rendered = f"{key} = {value}"
            pattern = re.compile(rf"^\s*{re.escape(key)}\s*=", re.IGNORECASE)
            for index, line in enumerate(lines):
                if pattern.search(line):
                    lines[index] = rendered
                    break
            else:
                lines.append(rendered)
        return "\n".join(lines).strip()

    def _enforce_projected_dos_incar_settings(
        self,
        incar_text: str,
        *,
        has_chgcar: bool,
        has_wavecar: bool,
        fft_grid: Optional[Dict[str, int]] = None,
    ) -> str:
        settings = {
            "IBRION": "-1",
            "NSW": "0",
            "LORBIT": "11",
            "NEDOS": "2001",
            "ISMEAR": "0",
            "SIGMA": "0.05",
            "LCHARG": ".TRUE.",
            "LWAVE": ".TRUE.",
            "ICHARG": "11" if has_chgcar else "2",
            "ISTART": "1" if has_wavecar else "0",
        }
        for key, value in (fft_grid or {}).items():
            normalized_key = str(key).upper()
            if normalized_key in {"NGX", "NGY", "NGZ", "NGXF", "NGYF", "NGZF"}:
                settings[normalized_key] = str(int(value))
        return self._upsert_incar_settings(incar_text, settings)

    def _enforce_qmof_high_spin_incar_settings(
        self,
        incar_text: str,
        initialization: Dict[str, Any],
    ) -> str:
        if not initialization.get("applicable"):
            return incar_text
        return self._upsert_incar_settings(
            incar_text,
            initialization["incar_settings"],
        )

    def _llm_patch_incar(self, original_text: str, replacements: Dict[str, Any]) -> str:
        rep_json = json.dumps(replacements, ensure_ascii=False, indent=2)

        from core.llm_logging import set_llm_context
        set_llm_context("VASPInputAgent", "incar_patch")
        resp = self.llm.invoke([
            SystemMessage(content=VASP_REPRO_PATCH_SYSTEM),
            HumanMessage(content=VASP_REPRO_PATCH_USER.format(
                replacements_json=rep_json,
                original_text=original_text
            )),
        ])
        out = (resp.content or "").strip()

        if out.startswith("```"):
            lines = out.splitlines()
            out = "\n".join(lines[1:-1]).strip()

        if not out:
            raise ValueError("LLM returned empty patched INCAR.")
        return out

    def _write_incar(self, system_label, system_role, target_dir, context,
                     reproduce_mode: bool = False,
                     example_incar_text: str = ""):
        os.makedirs(target_dir, exist_ok=True)

        prop = context.get("property", "binding_energy")
        vasp_stage = context.get("vasp_stage", "")
        vasp_calc_type = context.get("vasp_calc_type", "")

        dos_has_chgcar = bool(context.get("dos_has_chgcar", False))
        icharg = 11 if dos_has_chgcar else 2

        
        
        replacements = {
            "SYSTEM": str(system_label),
        }
        
        if (vasp_stage == "dos") or (vasp_calc_type == "dos"):
            replacements["ICHARG"] = str(icharg)

        
        if reproduce_mode and example_incar_text:
            incar_text = self._llm_patch_incar(example_incar_text, replacements)
            if vasp_stage == "projected_dos" or vasp_calc_type == "projected_dos":
                incar_text = self._enforce_projected_dos_incar_settings(
                    incar_text,
                    has_chgcar=bool(context.get("projected_dos_has_chgcar")),
                    has_wavecar=bool(context.get("projected_dos_has_wavecar")),
                    fft_grid=context.get("projected_dos_fft_grid"),
                )

            incar_path = os.path.join(target_dir, "INCAR")
            with open(incar_path, "w") as f:
                f.write(incar_text.rstrip() + "\n")
            maybe_interactive_review_input_file(
                software="VASP",
                path=incar_path,
                context=context,
                llm=self.llm,
                label="VASPInputAgent",
            )
            return incar_path

        
        method_paragraph = context.get("method_paragraph")

        query = {
            "job_name": context.get("job_name"),
            "user_request": context.get("query_text"),
            "property": prop,
            "mof": context.get("mof"),
            "guest": context.get("guest"),
            "system_label": system_label,
            "system_role": system_role,
            "vasp_stage": vasp_stage,
            "vasp_calc_type": vasp_calc_type,
            "dos_has_chgcar": dos_has_chgcar,
            "recommended_icharg": icharg,
            "incar_overrides": context.get("incar_overrides") or {},
        }

        magnetic_initialization = build_qmof_high_spin_initialization(
            os.path.join(target_dir, "POSCAR")
        )
        magnetic_initialization = _resolve_qmof_high_spin_initialization(
            magnetic_initialization,
            context,
        )
        query["qmof_high_spin_initialization"] = magnetic_initialization

        vasp_format_raw = render_vasp_format(query)
        vasp_format = vasp_format_raw.replace("{system}", str(system_label))

        
        vasp_incar_hints = ""
        vasp_manual_hints = ""
        if not reproduce_mode:
            literature_disabled = (
                os.getenv("SIMMOF_DISABLE_LITERATURE_RAG", "").strip().lower() in {"1", "true", "yes", "on"}
                or os.getenv("SIMMOF_VASP_LITERATURE_RAG", "1").strip().lower() in {"0", "false", "no", "off"}
            )
            if literature_disabled:
                print("[RAG] VASP literature INCAR hints disabled by environment")
            else:
                try:
                    from rag.agent import RagAgent
                    rag_ctx = {
                        "job_name": context.get("job_name"),
                        "mof": context.get("mof"),
                        "guest": context.get("guest"),
                        "property": prop,
                        "query_text": context.get("query_text") or "",
                        "vasp_stage": vasp_stage,
                        "vasp_calc_type": vasp_calc_type,
                    }
                    miner = RagAgent(agent_name="RagAgent")
                    rag_out = timed_call(
                        "RAG.get_vasp_incar_hints",
                        miner.run_for_vasp_incar,
                        rag_ctx,
                        top_files=5,
                        category="vasp_input_internal",
                        context=rag_ctx,
                        extra={"parent_agent": "VASPInputAgent"},
                    )
                    vasp_incar_hints = (rag_out.get("vasp_incar_hints") or "").strip()
                    if vasp_incar_hints:
                        print("[RAG] VASP INCAR hints enabled")
                    else:
                        print("[RAG] no relevant VASP INCAR hints")
                except Exception as e:
                    print(f"[RAG] VASP INCAR hints disabled due to error: {e}")
                    vasp_incar_hints = ""

            manual_disabled = os.getenv("SIMMOF_VASP_MANUAL_RAG", "1").strip().lower() in {"0", "false", "no", "off"}
            if manual_disabled:
                print("[RAG] VASP manual INCAR hints disabled by environment")
            else:
                try:
                    from input.vasp.manual_rag import retrieve_vasp_manual_hints_reranked

                    manual_query = " ".join(
                        str(x)
                        for x in [
                            context.get("query_text") or "",
                            prop,
                            vasp_stage,
                            vasp_calc_type,
                            system_role,
                            "VASP INCAR tag manual",
                        ]
                        if x
                    )
                    manual_out = timed_call(
                        "RAG.get_vasp_manual_incar_hints",
                        retrieve_vasp_manual_hints_reranked,
                        manual_query,
                        llm=self.llm,
                        top_k=6,
                        candidate_k=24,
                        category="vasp_input_internal",
                        context=context,
                        extra={"parent_agent": "VASPInputAgent"},
                    )
                    vasp_manual_hints = (manual_out.get("formatted_hints") or "").strip()
                    if vasp_manual_hints:
                        print("[RAG] VASP manual INCAR hints enabled")
                    else:
                        print("[RAG] no relevant VASP manual INCAR hints")
                    context.setdefault("results", {})["vasp_manual_incar_hints"] = {
                        "query": manual_out.get("query"),
                        "hits": [
                            {
                                "tag": h.get("tag"),
                                "section_number": h.get("section_number"),
                                "section_title": h.get("section_title"),
                            }
                            for h in manual_out.get("hits", [])[:6]
                        ],
                        "all_candidate_tags": [
                            h.get("tag")
                            for h in manual_out.get("all_candidate_hits", [])[:24]
                            if h.get("tag")
                        ],
                        "reranker_selection": manual_out.get("reranker_selection"),
                    }
                except Exception as e:
                    print(f"[RAG] VASP manual INCAR hints disabled due to error: {e}")
                    vasp_manual_hints = ""
        else:
            print("[RAG] skipped (reproduce mode)")

        prompt = create_vasp_incar_prompt(
            query,
            vasp_format,
            method_paragraph,
            rag_hints=vasp_incar_hints,
            manual_hints=vasp_manual_hints,
        )

        messages = [
            SystemMessage(content=get_vasp_system_message()),
            HumanMessage(content=prompt),
        ]
        from core.llm_logging import set_llm_context
        set_llm_context("VASPInputAgent", "incar_generated")
        resp = self.llm.invoke(messages)
        incar_text = (resp.content or "").strip()

        if incar_text.startswith("```"):
            lines = incar_text.splitlines()
            incar_text = "\n".join(lines[1:-1]).strip()

        if system_role == "guest":
            incar_text = self._enforce_guest_incar_settings(incar_text)
        if vasp_stage == "projected_dos" or vasp_calc_type == "projected_dos":
            incar_text = self._enforce_projected_dos_incar_settings(
                incar_text,
                has_chgcar=bool(context.get("projected_dos_has_chgcar")),
                has_wavecar=bool(context.get("projected_dos_has_wavecar")),
                fft_grid=context.get("projected_dos_fft_grid"),
            )
        incar_text = self._enforce_qmof_high_spin_incar_settings(
            incar_text,
            magnetic_initialization,
        )

        incar_path = os.path.join(target_dir, "INCAR")
        with open(incar_path, "w") as f:
            f.write(incar_text.rstrip() + "\n")
        maybe_interactive_review_input_file(
            software="VASP",
            path=incar_path,
            context=context,
            llm=self.llm,
            label="VASPInputAgent",
        )
        try:
            log_llm_decision("VASPInputAgent", "incar_generated",
                             {"system_label": system_label,
                              "system_role": system_role,
                              "incar_preview": incar_text[:1000]},
                             context)
        except Exception:
            pass
        return incar_path



    

    def _prepare_single_system(
        self,
        src_structure_path: str,
        base_dir: str,
        system_label: str,
        system_role: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        os.makedirs(base_dir, exist_ok=True)

        dst_structure_path = os.path.join(
            base_dir,
            os.path.basename(src_structure_path),
        )
        shutil.copy2(src_structure_path, dst_structure_path)

        try:
            import ase.io
            n_atoms = len(ase.io.read(dst_structure_path))
        except Exception:
            n_atoms = 0
        calc_type = context.get("vasp_calc_type") or context.get("vasp_stage") or context.get("property", "")
        spec = ResourceAllocator().recommend("VASP", calc_type, n_atoms, context)
        context["resource_allocation"] = {
            "software": "VASP",
            "calc_type": calc_type,
            "n_atoms": n_atoms,
            "nodes": spec.nodes,
            "ppn": spec.ppn,
            "np": spec.np,
            "queue": spec.queue,
            "rationale": spec.rationale,
        }

        self.file_agent.get_vasp_file(dst_structure_path, spec=spec)

        reproduce_mode = bool(context.get("reproduce_mode", False))
        example_incar_text = (context.get("example_vasp_text") or "").strip()

        incar_path = self._write_incar(
            system_label=system_label,
            system_role=system_role,
            target_dir=base_dir,
            context=context,
            reproduce_mode=reproduce_mode,
            example_incar_text=example_incar_text,
        )

        return {
            "role": system_role,
            "label": Path(dst_structure_path).stem,
            "dir": base_dir,
            "structure_path": dst_structure_path,
            "incar_path": incar_path,
        }

    

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if context.get("vasp_status") == "needs_structure_from_user":
            context.setdefault("results", {})[
                "vasp_input_status"
            ] = "blocked_missing_structure"
            return context

        work_dir = context["work_dir"]
        vasp_root = os.path.join(work_dir, "vasp")
        os.makedirs(vasp_root, exist_ok=True)

        sim_input = context.get("simulation_input") or {}
        example_vasp_text = _pick_snippet(sim_input, "VASP")
        reproduce_mode = bool(example_vasp_text)

        if reproduce_mode:
            print("[VASP] reproduce mode enabled (user INCAR snippet detected)")
        else:
            print("[VASP] standard mode")


        job_id = context.get("job_id", "")
        vasp_stage = context.get("vasp_stage", "")
        vasp_calc_type = context.get("vasp_calc_type", "")

        system_key: Optional[str] = None
        system_role: Optional[str] = None
        src_structure_path: Optional[str] = None
        system_label: Optional[str] = None

        
        if vasp_stage == "projected_dos" or vasp_calc_type == "projected_dos":
            pdos_struct = context.get("projected_dos_structure_path")
            if not pdos_struct:
                raise RuntimeError(
                    "[VASPInputAgent] projected DOS stage requires "
                    "context['projected_dos_structure_path']."
                )
            pdos_role = str(context.get("projected_dos_role") or "system")
            system_key = f"pdos_{pdos_role}"
            system_role = pdos_role
            src_structure_path = pdos_struct
            system_label = f"{Path(pdos_struct).stem}_{pdos_role}_pdos"

        elif vasp_stage == "dos" or vasp_calc_type == "dos":
            dos_struct = context.get("optimized_mof_path") or context.get("mof_path")
            if not dos_struct:
                raise RuntimeError("[VASPInputAgent] DOS stage requires optimized_mof_path or mof_path, but neither is available.")
            system_key = "dos"
            system_role = "mof"
            src_structure_path = dos_struct
            system_label = f"{Path(dos_struct).stem}_dos"

        
        elif vasp_stage in ("bandgap", "band_gap") or vasp_calc_type in ("bandgap", "band_gap"):
            
            bg_struct = context.get("optimized_mof_path") or context.get("mof_path")
            if not bg_struct:
                raise RuntimeError("[VASPInputAgent] Band gap stage requires optimized_mof_path or mof_path, but neither is available.")
            system_key = "bandgap"   
            system_role = "mof"
            src_structure_path = bg_struct
            system_label = f"{Path(bg_struct).stem}_bg"

        
        elif job_id.endswith("_mof") or vasp_stage == "mof_opt":
            if not context.get("mof_path"):
                raise RuntimeError("[VASPInputAgent] MOF job requires context['mof_path'], but it is missing.")
            system_key = "mof"
            system_role = "mof"
            src_structure_path = context["mof_path"]
            system_label = Path(src_structure_path).stem

        
        elif job_id.endswith("_guest") or vasp_stage == "guest":
            if not context.get("guest_cif_path"):
                raise RuntimeError("[VASPInputAgent] Guest job requires context['guest_cif_path'], but it is missing.")
            system_key = "guest"
            system_role = "guest"
            src_structure_path = context["guest_cif_path"]
            system_label = Path(src_structure_path).stem

        
        elif job_id.endswith("_complex") or vasp_stage == "complex":
            complex_one = context.get("complex_cif_path")
            if not complex_one:
                complex_list = context.get("complex_cif_paths") or []
                if not complex_list:
                    raise RuntimeError("[VASPInputAgent] Complex job requires complex_cif_path or complex_cif_paths, but neither is available.")
                complex_one = complex_list[0]
                context["complex_cif_path"] = complex_one

            context["complex_path"] = context.get("complex_path") or context["complex_cif_path"]

            system_key = "complex"
            system_role = "complex"
            src_structure_path = complex_one
            system_label = Path(src_structure_path).stem

        else:
            raise ValueError(
                f"[VASPInputAgent] Unknown job_id pattern: {job_id} "
                f"(stage={vasp_stage}, calc={vasp_calc_type})"
            )

        base_dir = os.path.join(vasp_root, system_key)

        vasp_system = self._prepare_single_system(
            src_structure_path=src_structure_path,
            base_dir=base_dir,
            system_label=system_label,
            system_role=system_role,
            context={**context, "reproduce_mode": reproduce_mode, "example_vasp_text": example_vasp_text},
        )

        vasp_system.setdefault("dir", base_dir)
        vasp_system.setdefault("label", system_label)
        vasp_system.setdefault("role", system_role)

        context["vasp_root"] = vasp_root
        context["vasp_system"] = vasp_system
        context["vasp_dir"] = vasp_system["dir"]
        context["vasp_label"] = vasp_system["label"]
        if vasp_system.get("role"):
            context["vasp_role"] = vasp_system["role"]

        paths = context.get("paths")
        if isinstance(paths, dict):
            paths.setdefault("vasp", {})
            paths["vasp"]["run_dir"] = vasp_system["dir"]

        context.setdefault("results", {})["vasp_input_status"] = "ok"
        return context
