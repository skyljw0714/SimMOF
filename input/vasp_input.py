import os
import shutil
import json

from pathlib import Path
from typing import Dict, Any, Optional

from config import LLM_DEFAULT
from langchain.schema import HumanMessage, SystemMessage

from file.agent import VASPFileAgent
from .vasp.prompt import (
    create_vasp_incar_prompt,
    get_vasp_system_message,
    select_vasp_format,
)


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

    

    def _llm_patch_incar(self, original_text: str, replacements: Dict[str, Any]) -> str:
        rep_json = json.dumps(replacements, ensure_ascii=False, indent=2)

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

            incar_path = os.path.join(target_dir, "INCAR")
            with open(incar_path, "w") as f:
                f.write(incar_text.rstrip() + "\n")
            return incar_path

        
        method_paragraph = context.get("method_paragraph")

        query = {
            "job_name": context.get("job_name"),
            "property": prop,
            "mof": context.get("mof"),
            "guest": context.get("guest"),
            "system_label": system_label,
            "system_role": system_role,
            "vasp_stage": vasp_stage,
            "vasp_calc_type": vasp_calc_type,
            "dos_has_chgcar": dos_has_chgcar,
            "recommended_icharg": icharg,
        }

        vasp_format_raw = select_vasp_format(query)
        vasp_format = vasp_format_raw.replace("{system}", str(system_label))
        vasp_format = vasp_format.replace("{ICHARG}", str(icharg))

        
        vasp_incar_hints = ""
        if not reproduce_mode:
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
                rag_out = miner.run_for_vasp_incar(rag_ctx, top_files=5)
                vasp_incar_hints = (rag_out.get("vasp_incar_hints") or "").strip()
                if vasp_incar_hints:
                    print("[RAG] VASP INCAR hints enabled")
                else:
                    print("[RAG] no relevant VASP INCAR hints")
            except Exception as e:
                print(f"[RAG] VASP INCAR hints disabled due to error: {e}")
                vasp_incar_hints = ""
        else:
            print("[RAG] skipped (reproduce mode)")

        prompt = create_vasp_incar_prompt(
            query,
            vasp_format,
            method_paragraph,
            rag_hints=vasp_incar_hints,
        )

        messages = [
            SystemMessage(content=get_vasp_system_message()),
            HumanMessage(content=prompt),
        ]
        resp = self.llm.invoke(messages)
        incar_text = (resp.content or "").strip()

        if incar_text.startswith("```"):
            lines = incar_text.splitlines()
            incar_text = "\n".join(lines[1:-1]).strip()

        incar_path = os.path.join(target_dir, "INCAR")
        with open(incar_path, "w") as f:
            f.write(incar_text.rstrip() + "\n")

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

        
        self.file_agent.get_vasp_file(dst_structure_path)

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

        
        if vasp_stage == "dos" or vasp_calc_type == "dos":
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