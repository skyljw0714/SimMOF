import json
import pandas as pd

from pathlib import Path
from typing import Any, Dict, List, Optional

from tool.parsing import extract_conditions_with_llm
from tool.utils import (
    run_mofchecker,
    run_ase_atom_count,
    run_zeopp,
    run_raspa_henry,
)
from tool.remote_mlip import run_remote_mlip_be
from config import SCREENING_WORK_ROOT

SCREENING_DEFAULT_HENRY_TEMPERATURE_K = 298.0
SCREENING_DEFAULT_TOP_N = 100


class ToolAgent:
    def __init__(
        self,
        llm=None,
        work_root: str = str(SCREENING_WORK_ROOT),
    ):
        self.llm = llm
        self.work_root = Path(work_root)

    def _load_workflow(self, context: Dict[str, Any]) -> Dict[str, Any]:
        wf = context.get("screening_workflow")
        if isinstance(wf, dict) and "steps" in wf:
            return wf

        wf_path = context.get("screening_workflow_path")
        if not wf_path:
            raise RuntimeError("[ToolAgent] screening_workflow or screening_workflow_path not found in context.")

        wf_path = Path(wf_path)
        if not wf_path.exists():
            raise FileNotFoundError(f"[ToolAgent] workflow file not found: {wf_path}")

        return json.loads(wf_path.read_text(encoding="utf-8"))

    def _resolve_input_dir(self, context: Dict[str, Any]) -> Path:
        for k in ("screening_input_dir", "cif_dir", "input_dir"):
            v = context.get(k)
            if v:
                p = Path(v)
                if not p.exists():
                    raise FileNotFoundError(f"[ToolAgent] {k} path does not exist: {p}")
                return p
        raise RuntimeError("[ToolAgent] No input CIF directory in context.")

    def _run_screening(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if self.llm is None:
            raise RuntimeError("[ToolAgent] ToolAgent needs llm=... because parsing uses LLM.")

        
        job_name = context.get("job_name", "screening_job")
        guest = context.get("guest") or context.get("Guest")
        query_text = context.get("query_text", "")

        
        wf = self._load_workflow(context)
        goal = wf.get("goal", query_text)
        steps: List[dict] = wf.get("steps", [])
        if not steps:
            raise RuntimeError("[ToolAgent] workflow has no steps.")

        
        work_root = self.work_root
        work_root.mkdir(parents=True, exist_ok=True)

        job_dir = work_root / job_name
        job_dir.mkdir(parents=True, exist_ok=True)

        
        current_dir: Path = self._resolve_input_dir(context)

        
        
        work_dir: Path = job_dir

        logs: List[dict] = []

        
        base_context = {
            "job_name": context.get("job_name", job_name),
            "plan_name": context.get("plan_name"),
            "plan_root": context.get("plan_root"),
            "guest": guest,
            "property": context.get("property"),
            "query_text": context.get("query_text", query_text),
        }

        for step in steps:
            step_idx = int(step.get("step", 0))
            tool = str(step.get("tool", "")).strip()
            condition = str(step.get("condition", "")).strip()

            step_dir = job_dir / f"step_{step_idx:02d}_{tool}"
            step_dir.mkdir(parents=True, exist_ok=True)

            okdir = step_dir / "ok_cifs"
            okdir.mkdir(parents=True, exist_ok=True)

            before_n = len(list(current_dir.glob("*.cif")))

            print("[STEP]", tool, "|", condition, "|", goal, "|", guest)

            
            try:
                params = extract_conditions_with_llm(step, goal=goal, llm=self.llm)
            except Exception as e:
                print(f"[WARN] LLM parsing failed at step {step_idx} ({tool}): {e}")
                params = {}

            print("[PARSED]", params)

            
            if tool == "MOFChecker":
                kept_paths = run_mofchecker(str(current_dir), str(okdir))

            elif tool == "ASE_atom_count":
                max_atoms = params.get("max_atoms", step.get("max_atoms"))
                if max_atoms is None:
                    raise ValueError(f"[ToolAgent] ASE_atom_count needs max_atoms. step={step}")
                kept_paths = run_ase_atom_count(str(current_dir), str(okdir), int(max_atoms))

            elif tool.lower() in ("zeo++", "zeopp", "zeo"):
                
                kept_paths = run_zeopp(
                    step=step,
                    goal=goal,
                    input_cif_dir=str(current_dir),
                    okdir=str(okdir),
                    work_dir=str(work_dir),
                    llm=self.llm,
                    base_context=base_context,   
                )

            elif tool == "RASPA_henry":
                molecule = params.get("molecule") or (guest if guest else None)
                temperature = float(params.get("temperature_K", SCREENING_DEFAULT_HENRY_TEMPERATURE_K))
                mol_map = {
                    "H2": "hydrogen",
                    "carbon dioxide": "CO2",
                    "nitrogen": "N2",
                    "CH4": "methane",
                }
                molecule = mol_map.get(molecule, molecule)
                top_n = int(params.get("top_n", SCREENING_DEFAULT_TOP_N))

                if molecule is None:
                    raise ValueError(f"[ToolAgent] RASPA_henry needs molecule. step={step}")

                parse_dir = step_dir / "raspa_parse"
                kept_paths = run_raspa_henry(
                    cif_dir=str(current_dir),
                    parse_dir=str(parse_dir),
                    okdir=str(okdir),
                    molecule=str(molecule),
                    temperature=temperature,
                    top_n=top_n,
                )

            else:
                raise ValueError(f"[ToolAgent] Unsupported tool: {tool}")

            after_n = len(kept_paths)
            dropped = before_n - after_n

            print(f"[STEP {step_idx:02d}] {tool}")
            print(f"  input : {before_n}")
            print(f"  kept  : {after_n}")
            print(f"  drop  : {dropped}")
            print(f"  outdir: {okdir}\n")

            logs.append(
                {
                    "step": step_idx,
                    "tool": tool,
                    "condition": condition,
                    "input_dir": str(current_dir),
                    "output_okdir": str(okdir),
                    "before_n": before_n,
                    "after_n": after_n,
                    "parsed_params": params,
                }
            )

            
            current_dir = okdir

        
        result_json = job_dir / "screening_results.json"
        result_json.write_text(
            json.dumps(
                {
                    "goal": goal,
                    "final_okdir": str(current_dir),
                    "final_count": len(list(current_dir.glob("*.cif"))),
                    "logs": logs,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        context.setdefault("results", {})
        context["results"]["screening_execution"] = {
            "final_okdir": str(current_dir),
            "final_count": len(list(current_dir.glob("*.cif"))),
            "logs": logs,
            "results_path": str(result_json),
        }
        context["screening_okdir"] = str(current_dir)

        print(f"[ToolAgent] Screening done. final={current_dir}")
        return context

    def _run_mlip_binding(self, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            work_dir = Path(context["work_dir"]).resolve()
            host_cif = Path(context["mof_path"]).resolve()
            guest_xyz = Path(context["guest_path"]).resolve()
            complex_paths: List[str] = context["complex_cif_paths"]
        except KeyError as e:
            raise RuntimeError(f"[ToolAgent.mlip] context key '{e.args[0]}' is missing.") from e

        if not complex_paths:
            raise RuntimeError("[ToolAgent.mlip] complex_cif_paths is empty.")

        packmol_dir = Path(complex_paths[0]).resolve().parent

        local_out = work_dir / "mlip"
        local_out.mkdir(parents=True, exist_ok=True)

        okdir_name = f"ok_{host_cif.stem}_{guest_xyz.stem}"
        top_n = len(complex_paths)

        print(f"[ToolAgent.mlip] run MLIP for {packmol_dir}")
        csv_path, okdir_path = run_remote_mlip_be(
            host_cif=str(host_cif),
            complex_dir=str(packmol_dir),
            guest_xyz=str(guest_xyz),
            okdir=okdir_name,
            top_n=top_n,
            local_output_dir=str(local_out),
        )

        print(f"[ToolAgent.mlip] remote MLIP done. csv={csv_path}, okdir={okdir_path}")

        if not Path(csv_path).exists():
            raise RuntimeError(f"[ToolAgent.mlip] MLIP result CSV does not exist: {csv_path}")

        df = pd.read_csv(csv_path)

        energy_col = "BindingEnergy(eV)"
        cif_col = "path"

        best_idx = df[energy_col].idxmin()
        best_row = df.loc[best_idx]

        best_energy = float(best_row[energy_col])
        best_cif_path = Path(best_row[cif_col])

        print(f"[ToolAgent.mlip] best = {best_cif_path.name}, E = {best_energy:.4f}")

        context.setdefault("results", {})
        context["results"].setdefault("mlip_binding", {})
        context["results"]["mlip_binding"].update(
            {
                "csv_path": str(csv_path),
                "best_energy": best_energy,
                "best_cif": str(best_cif_path),
            }
        )

        context["best_complex_cif_path"] = str(best_cif_path)
        context["best_binding_energy_mlip"] = best_energy

        return context

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        mode = context.get("tool_mode")
        if mode is None:
            raise RuntimeError("ToolAgent requires explicit tool_mode")

        if mode == "screening":
            return self._run_screening(context)
        elif mode == "mlip_binding":
            return self._run_mlip_binding(context)
        else:
            raise ValueError(
                f"[ToolAgent] Unknown tool_mode='{mode}'. Expected 'screening' or 'mlip_binding'."
            )
