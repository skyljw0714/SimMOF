import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import LLM_DEFAULT, AGENT_LLM_MAP


class ScreeningOutputAgent:

    def __init__(self, llm=None, preview_n: int = 20, save_json: bool = True):
        self.llm = llm or AGENT_LLM_MAP.get("ScreeningOutputAgent", LLM_DEFAULT)
        self.preview_n = int(preview_n)
        self.save_json = bool(save_json)

    def _list_cifs(self, cif_dir: Path) -> List[str]:
        if not cif_dir.is_dir():
            return []
        return sorted([p.name for p in cif_dir.glob("*.cif")])

    def _core_params(self, tool: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(params, dict):
            return {}

        tool_l = (tool or "").lower()
        keep_keys = []

        if tool_l == "ase_atom_count":
            keep_keys = ["max_atoms"]
        elif tool_l in ("zeo++", "zeopp", "zeo"):
            
            
            keep_keys = ["probe", "probe_diameter", "free_sphere", "pld", "lcd", "threshold", "op", "value"]
        elif tool_l == "raspa_henry":
            keep_keys = ["molecule", "temperature_K", "temperature", "top_n"]
        else:
            
            keep_keys = ["threshold", "op", "value", "top_n", "temperature_K", "max_atoms", "molecule"]

        core = {k: params.get(k) for k in keep_keys if k in params}
        
        return {k: v for k, v in core.items() if v is not None}

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        context.setdefault("results", {})
        ex = context["results"].get("screening_execution")

        if not isinstance(ex, dict):
            raise RuntimeError("[ScreeningOutputAgent] context['results']['screening_execution'] not found.")

        final_okdir = ex.get("final_okdir")
        final_count = ex.get("final_count")
        logs = ex.get("logs", [])
        results_path = ex.get("results_path")

        
        
        goal = None
        plan = context["results"].get("screening_plan", {})
        if isinstance(plan, dict):
            goal = plan.get("goal")  
        if not goal:
            goal = context.get("query_text") or context.get("user_query") or ""

        
        candidates: List[str] = []
        preview: List[str] = []
        if final_okdir:
            cif_dir = Path(final_okdir)
            candidates = self._list_cifs(cif_dir)
            preview = candidates[: self.preview_n]

        
        step_summaries = []
        if isinstance(logs, list):
            for s in logs:
                if not isinstance(s, dict):
                    continue
                tool = s.get("tool")
                step_summaries.append({
                    "step": s.get("step"),
                    "tool": tool,
                    "condition": s.get("condition"),
                    "before_n": s.get("before_n"),
                    "after_n": s.get("after_n"),
                    "dropped": (s.get("before_n", 0) - s.get("after_n", 0))
                               if isinstance(s.get("before_n"), int) and isinstance(s.get("after_n"), int)
                               else None,
                    "output_okdir": s.get("output_okdir"),
                    "parsed_params_core": self._core_params(str(tool or ""), s.get("parsed_params", {})),
                })

        notes = [
            "This is a pre-screening result (filters/ranking to reduce candidates).",
            "For 'top 10 H2 uptake at 1 bar and 298K', you still need the RASPA uptake workflow to compute and rank finite-pressure uptake."
        ]

        summary = {
            "goal": goal,
            "guest": context.get("guest"),
            "job_name": context.get("job_name"),
            "final_okdir": final_okdir,
            "final_count": final_count if isinstance(final_count, int) else len(candidates),
            "results_path": results_path,
            "steps": step_summaries,
            "candidates_total": len(candidates),
            "candidates_preview_n": self.preview_n,
            "candidates_preview": preview,
            
            "candidates_all": candidates,
            "notes": notes,
        }

        context["results"]["screening_summary"] = summary

        
        if self.save_json:
            try:
                out_path = None
                if results_path:
                    out_path = Path(results_path).with_name("screening_summary.json")
                else:
                    
                    job_name = context.get("job_name", "screening_job")
                    out_path = Path("/home/users/skyljw0714/MOFScientist/working_dir/screening") / job_name / "screening_summary.json"

                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                context["results"]["screening_summary_path"] = str(out_path)
            except Exception as e:
                print(f"[ScreeningOutputAgent] Warning: summary save failed: {e}")

        print(f"[ScreeningOutputAgent] summary saved. final_count={summary['final_count']} final_okdir={final_okdir}")
        return context
