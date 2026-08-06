import json
from pathlib import Path
from typing import Any, Dict, Optional

from langchain.schema import SystemMessage, HumanMessage

from config import SCREENING_CIF_ROOT, SCREENING_WORK_ROOT
from core.llm_logging import log_llm_decision, set_llm_context

SCREENING_ATOM_LIMITS = {"RASPA": 2000, "VASP": 400, "LAMMPS": 5000}

WORKFLOW_PROMPT_TEMPLATE = """
You design screening workflows that are minimal and goal-directed.

Goal:
"{query}"

Context:
- Early manual pre-filters are already applied (structure validity, atom-count cutoff, PLD/probe feasibility).
- You will propose only the additional steps after those manual filters.
- Estimated candidate set size (before manual filters): {n_candidates}

RAG_HINTS (past successful workflows / lab defaults; may be irrelevant):
{rag_hints}

Decision policy:
- Prefer the SMALLEST number of steps that can reliably reach the goal.
- Start from 1 step by default.
- Add a 2nd or 3rd step ONLY if the goal cannot be achieved reliably without it.
- Each step must pay for itself: the reason must explicitly connect to the goal (not generic “better screening”).
- Avoid redundant stability/relaxation filters (e.g., MLIP_geo) unless the goal explicitly mentions stability or collapse.
- Cost–fidelity policy: prefer direct physics-based screening (RASPA_henry) when practical. Use ML surrogates (MOFTransformer) only for explicit fast/approximate prediction or to reduce an impractical physics-based workload. Do not add a surrogate step unless its reason explicitly states the bottleneck it removes.
- Time cost reference: RASPA_henry ~60 s per structure; MOFTransformer ~2 s per structure. Estimate total wall-clock time = n_candidates × cost for each proposed step. If a tool's estimated total time exceeds 7 days, consider using a faster alternative. State the estimated total time in your reason.
- Add cheap prerequisite checks only when they make downstream ranking physically meaningful or prevent avoidable failed calculations.

Ranking policy (STRICT):
- If the goal asks for Top-K, do NOT use K as the cutoff for any screening step.
- When using a proxy metric (approximates but does not directly equal the target property at the stated conditions): eliminate the clearly bottom-ranked candidates based on the proxy score. Use max(1000, 100×K) as the default keep count — treat this as the target, not a floor to exceed. In the reason, state (1) under what idealized conditions the proxy metric would exactly equal the target property, and (2) why those conditions do not hold at the stated simulation conditions.
- A tighter cutoff is allowed only when the proxy metric directly equals the target property under the stated conditions. In that case, keep at least max(10×K, 100) — do NOT use K itself as the cutoff. State the physical reasoning explicitly.
- Express the keep count as an absolute number.
- The condition must say “keep top <keep>” (not “return top K”).
- The final Top-K should be selected only after downstream evaluation on the kept set (e.g., RASPA GCMC, which runs outside this screening workflow).

Tool usage constraint (HARD):
- Each tool name may appear AT MOST ONCE in the workflow steps.
- Do NOT repeat the same tool (e.g., do not use RASPA_henry twice).
- If higher accuracy is desired, express it within a single step (e.g., “use higher precision settings”) rather than adding a repeated step.

Allowed tools:
- zeo++ (fast geometric descriptor calculation: PLD, LCD, void fraction; useful for prerequisite checks before pore-related simulations)
- RASPA_henry (physics-based Henry coefficient via Widom insertion; preferred first-pass ranking tool for adsorption-based screening)
- MLIP_geo (stability/relaxation filter only)
- MLIP_binding (only if binding energy is the target)
- OMD (Open Metal Detector: detect whether MOFs have open metal sites; use to filter IN when OMS is required for target interaction, or filter OUT when OMS causes unwanted reactivity/instability)
- MOFSimplify (ML prediction of thermal stability score and water stability score; use when stability under heat or humidity is a goal criterion)
- MOFTransformer (transformer-based ML prediction of adsorption properties; use as a fast surrogate only when physics-based screening is impractical or explicitly requested)
  Supported downstream keys (use in condition text so the parser picks the right model):
  * "h2_uptake"  — H2 uptake [fine-tuned, high accuracy]
  * "bandgap"    — electronic band gap [fine-tuned, high accuracy]
  * "default"    — general surrogate for CO2, N2, void fraction, surface area, or any other property

Output 0–3 steps. Return JSON only in this format:
{{
  "goal": "<application goal>",
  "steps": [
    {{
      "step": 1,
      "tool": "TOOL_NAME",
      "condition": "FILTERING OR RANKING CONDITION",
      "reason": "WHY THIS CONDITION MATTERS"
    }}
  ]
}}
"""


class ScreeningWorkflowAgent:
    
    def __init__(
        self,
        llm=None,
        save_root: str = str(SCREENING_WORK_ROOT),
        cif_root: str = str(SCREENING_CIF_ROOT),
    ):
        if llm is None:
            raise RuntimeError("ScreeningWorkflowAgent requires llm=...")

        self.llm = llm
        self.save_root = Path(save_root)
        self.save_root.mkdir(parents=True, exist_ok=True)

        self.cif_root = Path(cif_root)
        self.cif_root.mkdir(parents=True, exist_ok=True)

    def _get_screening_rag_hints(self, context: Dict[str, Any], top_files: int = 5) -> str:
        try:
            from rag.agent import RagAgent

            rag_ctx = {
                "job_name": context.get("job_name") or "",
                "mof": context.get("mof") or "",
                "guest": context.get("guest") or "",
                "property": context.get("property") or "",
                "query_text": context.get("user_query") or context.get("query_text") or "",
            }

            agent = RagAgent(agent_name="RagAgent")
            r = agent.run_for_screening_workflows(rag_ctx, top_files=top_files)

            
            return (r.get("workflow_hints") or "").strip()

        except Exception as e:
            print(f"[RAG] Screening workflow hints disabled due to error: {e}")
            return ""

        
    def _safe_json_loads(self, text: str) -> Dict[str, Any]:
        t = (text or "").strip()
        if t.startswith("```"):
            lines = t.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            t = "\n".join(lines).strip()
        return json.loads(t)

    def extract_task(self, query: str, guest: Optional[str]) -> dict:
        guest_text = guest if guest else "unknown"
        prompt = f"""
You are an expert in MOF simulations.
Goal: "{query}"

Known guest gas (if any): "{guest_text}"

Decide simulation task: "RASPA", "VASP", or "LAMMPS".
If task is "RASPA", return:
- gas: the guest gas name (use the known guest if provided)
- probe_diameter: the kinetic diameter (Å) for the gas molecule.

Return JSON only:
{{
  "task": "RASPA|VASP|LAMMPS",
  "gas": "string or null",
  "probe_diameter": number or null
}}
"""
        set_llm_context("ScreeningWorkflow", "task_extraction")
        resp = self.llm.invoke([HumanMessage(content=prompt)])
        data = self._safe_json_loads(getattr(resp, "content", str(resp)))

        
        if guest and data.get("gas") and str(data["gas"]).upper() != str(guest).upper():
            data["gas"] = guest

        
        pd = data.get("probe_diameter", None)
        try:
            if pd is not None:
                pd = float(pd)
                if pd <= 0 or pd > 10:
                    pd = None
        except Exception:
            pd = None
        data["probe_diameter"] = pd
        return data

    def get_manual_steps(self, query_text: str, guest: Optional[str], task_info: Optional[dict] = None) -> list[dict]:
        task_info = task_info or self.extract_task(query_text, guest=guest)
        task = task_info.get("task")
        probe = task_info.get("probe_diameter", None)

        manual_steps = [
            {
                "step": 1,
                "tool": "MOFChecker",
                "condition": "remove MOFs with structural issues",
                "reason": "invalid structures cannot be simulated",
            }
        ]

        if task == "RASPA":
            atom_limit = SCREENING_ATOM_LIMITS["RASPA"]
            atom_reason = "RASPA is costly for very large unit cells"
        elif task == "VASP":
            atom_limit = SCREENING_ATOM_LIMITS["VASP"]
            atom_reason = "DFT is very costly for large systems"
        elif task == "LAMMPS":
            atom_limit = SCREENING_ATOM_LIMITS["LAMMPS"]
            atom_reason = "MD becomes costly for very large systems"
        else:
            atom_limit = SCREENING_ATOM_LIMITS["RASPA"]
            atom_reason = "large unit cells are computationally expensive"

        manual_steps.append({
            "step": 2,
            "tool": "ASE_atom_count",
            "condition": f"filter out MOFs with >{atom_limit} atoms",
            "reason": atom_reason,
            "max_atoms": atom_limit,
        })

        if task == "RASPA" and probe is not None:
            manual_steps.append({
                "step": 3,
                "tool": "zeo++",
                "condition": f"free_sphere >= {probe} Å",
                "reason": "gas molecules cannot diffuse/insert if pore limiting diameter is too small",
            })

        return manual_steps

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        job_name = context.get("job_name", "screening_job")
        guest = context.get("guest", None)
        query_text = context.get("query_text", "")

        job_dir = self.save_root / job_name
        job_dir.mkdir(parents=True, exist_ok=True)

        
        context["screening_input_dir"] = str(self.cif_root)

        
        manual_steps = self.get_manual_steps(
            query_text=query_text, guest=guest,
            task_info=context.get("_task_info"),
        )
        manual_n = len(manual_steps)

        
        rag_text = self._get_screening_rag_hints(context, top_files=5)

        n_candidates = len(list(Path(self.cif_root).glob("*.cif")))
        n_candidates_str = str(n_candidates) if n_candidates > 0 else "unknown"

        prompt = WORKFLOW_PROMPT_TEMPLATE.format(
            query=query_text,
            rag_hints=rag_text if rag_text else "None",
            n_candidates=n_candidates_str,
        )

        messages = [
            SystemMessage(content="You design compact, executable screening workflows. Output valid JSON only."),
            HumanMessage(content=prompt),
        ]
        set_llm_context("ScreeningWorkflow", "workflow_design")
        resp = self.llm.invoke(messages)
        wf_extra = self._safe_json_loads(getattr(resp, "content", str(resp)))

        extra_steps = wf_extra.get("steps", [])
        if not isinstance(extra_steps, list):
            extra_steps = []

        
        for i, s in enumerate(extra_steps, start=manual_n + 1):
            s["step"] = i

        final_workflow = {
            "goal": wf_extra.get("goal", query_text),
            "steps": manual_steps + extra_steps,
        }
        try:
            log_llm_decision("ScreeningWorkflow", "workflow_design", final_workflow,
                             {"query_text": query_text})
        except Exception:
            pass

        from config import ask_user_confirmation

        plan_summary = "\n".join(
            f"  Step {s.get('step','?')}: {s.get('tool','?')} — {s.get('condition', s.get('reason',''))}"
            for s in final_workflow["steps"]
        )
        print(f"\n[ScreeningAgent] Proposed screening workflow:\n{plan_summary}")

        def _reinvoke_screening(instruction: str) -> str:
            revised_prompt = prompt + f"\n\nUser instruction: {instruction}\nRevise your workflow accordingly."
            set_llm_context("ScreeningWorkflow", "workflow_design_revision")
            r = self.llm.invoke([
                SystemMessage(content="You design compact, executable screening workflows. Output valid JSON only."),
                HumanMessage(content=revised_prompt),
            ])
            return getattr(r, "content", str(r))

        action, revised_text = ask_user_confirmation(
            "ScreeningAgent", plan_summary, reinvoke_fn=_reinvoke_screening, required=True
        )
        if action == "apply" and revised_text != plan_summary:
            try:
                wf2 = self._safe_json_loads(revised_text)
                extra2 = wf2.get("steps", [])
                if isinstance(extra2, list):
                    for i, s in enumerate(extra2, start=manual_n + 1):
                        s["step"] = i
                    final_workflow = {
                        "goal": wf2.get("goal", query_text),
                        "steps": manual_steps + extra2,
                    }
                    print("[ScreeningAgent] Workflow updated per user instruction.")
            except Exception:
                pass

        out_path = job_dir / "screening_workflow.json"
        out_path.write_text(json.dumps(final_workflow, ensure_ascii=False, indent=2), encoding="utf-8")

        
        context["screening_workflow"] = final_workflow
        context["screening_workflow_path"] = str(out_path)
        context.setdefault("results", {})
        context["results"]["screening_plan"] = {
            "goal": final_workflow.get("goal", query_text),
            "n_manual_steps": manual_n,
            "n_extra_steps": len(extra_steps),
            "tools": [s.get("tool") for s in final_workflow["steps"]],
            "path": str(out_path),
        }
        context["tool_mode"] = "screening"
        return context
