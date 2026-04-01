import json
import re
from pathlib import Path
from typing import Dict, Any, List

from langchain.schema import HumanMessage, SystemMessage

from config import working_dir, LLM_DEFAULT
from core.pipeline import make_pipeline_chain

from structure.agent import RASPAStructureAgent
from input.raspa_input import RASPAInputAgent
from RASPA.runner import RASPARunner
from error.raspa_error import RASPAErrorAgent
from output.raspa_output import RASPAOutputAgent


class RASPAAgent:

    def __init__(self, llm=None, debug_dump: bool = False):
        self.llm = llm or LLM_DEFAULT
        self.debug_dump = debug_dump

        self.structure_agent = RASPAStructureAgent()
        self.input_agent = RASPAInputAgent(llm=self.llm)
        self.runner_agent = RASPARunner(llm=self.llm)
        self.error_agent = RASPAErrorAgent(llm=self.llm)
        self.output_agent = RASPAOutputAgent()

        self.chain = make_pipeline_chain(
            steps=[
                ("ensure_context_defaults", self._ensure_context_defaults),
                ("RASPAAgent_START", self._marker),
                ("attach_screening_okdir", self._attach_screening_okdir_from_upstream),
                ("RASPAStructureAgent", self.structure_agent.run),
                ("maybe_build_batch", self._maybe_build_pressure_batch),
                ("RASPAInputAndSubmit", self._input_and_submit), 
                ("RASPA_SUBMITTED", self._marker),
                ("RASPAErrorAgent", self.error_agent.run),
                ("RASPAOutputAgent", self.output_agent.run),
                ("RASPAAgent_END", self._marker),
            ],
            dump_step=(self._dump_step if self.debug_dump else None),
        )

    def _marker(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        return ctx

    def _infer_pressure_grid_with_llm(self, context: Dict[str, Any], default_single: float = 1.0) -> List[float]:
        qtxt = (context.get("query_text") or context.get("user_query") or "").strip()
        if not qtxt or self.llm is None:
            return [default_single]

        system_msg = (
            "You create a pressure grid (in bar) for adsorption simulations.\n"
            "Return ONLY JSON like {\"pressures_bar\": [0.01, 0.02, ..., 10.0]}.\n"
            "Rules (follow in this priority order):\n"
            "1) If the user specifies an explicit LIST of pressures (e.g., '0.15 and 1 bar' or '0.05, 0.1, 1 bar'), return EXACTLY those values (do NOT interpolate).\n"
            "2) If the user asks for ONLY two pressures / two-point / working capacity / endpoints only, return EXACTLY the two endpoint pressures.\n"
            "3) If the user specifies a pressure RANGE and also requests N points, return N points (prefer log-spaced for wide ranges).\n"
            "4) If the user specifies a pressure RANGE without N, choose a reasonable number of points (typically 8-12) and prefer log-spaced for wide ranges.\n"
            "5) If no pressure is specified, return [1.0] unless a single pressure is specified.\n"
            "Additional constraints:\n"
            "- Include endpoints when generating a range grid.\n"
            "- Do not include any extra text."
        )
        human_msg = HumanMessage(content=f'User query:\n"""{qtxt}"""\n\nReturn pressures_bar.')

        try:
            resp = self.llm.invoke([SystemMessage(content=system_msg), human_msg])
            text = (resp.content or "").strip()
            if text.startswith("```"):
                text = "\n".join(text.splitlines()[1:-1]).strip()

            obj = json.loads(text)
            arr = obj.get("pressures_bar", None)
            if not isinstance(arr, list) or len(arr) == 0:
                return [default_single]

            vals: List[float] = []
            for x in arr:
                try:
                    v = float(x)
                    if v > 0:
                        vals.append(v)
                except Exception:
                    pass

            if not vals:
                return [default_single]

            return sorted(set(vals))

        except Exception as e:
            print(f"[RASPAAgent] pressure grid LLM/parsing failed: {e}")
            return [default_single]

    def _ensure_context_defaults(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        ctx.setdefault("results", {})
        if "job_name" not in ctx and "plan_name" in ctx:
            ctx["job_name"] = ctx["plan_name"]
        ctx.setdefault("query_text", "")

        if not ctx.get("work_dir"):
            job_name = ctx.get("job_name") or ctx.get("plan_name") or "raspa_job"
            wd = str(Path(working_dir) / job_name)
            Path(wd).mkdir(parents=True, exist_ok=True)
            ctx["work_dir"] = wd

        return ctx

    def _attach_screening_okdir_from_upstream(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if context.get("screening_okdir"):
            return context

        upstream = context.get("upstream_plans") or {}
        found = None
        for _plan_name, jobs in upstream.items():
            if not isinstance(jobs, dict):
                continue
            for _job_id, job in jobs.items():
                if not isinstance(job, dict):
                    continue
                found = (
                    job.get("results", {})
                    .get("screening_execution", {})
                    .get("final_okdir")
                )
                if found:
                    break
            if found:
                break

        if found:
            context["screening_okdir"] = found
        return context

    def _maybe_build_pressure_batch(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(context.get("batch"), list) and len(context["batch"]) > 0:
            return context

        if context.get("pressure_bar") is not None or context.get("pressure_pa") is not None:
            context.pop("batch", None)
            return context

        job_name = str(context.get("job_name") or "")
        m = re.search(r"_(\d+(?:\.\d+)?)bar_", job_name)
        if m:
            pbar = float(m.group(1))
            context["pressure_bar"] = pbar
            context["pressure_pa"] = pbar * 1e5
            context.pop("batch", None)
            return context

        prop = (context.get("property") or "").strip().lower().replace(" ", "_").replace("-", "_")

        context.pop("batch", None)

        if prop in ("uptake", "adsorption_isotherm", "isotherm","isosteric_heat"):
            pressures_bar = self._infer_pressure_grid_with_llm(context, default_single=1.0)

            if len(pressures_bar) > 1:
                base_job = context.get("job_name", "raspa_uptake")
                base_wd = Path(context["work_dir"])

                batch = []
                for pbar in pressures_bar:
                    sub = dict(context)

                    sub["pressure_bar"] = float(pbar)
                    sub["pressure_pa"] = float(pbar) * 1e5

                    tag = f"{pbar:.6g}bar".replace(".", "p")
                    sub["job_name"] = f"{base_job}_{tag}"

                    sub_wd = base_wd.parent / sub["job_name"]
                    sub_wd.mkdir(parents=True, exist_ok=True)
                    sub["work_dir"] = str(sub_wd)

                    batch.append(sub)

                context["batch"] = batch

        return context
        
    def _input_and_submit(self, context: Dict[str, Any]) -> Dict[str, Any]:
        batch = context.get("batch")

        if isinstance(batch, list) and len(batch) > 0:
            
            
            if "raspa_rag_hints" not in context:
                
                _ = self.input_agent._get_raspa_rag_hints(context, top_files=5)

            
            common_hints = context.get("raspa_rag_hints")

            new_batch = []
            for subctx in batch:
                subctx = self._ensure_context_defaults(subctx)

                if common_hints and "raspa_rag_hints" not in subctx:
                    subctx["raspa_rag_hints"] = common_hints

                subctx = self.input_agent.run(subctx)
                subctx = self.runner_agent.run(subctx)
                new_batch.append(subctx)

            context["batch"] = new_batch
            return context

        context = self.input_agent.run(context)
        context = self.runner_agent.run(context)
        return context

    def _dump_step(self, ctx: Dict[str, Any], step_agent: str, step_order: int):
        if not self.debug_dump:
            return

        def _dump_one(one: Dict[str, Any], suffix: str):
            base = Path(one.get("work_dir", working_dir))
            debug_dir = base / "_debug"
            debug_dir.mkdir(parents=True, exist_ok=True)

            job_id = one.get("job_id", "unknown_job")
            out = debug_dir / f"context_step{step_order:02d}_{step_agent}_{job_id}{suffix}.json"

            try:
                with open(out, "w", encoding="utf-8") as f:
                    json.dump(one, f, indent=2, ensure_ascii=False, default=str)
            except Exception as e:
                print(f"[RASPAAgent] Warning: context dump failed at {step_agent}: {e}")

        _dump_one(ctx, "")

        batch = ctx.get("batch")
        if isinstance(batch, list) and batch:
            for i, sub in enumerate(batch):
                if isinstance(sub, dict):
                    _dump_one(sub, f"__sub{i:03d}")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return self.chain.invoke(context)
