from typing import Dict, Any
from pathlib import Path
import json

from langchain_core.runnables import RunnableLambda

from config import working_dir, LLM_DEFAULT
from core.pipeline import make_pipeline_chain  

from structure.agent import ZeoppStructureAgent
from input.zeopp_input import ZeoppInputAgent
from Zeopp.runner import ZeoppRunner
from output.zeopp_output import ZeoppOutputAgent
from error.zeopp_error import ZeoppErrorAgent


class ZeoppAgent:
    
    def __init__(self, llm=None, max_retries: int = 2, debug_dump: bool = True):
        self.llm = llm or LLM_DEFAULT
        self.max_retries = max_retries
        self.debug_dump = debug_dump

        self.structure_agent = ZeoppStructureAgent()
        self.input_agent = ZeoppInputAgent(llm=self.llm)
        self.runner_agent = ZeoppRunner()
        self.output_agent = ZeoppOutputAgent()
        self.error_agent = ZeoppErrorAgent(
            llm=self.llm,
            max_retries=max_retries,
            zeopp_runner=self.runner_agent,
            zeopp_input_agent=self.input_agent,
        )

        
        self.chain = make_pipeline_chain(
            steps=[
                ("ensure_context_defaults", self._ensure_context_defaults),
                ("ZeoppAgent_START", self._start_marker),
                ("ZeoppStructureAgent", self.structure_agent.run),
                ("ZeoppInputAgent", self.input_agent.run),
                ("ZeoppRunner", self.runner_agent.run),
                ("ZeoppErrorAgent", self.error_agent.run),
                ("ZeoppOutputAgent", self.output_agent.run),
                ("ZeoppAgent_END", self._end_marker),
            ],
            dump_step=(self._dump_step if self.debug_dump else None),
        )

    
    
    
    def _start_marker(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        
        
        return ctx

    def _end_marker(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        return ctx

    
    
    
    def _ensure_context_defaults(self, context: Dict[str, Any]) -> Dict[str, Any]:
        context.setdefault("results", {})

        
        if "job_name" not in context and "plan_name" in context:
            context["job_name"] = context["plan_name"]

        context.setdefault("query_text", "")

        if not context.get("work_dir"):
            job_name = context.get("job_name") or context.get("plan_name") or "zeopp_job"
            wd = str(Path(working_dir) / job_name)
            Path(wd).mkdir(parents=True, exist_ok=True)
            context["work_dir"] = wd

        return context

    
    
    
    def _dump_step(self, ctx: Dict[str, Any], step_agent: str, step_order: int):
        
        base = Path(ctx.get("work_dir", working_dir))
        debug_dir = base / "_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        job_id = ctx.get("job_id", "unknown_job")
        out = debug_dir / f"context_step{step_order:02d}_{step_agent}_{job_id}.json"

        try:
            with open(out, "w", encoding="utf-8") as f:
                json.dump(ctx, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"[ZeoppAgent] Warning: context dump failed at {step_agent}: {e}")

    
    
    
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        
        
        return self.chain.invoke(context)
