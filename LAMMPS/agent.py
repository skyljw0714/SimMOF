import json
from pathlib import Path
from typing import Dict, Any

from config import working_dir, AGENT_LLM_MAP, LLM_DEFAULT

from core.pipeline import make_pipeline_chain

from structure.agent import LAMMPSStructureAgent
from input.lammps_input import LAMMPSInputAgent
from LAMMPS.runner import LAMMPSRunner
from error.lammps_error import LAMMPSErrorAgent
from output.lammps_output import LAMMPSOutputAgent


class LAMMPSAgent:
    
    def __init__(self, llm=None, max_retries: int = 2, debug_dump: bool = True):
        self.llm = llm or AGENT_LLM_MAP.get("LAMMPSAgent", LLM_DEFAULT)

        self.structure_agent = LAMMPSStructureAgent()
        self.input_agent = LAMMPSInputAgent()
        self.runner_agent = LAMMPSRunner()

        self.error_agent = LAMMPSErrorAgent(
            llm=AGENT_LLM_MAP.get("LAMMPSErrorAgent", self.llm),
            max_lines=200,
        )
        self.output_agent = LAMMPSOutputAgent()

        self.debug_dump = debug_dump

        self.chain = make_pipeline_chain(
            steps=[
                ("ensure_context_defaults", self._ensure_context_defaults),
                ("LAMMPSAgent_START", self._marker),
                ("LAMMPSStructureAgent", self.structure_agent.run),
                ("LAMMPSInputAgent", self.input_agent.run),
                ("LAMMPSRunner", self.runner_agent.run),
                ("LAMMPSErrorAgent", self.error_agent.run),
                ("LAMMPSOutputAgent", self.output_agent.run),
                ("LAMMPSAgent_END", self._marker),
            ],
            dump_step=(self._dump_step if self.debug_dump else None),
        )

    def _marker(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        return ctx

    def _ensure_context_defaults(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        if not ctx.get("work_dir"):
            job_name = ctx.get("job_name") or ctx.get("plan_name") or "lammps_job"
            wd = str(Path(working_dir) / job_name)
            Path(wd).mkdir(parents=True, exist_ok=True)
            ctx["work_dir"] = wd

        ctx.setdefault("results", {})
        return ctx

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
            print(f"[LAMMPSAgent] Warning: context dump failed at {step_agent}: {e}")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return self.chain.invoke(context)
