import json
from pathlib import Path
from typing import Dict, Any

from config import working_dir, AGENT_LLM_MAP, LLM_DEFAULT

from core.pipeline import make_pipeline_chain
from core.timing import timed_call

from structure.agent import LAMMPSStructureAgent
from input.lammps_input import LAMMPSInputAgent
from LAMMPS.runner import LAMMPSRunner
from error.lammps_error import LAMMPSErrorAgent
from error.structure_regeneration import StructureRegenerationCoordinator
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
        self.structure_regeneration = StructureRegenerationCoordinator("lammps")
        self.output_agent = LAMMPSOutputAgent()

        self.debug_dump = debug_dump

        self.chain = make_pipeline_chain(
            steps=[
                ("ensure_context_defaults", self._timed_step("ensure_context_defaults", self._ensure_context_defaults)),
                ("LAMMPSAgent_START", self._timed_step("LAMMPSAgent_START", self._marker)),
                ("LAMMPSStructureAgent", self._timed_step("LAMMPSStructureAgent", self.structure_agent.run)),
                ("LAMMPSInputAgent", self._timed_step("LAMMPSInputAgent", self.input_agent.run)),
                ("LAMMPSPreRunReview", self._timed_step("LAMMPSPreRunReview", self.error_agent.pre_run_review)),
                ("LAMMPSStructureRegenerationPreRun", self._timed_step("LAMMPSStructureRegenerationPreRun", self.structure_regeneration.run)),
                ("LAMMPSMarkPreRunStructureRegeneration", self._timed_step("LAMMPSMarkPreRunStructureRegeneration", self._mark_pre_run_structure_regeneration)),
                ("LAMMPSRunner", self._timed_step("LAMMPSRunner", self.runner_agent.run)),
                ("LAMMPSErrorAgent", self._timed_step("LAMMPSErrorAgent", self.error_agent.run)),
                ("LAMMPSStructureRegenerationPostError", self._timed_step("LAMMPSStructureRegenerationPostError", self.structure_regeneration.run)),
                ("LAMMPSRetryAfterStructureRegeneration", self._timed_step("LAMMPSRetryAfterStructureRegeneration", self._retry_after_structure_regeneration)),
                ("LAMMPSOutputAgent", self._timed_step("LAMMPSOutputAgent", self.output_agent.run)),
                ("LAMMPSAgent_END", self._timed_step("LAMMPSAgent_END", self._marker)),
            ],
            dump_step=(self._dump_step if self.debug_dump else None),
        )

    def _marker(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        return ctx

    def _timed_step(self, step_name: str, fn):
        def wrapper(ctx: Dict[str, Any]) -> Dict[str, Any]:
            return timed_call(
                step_name,
                fn,
                ctx,
                category="lammps_step",
                context=ctx,
                extra={"parent_agent": "LAMMPSAgent"},
            )
        return wrapper

    def _ensure_context_defaults(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        if not ctx.get("work_dir"):
            job_name = ctx.get("job_name") or ctx.get("plan_name") or "lammps_job"
            wd = str(Path(working_dir) / job_name)
            Path(wd).mkdir(parents=True, exist_ok=True)
            ctx["work_dir"] = wd

        ctx.setdefault("results", {})
        return ctx

    def _mark_pre_run_structure_regeneration(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        attempts = int(ctx.get("structure_regeneration_attempts", 0) or 0)
        if attempts and ctx.get("lammps_success") is not False:
            ctx["_lammps_post_error_structure_regen_reruns"] = max(
                attempts,
                int(ctx.get("_lammps_post_error_structure_regen_reruns", 0) or 0),
            )
        return ctx

    def _retry_after_structure_regeneration(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        attempts = int(ctx.get("structure_regeneration_attempts", 0) or 0)
        already = int(ctx.get("_lammps_post_error_structure_regen_reruns", 0) or 0)
        if attempts <= already:
            return ctx
        if ctx.get("lammps_success") is True:
            return ctx

        ctx["_lammps_post_error_structure_regen_reruns"] = attempts
        ctx = self.error_agent.pre_run_review(ctx)
        ctx = self.runner_agent.run(ctx)
        ctx = self.error_agent.run(ctx)
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

    def resume(self, context: Dict[str, Any]) -> Dict[str, Any]:
        ctx = dict(context)
        ctx.setdefault("results", {})
        ctx["resume_mode"] = True

        work_dir = Path(ctx.get("work_dir", "")) if ctx.get("work_dir") else None
        has_marker = bool(work_dir and any((work_dir / name).exists() for name in ("START", "DONE", "FAILED")))
        has_scheduler = bool(ctx.get("scheduler_job_id"))
        was_submitted = bool(ctx.get("lammps_submitted"))

        if not (has_marker or has_scheduler or was_submitted):
            raise RuntimeError(
                "LAMMPSAgent.resume requires a saved context from or after LAMMPS submission "
                "(scheduler job id, lammps_submitted flag, or START/DONE/FAILED marker required)."
            )

        if has_marker or has_scheduler:
            ctx["lammps_submitted"] = True

        if ctx.get("lammps_success") is not True:
            ctx = self.error_agent.run(ctx)
            ctx = self.structure_regeneration.run(ctx)
            ctx = self._retry_after_structure_regeneration(ctx)

        ctx = self.output_agent.run(ctx)
        return ctx
