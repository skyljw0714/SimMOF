from typing import Any, Dict, Optional
from pathlib import Path

from screening.workflow import ScreeningWorkflowAgent
from tool.agent import ToolAgent
from output.screening_output import ScreeningOutputAgent
from config import LLM_DEFAULT, SCREENING_CIF_ROOT, SCREENING_WORK_ROOT

SCREENING_PREVIEW_N = 20
SCREENING_SAVE_SUMMARY_JSON = True

class ScreeningAgent:
    
    def __init__(
        self,
        llm=None,
        
        work_root: str = str(SCREENING_WORK_ROOT),
        cif_root: str = str(SCREENING_CIF_ROOT),
        preview_n: int = SCREENING_PREVIEW_N,
        save_summary_json: bool = SCREENING_SAVE_SUMMARY_JSON,
    ):
        self.llm = llm or LLM_DEFAULT
        self.work_root = str(Path(work_root))
        self.cif_root = str(Path(cif_root))

        
        self.workflow_agent = ScreeningWorkflowAgent(
            llm=self.llm,
            save_root=self.work_root,
            cif_root=self.cif_root,
        )
        
        self.tool_agent = ToolAgent(llm=self.llm, work_root=self.work_root)
        
        self.output_agent = ScreeningOutputAgent(
            llm=self.llm,
            preview_n=preview_n,
            save_json=save_summary_json,
        )

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        
        _mof = context.get("mof") or context.get("MOF")
        _guest = context.get("guest") or context.get("Guest")
        _prop = context.get("property") or context.get("Property")
        _job = context.get("job_name")

        
        new_ctx = self.workflow_agent.run(context)

        
        if _job and "job_name" not in new_ctx:
            new_ctx["job_name"] = _job
        if _mof and "mof" not in new_ctx:
            new_ctx["mof"] = _mof
        if _guest is not None and "guest" not in new_ctx:
            new_ctx["guest"] = _guest
        if _prop and "property" not in new_ctx:
            new_ctx["property"] = _prop

        assert "mof" in new_ctx, f"missing mof; keys={list(new_ctx.keys())[:30]}"
        
        new_ctx["tool_mode"] = "screening"
        new_ctx = self.tool_agent.run(new_ctx)

        
        new_ctx = self.output_agent.run(new_ctx)
        return new_ctx