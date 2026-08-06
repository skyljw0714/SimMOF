from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

from core.job_manager import record_job_event


REQUEST_KEY = "structure_regeneration_request"


def request_structure_regeneration(
    context: Dict[str, Any],
    *,
    software: str,
    reason: str,
    action: str = "regenerate_structure",
    status: str = "requested",
    max_attempts: int = 2,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    request = {
        "status": status,
        "software": software.lower(),
        "reason": reason,
        "action": action,
        "max_attempts": int(max_attempts),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if metadata:
        request["metadata"] = metadata

    context[REQUEST_KEY] = request
    context.setdefault("results", {})[REQUEST_KEY] = request
    context[f"{software.lower()}_needs_structure_regeneration"] = True
    return context


def get_structure_regeneration_request(context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    request = context.get(REQUEST_KEY)
    if isinstance(request, dict) and request.get("status") in {"requested", "retry"}:
        return request
    return None


def clear_structure_regeneration_request(context: Dict[str, Any], *, status: str = "handled") -> None:
    request = context.get(REQUEST_KEY)
    if isinstance(request, dict):
        request["status"] = status
        context.setdefault("results", {})[REQUEST_KEY] = request


class StructureRegenerationCoordinator:

    def __init__(self, software: str, mode: str = "prepare_only"):
        self.software = software.lower()
        self.mode = mode

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        request = get_structure_regeneration_request(context)
        if not request:
            return context
        if request.get("software") not in {self.software, "generic"}:
            return context

        attempts = int(context.get("structure_regeneration_attempts", 0) or 0)
        max_attempts = int(request.get("max_attempts", 2) or 0)
        if attempts >= max_attempts:
            request["status"] = "blocked"
            request["blocked_reason"] = "maximum structure regeneration attempts reached"
            context.setdefault("results", {})[REQUEST_KEY] = request
            record_job_event(
                context,
                "blocked",
                message="Structure regeneration maximum attempts reached",
                metadata=request,
            )
            return context

        context["structure_regeneration_attempts"] = attempts + 1
        record_job_event(
            context,
            "structure_regeneration_started",
            message=f"{self.software.upper()} structure regeneration started",
            metadata=request,
        )

        if self.software == "lammps":
            context = self._run_lammps(context)
        elif self.software == "raspa":
            context = self._run_raspa(context)
        elif self.software == "zeopp":
            context = self._run_zeopp(context)
        else:
            request["status"] = "blocked"
            request["blocked_reason"] = f"unsupported software: {self.software}"
            context.setdefault("results", {})[REQUEST_KEY] = request
            return context

        clear_structure_regeneration_request(context, status="handled")
        context[f"{self.software}_needs_structure_regeneration"] = False
        record_job_event(
            context,
            "structure_regeneration_done",
            message=f"{self.software.upper()} structure regeneration handled",
            metadata=context.get(REQUEST_KEY) or {},
        )
        return context

    def _reset_work_dir_for_regeneration(self, context: Dict[str, Any], software: str) -> None:
        work_dir = context.get("work_dir")
        if not work_dir:
            return
        root = Path(work_dir)
        regen_dir = root / "structure_regeneration" / f"attempt_{context['structure_regeneration_attempts']:02d}"
        regen_dir.mkdir(parents=True, exist_ok=True)
        context["previous_work_dir"] = str(root)
        context["work_dir"] = str(regen_dir)
        context["job_name"] = f"{context.get('job_name') or software}_regen_{context['structure_regeneration_attempts']:02d}"

    def _run_lammps(self, context: Dict[str, Any]) -> Dict[str, Any]:
        from structure.agent import LAMMPSStructureAgent
        from input.lammps_input import LAMMPSInputAgent

        self._reset_work_dir_for_regeneration(context, "lammps")
        context = LAMMPSStructureAgent().run(context)
        context = LAMMPSInputAgent().run(context)
        return context

    def _run_raspa(self, context: Dict[str, Any]) -> Dict[str, Any]:
        from structure.agent import RASPAStructureAgent
        from input.raspa_input import RASPAInputAgent

        self._reset_work_dir_for_regeneration(context, "raspa")
        context = RASPAStructureAgent().run(context)
        context = RASPAInputAgent().run(context)
        return context

    def _run_zeopp(self, context: Dict[str, Any]) -> Dict[str, Any]:
        from structure.agent import ZeoppStructureAgent
        from input.zeopp_input import ZeoppInputAgent

        self._reset_work_dir_for_regeneration(context, "zeopp")
        context = ZeoppStructureAgent().run(context)
        context = ZeoppInputAgent().run(context)
        return context
