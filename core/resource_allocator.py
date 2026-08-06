import json
import os
import re
import shutil
import sqlite3
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional

from langchain.schema import HumanMessage, SystemMessage
from core.llm_logging import log_llm_decision, set_llm_context

HARDWARE = {
    "node_type": "aa",
    "parallel_ppn_choices": [4, 8, 16],
    "serial_ppn_choices": [1],
    "max_nodes": 4,
    "queue": "long",
}

_SYSTEM_PROMPT = """\
You are an HPC resource allocation expert for computational chemistry on a PBS cluster.
Return ONLY valid JSON. No markdown. No extra keys."""

_USER_PROMPT = """\
Recommend PBS resource allocation for the following simulation.

Cluster hardware constraints (must be satisfied):
- nodes: positive integer (1, 2, 3, ...)
- ppn: must be one of {ppn_choices}
- Maximum nodes: 4
- Queue: long

Current scheduler availability snapshot:
{availability_summary}

Simulation details:
  software:  {software}
  calc_type: {calc_type}
  n_atoms:   {n_atoms}
  mof:       {mof}
  guest:     {guest}

Based on your knowledge of computational chemistry and HPC workloads,
determine the appropriate number of nodes and cores per node
considering the calculation type, system size, and current scheduler
availability. If free nodes are scarce, prefer a smaller allocation that can
start sooner; do not request more currently free nodes than reported.

Return JSON exactly:
{{
  "nodes": <integer 1-4>,
  "ppn":   <integer, must be one of {ppn_choices}>,
  "np":    <integer = nodes * ppn>,
  "queue": "long",
  "rationale": "<one sentence>"
}}"""


class ResourceAllocationError(RuntimeError):
    pass


def ppn_choices_for_software(software: str) -> List[int]:
    software_name = (software or "").upper()
    if software_name == "RASPA":
        return HARDWARE["serial_ppn_choices"]
    return HARDWARE["parallel_ppn_choices"]


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f} sec"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f} min"
    hours = minutes / 60.0
    if hours < 48:
        return f"{hours:.1f} h"
    return f"{hours / 24.0:.1f} d"


@dataclass
class ResourceSpec:
    nodes: int
    ppn: int
    np: int
    queue: str
    rationale: str

    def pbs_nodes_string(self) -> str:
        return f"nodes={self.nodes}:ppn={self.ppn}:{HARDWARE['node_type']}"


@dataclass
class SchedulerAvailability:
    available: bool
    source: str
    total_nodes: int = 0
    free_nodes: int = 0
    busy_nodes: int = 0
    offline_nodes: int = 0
    total_cores: int = 0
    free_cores: int = 0
    running_jobs: int = 0
    queued_jobs: int = 0
    free_node_cores: List[int] = field(default_factory=list)
    message: str = ""
    raw_excerpt: str = ""

    def summary(self) -> str:
        if not self.available:
            return f"unavailable ({self.source}): {self.message or 'scheduler snapshot could not be collected'}"
        free_ppn = ", ".join(str(v) for v in sorted(self.free_node_cores, reverse=True)[:8])
        if len(self.free_node_cores) > 8:
            free_ppn += ", ..."
        return (
            f"source={self.source}; total_nodes={self.total_nodes}; free_nodes={self.free_nodes}; "
            f"busy_nodes={self.busy_nodes}; offline_nodes={self.offline_nodes}; "
            f"free_cores={self.free_cores}/{self.total_cores}; "
            f"running_jobs={self.running_jobs}; queued_jobs={self.queued_jobs}; "
            f"free_node_cores=[{free_ppn or 'none'}]"
        )

    def nodes_with_at_least(self, ppn: int) -> int:
        return sum(1 for cores in self.free_node_cores if cores >= ppn)


def _run_scheduler_command(cmd: List[str], timeout: int = 5) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def _parse_int(value: str, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _parse_pbsnodes(text: str, node_type: str = "aa") -> SchedulerAvailability:
    nodes = []
    current = None

    for raw_line in (text or "").splitlines():
        if not raw_line.strip():
            continue
        if raw_line[:1].isspace():
            if current is None or "=" not in raw_line:
                continue
            key, value = raw_line.strip().split("=", 1)
            current[key.strip()] = value.strip()
            continue
        current = {"name": raw_line.strip()}
        nodes.append(current)

    filtered = []
    for node in nodes:
        properties = node.get("properties") or node.get("resources_available.host") or ""
        if not node_type or node_type in properties.split(",") or node_type in node.get("name", ""):
            filtered.append(node)

    if not filtered and nodes:
        filtered = nodes

    free_node_cores = []
    total_nodes = len(filtered)
    busy_nodes = 0
    offline_nodes = 0
    total_cores = 0

    for node in filtered:
        state = (node.get("state") or "unknown").lower()
        state_tokens = {part.strip() for part in re.split(r"[, ]+", state) if part.strip()}
        ncpus = (
            _parse_int(node.get("np", "0"))
            or _parse_int(node.get("pcpus", "0"))
            or _parse_int(node.get("resources_available.ncpus", "0"))
            or max(HARDWARE["parallel_ppn_choices"])
        )
        total_cores += ncpus
        if state_tokens & {"down", "offline", "unknown", "stale"}:
            offline_nodes += 1
        elif "free" in state_tokens and not (state_tokens & {"job-exclusive", "job-sharing", "busy"}):
            free_node_cores.append(ncpus)
        else:
            busy_nodes += 1

    return SchedulerAvailability(
        available=True,
        source="pbsnodes",
        total_nodes=total_nodes,
        free_nodes=len(free_node_cores),
        busy_nodes=busy_nodes,
        offline_nodes=offline_nodes,
        total_cores=total_cores,
        free_cores=sum(free_node_cores),
        free_node_cores=free_node_cores,
        raw_excerpt=(text or "")[:1000],
    )


def _parse_qstat_jobs(text: str) -> Dict[str, int]:
    running = 0
    queued = 0
    for line in (text or "").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        state = parts[-2] if len(parts[-2]) == 1 else parts[-1]
        if state == "R":
            running += 1
        elif state in {"Q", "H", "W"}:
            queued += 1
    return {"running_jobs": running, "queued_jobs": queued}


def _availability_from_context(context: Dict[str, Any]) -> Optional[SchedulerAvailability]:
    data = context.get("scheduler_availability") or context.get("resource_availability")
    if not isinstance(data, dict):
        return None

    free_node_cores = data.get("free_node_cores") or []
    free_node_cores = [_parse_int(v) for v in free_node_cores if _parse_int(v) > 0]
    total_nodes = _parse_int(data.get("total_nodes"), len(free_node_cores))
    free_nodes = _parse_int(data.get("free_nodes"), len(free_node_cores))
    total_cores = _parse_int(data.get("total_cores"), sum(free_node_cores))
    free_cores = _parse_int(data.get("free_cores"), sum(free_node_cores))

    return SchedulerAvailability(
        available=bool(data.get("available", True)),
        source=str(data.get("source", "context")),
        total_nodes=total_nodes,
        free_nodes=free_nodes,
        busy_nodes=_parse_int(data.get("busy_nodes")),
        offline_nodes=_parse_int(data.get("offline_nodes")),
        total_cores=total_cores,
        free_cores=free_cores,
        running_jobs=_parse_int(data.get("running_jobs")),
        queued_jobs=_parse_int(data.get("queued_jobs")),
        free_node_cores=free_node_cores,
        message=str(data.get("message", "")),
        raw_excerpt=str(data.get("raw_excerpt", ""))[:1000],
    )


def get_scheduler_availability(
    context: Optional[Dict[str, Any]] = None,
    node_type: str = None,
    timeout: int = 5,
) -> SchedulerAvailability:
    context = context or {}
    context_availability = _availability_from_context(context)
    if context_availability is not None:
        return context_availability

    node_type = node_type or HARDWARE["node_type"]
    if shutil.which("pbsnodes") is None:
        return SchedulerAvailability(
            available=False,
            source="pbsnodes",
            message="pbsnodes command not found",
        )

    try:
        proc = _run_scheduler_command(["pbsnodes", "-a"], timeout=timeout)
    except Exception as exc:
        return SchedulerAvailability(
            available=False,
            source="pbsnodes",
            message=str(exc),
        )

    raw = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        return SchedulerAvailability(
            available=False,
            source="pbsnodes",
            message=f"pbsnodes returned {proc.returncode}",
            raw_excerpt=raw[:1000],
        )

    availability = _parse_pbsnodes(proc.stdout or "", node_type=node_type)
    if shutil.which("qstat") is not None:
        try:
            qstat_proc = _run_scheduler_command(["qstat"], timeout=timeout)
            if qstat_proc.returncode == 0:
                jobs = _parse_qstat_jobs(qstat_proc.stdout or "")
                availability.running_jobs = jobs["running_jobs"]
                availability.queued_jobs = jobs["queued_jobs"]
        except Exception:
            pass

    return availability


def count_atoms_from_context(context: Dict[str, Any]) -> int:
    try:
        import ase.io
    except ImportError:
        return 0

    candidates = []

    vasp_dir = context.get("vasp_dir") or (context.get("vasp_system") or {}).get("dir")
    if vasp_dir:
        poscar = Path(vasp_dir) / "POSCAR"
        if poscar.exists():
            candidates.append(poscar)

    for key in ("mof_path", "optimized_mof_path", "complex_cif_path", "guest_cif_path"):
        p = context.get(key)
        if p and Path(p).exists():
            candidates.append(Path(p))

    work_dir = context.get("work_dir")
    if work_dir:
        for cif in sorted(Path(work_dir).glob("*.cif")):
            candidates.append(cif)

    for path in candidates:
        try:
            import ase.io
            atoms = ase.io.read(str(path))
            return len(atoms)
        except Exception:
            continue

    return 0


class ResourceAllocator:
    def __init__(self, llm=None):
        self._llm = llm

    def _get_llm(self):
        if self._llm is None:
            from config import LLM_DEFAULT
            self._llm = LLM_DEFAULT
        return self._llm

    def recommend(
        self,
        software: str,
        calc_type: str,
        n_atoms: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> ResourceSpec:
        context = context or {}
        mof = context.get("mof", "unknown")
        guest = context.get("guest", "none")
        availability = get_scheduler_availability(context)
        context["scheduler_availability"] = asdict(availability)
        ppn_choices = ppn_choices_for_software(software)

        try:
            set_llm_context("ResourceAllocator", "resource_allocation")
            resp = self._get_llm().invoke([
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=_USER_PROMPT.format(
                    software=software,
                    calc_type=calc_type,
                    n_atoms=n_atoms if n_atoms > 0 else "unknown",
                    mof=mof,
                    guest=guest,
                    availability_summary=availability.summary(),
                    ppn_choices=ppn_choices,
                )),
            ]).content.strip()

            if resp.startswith("```"):
                lines = resp.splitlines()
                if lines[-1].strip().startswith("```"):
                    lines = lines[1:-1]
                else:
                    lines = lines[1:]
                resp = "\n".join(lines).strip()

            data = json.loads(resp)

            nodes = max(1, min(int(data["nodes"]), HARDWARE["max_nodes"]))
            ppn = int(data["ppn"])
            if ppn not in ppn_choices:
                raise ResourceAllocationError(
                    f"LLM returned invalid ppn={ppn}; expected one of {ppn_choices}"
                )
            np_ = nodes * ppn
            queue = str(data["queue"])
            rationale = str(data.get("rationale", ""))
            if queue != HARDWARE["queue"]:
                raise ResourceAllocationError(
                    f"LLM returned invalid queue={queue!r}; expected {HARDWARE['queue']!r}"
                )
            if int(data["np"]) != np_:
                raise ResourceAllocationError(
                    f"LLM returned np={data['np']} but nodes*ppn={np_}"
                )

            spec = ResourceSpec(nodes=nodes, ppn=ppn, np=np_, queue=queue, rationale=rationale)
            spec = self._fit_to_availability(spec, availability, ppn_choices)
            estimate = estimate_runtime_from_history(software, calc_type, n_atoms, spec, context)
            if estimate:
                context["resource_runtime_estimate"] = estimate
                self._show_interactive_runtime_estimate(estimate)
            spec = self._maybe_apply_interactive_override(
                spec,
                software,
                n_atoms,
                availability,
                ppn_choices,
                context,
            )
            if context.get("resource_allocation_user_override"):
                estimate = estimate_runtime_from_history(software, calc_type, n_atoms, spec, context)
                if estimate:
                    context["resource_runtime_estimate"] = estimate
                    self._show_interactive_runtime_estimate(estimate)
            print(
                f"[ResourceAllocator] {software}/{calc_type} n_atoms={n_atoms} "
                f"→ {spec.pbs_nodes_string()} np={spec.np} | {spec.rationale}"
            )
            try:
                log_llm_decision("ResourceAllocator", "resource_allocation",
                                 {"software": software, "calc_type": calc_type,
                                  "n_atoms": n_atoms, "nodes": spec.nodes, "ppn": spec.ppn,
                                  "queue": spec.queue, "rationale": spec.rationale,
                                  "availability": asdict(availability)},
                                 context)
            except Exception:
                pass
            return spec

        except Exception as e:
            raise ResourceAllocationError(
                f"Prompt-based resource allocation failed for {software}/{calc_type}: {e}"
            ) from e

    def _fit_to_availability(
        self,
        spec: ResourceSpec,
        availability: SchedulerAvailability,
        ppn_choices: List[int],
    ) -> ResourceSpec:
        if not availability.available or availability.free_nodes <= 0:
            return spec

        sorted_ppn_choices = sorted(ppn_choices, reverse=True)
        ppn = spec.ppn
        nodes = min(spec.nodes, HARDWARE["max_nodes"], availability.free_nodes)

        if availability.nodes_with_at_least(ppn) < nodes:
            for candidate_ppn in sorted_ppn_choices:
                candidate_nodes = min(nodes, availability.nodes_with_at_least(candidate_ppn))
                if candidate_nodes >= 1:
                    ppn = candidate_ppn
                    nodes = candidate_nodes
                    break

        nodes = max(1, nodes)
        np_ = nodes * ppn
        if nodes != spec.nodes or ppn != spec.ppn:
            rationale = (
                f"{spec.rationale} Adjusted to current scheduler availability "
                f"({availability.free_nodes} free nodes, {availability.free_cores} free cores)."
            ).strip()
        else:
            rationale = spec.rationale

        return ResourceSpec(
            nodes=nodes,
            ppn=ppn,
            np=np_,
            queue=spec.queue,
            rationale=rationale,
        )

    def _show_interactive_runtime_estimate(self, estimate: Dict[str, Any]) -> None:
        try:
            from config import INTERACTION_MODE
        except Exception:
            return
        if INTERACTION_MODE != "interactive":
            return
        print("\n[ResourceAllocator] Runtime estimate from previous jobs:")
        print(f"  Proposed resources: {estimate['current_np']} cores "
              f"for {estimate['current_n_atoms']} atoms")
        print(f"  Estimated wall time: {estimate['estimated_time']}")
        print(f"  Based on {estimate['sample_count']} completed prior job(s).")
        closest = estimate.get("closest_sample") or {}
        if closest:
            print(
                "  Closest sample: "
                f"{closest.get('n_atoms')} atoms, {closest.get('np')} cores, "
                f"{closest.get('elapsed')}"
            )

    def _maybe_apply_interactive_override(
        self,
        spec: ResourceSpec,
        software: str,
        n_atoms: int,
        availability: SchedulerAvailability,
        ppn_choices: List[int],
        context: Dict[str, Any],
    ) -> ResourceSpec:
        try:
            from config import INTERACTION_MODE
        except Exception:
            return spec
        if INTERACTION_MODE != "interactive":
            return spec

        print("\n[ResourceAllocator] Proposed resource allocation:")
        print(f"  software={software}, n_atoms={n_atoms}")
        print(f"  nodes={spec.nodes}, ppn={spec.ppn}, np={spec.np}, queue={spec.queue}")
        print(f"  allowed ppn choices={ppn_choices}; max_nodes={HARDWARE['max_nodes']}")
        print("  Press Enter to accept, or type a replacement as 'nodes,ppn' (for example: 1,16).")

        user_value = input("[ResourceAllocator] resource allocation: ").strip()
        if not user_value:
            return spec

        try:
            parts = [part.strip() for part in user_value.replace("x", ",").split(",")]
            if len(parts) != 2:
                raise ValueError("expected two values: nodes,ppn")
            nodes = int(parts[0])
            ppn = int(parts[1])
            if nodes < 1 or nodes > HARDWARE["max_nodes"]:
                raise ValueError(f"nodes must be between 1 and {HARDWARE['max_nodes']}")
            if ppn not in ppn_choices:
                raise ValueError(f"ppn must be one of {ppn_choices}")
            if availability.available and availability.free_nodes > 0:
                if nodes > availability.free_nodes:
                    raise ValueError(f"requested {nodes} nodes but only {availability.free_nodes} are free")
                if availability.nodes_with_at_least(ppn) < nodes:
                    raise ValueError(f"not enough free nodes with at least {ppn} cores")
        except Exception as exc:
            print(f"[ResourceAllocator] Invalid override ({exc}). Keeping proposed allocation.")
            return spec

        override = ResourceSpec(
            nodes=nodes,
            ppn=ppn,
            np=nodes * ppn,
            queue=spec.queue,
            rationale=f"{spec.rationale} User override applied in interactive mode.",
        )
        context["resource_allocation_user_override"] = {
            "original": {
                "nodes": spec.nodes,
                "ppn": spec.ppn,
                "np": spec.np,
            },
            "override": {
                "nodes": override.nodes,
                "ppn": override.ppn,
                "np": override.np,
            },
        }
        print(f"[ResourceAllocator] Using user override: nodes={nodes}, ppn={ppn}, np={override.np}")
        return override


def _runtime_history_paths(context: Dict[str, Any]) -> List[Path]:
    explicit = context.get("runtime_history_paths")
    if explicit:
        return [Path(p) for p in explicit]

    try:
        from config import working_dir
        root = Path(working_dir)
    except Exception:
        root = Path(os.getenv("SIMMOF_WORKING_DIR", Path.cwd() / "working_dir"))

    return sorted(root.glob("job_state_*.sqlite3"))


def _load_runtime_samples(context: Dict[str, Any], software: str) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    target = (software or "").upper()

    for db_path in _runtime_history_paths(context):
        if not db_path.exists():
            continue
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT j.plan_name, j.job_id, j.metadata_json AS job_metadata,
                       e.metadata_json AS event_metadata
                FROM jobs j
                JOIN job_events e
                  ON j.plan_name = e.plan_name AND j.job_id = e.job_id
                WHERE e.status = 'done_ok'
                """
            ).fetchall()
        except Exception:
            continue
        finally:
            try:
                conn.close()
            except Exception:
                pass

        for row in rows:
            job_meta = _safe_json_dict(row["job_metadata"])
            event_meta = _safe_json_dict(row["event_metadata"])
            allocation = job_meta.get("resource_allocation") or {}
            sample_software = str(allocation.get("software") or job_meta.get("software") or "").upper()
            if sample_software != target:
                continue
            n_atoms = _safe_int(allocation.get("n_atoms"))
            np_ = _safe_int(allocation.get("np"))
            elapsed = _safe_float(
                event_meta.get("marker_elapsed_sec")
                or event_meta.get("elapsed_sec")
                or job_meta.get("marker_elapsed_sec")
            )
            if n_atoms <= 0 or np_ <= 0 or elapsed <= 0:
                continue
            samples.append({
                "software": sample_software,
                "calc_type": allocation.get("calc_type", ""),
                "n_atoms": n_atoms,
                "np": np_,
                "nodes": _safe_int(allocation.get("nodes")),
                "ppn": _safe_int(allocation.get("ppn")),
                "elapsed_sec": elapsed,
                "db_path": str(db_path),
            })

    return samples


def _safe_json_dict(value: Any) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        data = json.loads(value) if isinstance(value, str) else value
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def estimate_runtime_from_history(
    software: str,
    calc_type: str,
    n_atoms: int,
    spec: ResourceSpec,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    context = context or {}
    if n_atoms <= 0 or spec.np <= 0:
        return None

    samples = _load_runtime_samples(context, software)
    if not samples:
        return None

    estimates = []
    for sample in samples:
        estimated_sec = sample["elapsed_sec"] * (n_atoms / sample["n_atoms"]) * (sample["np"] / spec.np)
        estimates.append({**sample, "estimated_sec": estimated_sec})

    estimates.sort(key=lambda s: (
        abs(s["n_atoms"] - n_atoms) / max(n_atoms, 1),
        abs(s["np"] - spec.np) / max(spec.np, 1),
    ))
    closest = estimates[0]
    estimated_sec = median([sample["estimated_sec"] for sample in estimates[: min(5, len(estimates))]])

    return {
        "software": software,
        "calc_type": calc_type,
        "current_n_atoms": n_atoms,
        "current_np": spec.np,
        "current_nodes": spec.nodes,
        "current_ppn": spec.ppn,
        "estimated_sec": estimated_sec,
        "estimated_time": _format_duration(estimated_sec),
        "sample_count": len(estimates),
        "closest_sample": {
            "n_atoms": closest["n_atoms"],
            "np": closest["np"],
            "nodes": closest["nodes"],
            "ppn": closest["ppn"],
            "elapsed_sec": closest["elapsed_sec"],
            "elapsed": _format_duration(closest["elapsed_sec"]),
        },
    }
