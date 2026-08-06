import json
import os
import asyncio
import inspect
import re

from core.timing import timer
from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional
from pydantic import BaseModel
from langchain.schema import SystemMessage, HumanMessage
from config import LLM_DEFAULT, AGENT_LLM_MAP, working_dir
from core.job_manager import record_job_event
from core.llm_logging import log_llm_decision, set_llm_context
from pathlib import Path
from collections import Counter, defaultdict, deque
from core.databases import resolve_cif_dir
from core.simulation_contracts import canonical_property



class WorkflowJob(BaseModel):
    job_id: str                 
    depends_on: List[str] = []   

class WorkflowPlan(BaseModel):
    job_name: str
    agent: str
    mof: Optional[str] = None
    guest: Optional[str] = None
    cif_path: Optional[str] = None
    cif_dir: Optional[str] = None
    hmof_params: Optional[dict] = None
    metal_filter: Optional[List[str]] = None
    property: str
    depends_on_plans: List[str] = []
    jobs: List[WorkflowJob] = []
    query_text: str = ""


def _model_dump(value: BaseModel) -> dict:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value.dict()



class WorkingAgent:
    def __init__(
        self,
        parsed_queries: List[dict],
        analysis_enabled: bool = False,
        agents=None,
        simulation_input: Optional[dict] = None,
        analysis_recommendation: Optional[dict] = None,
        semantic_guardrails: bool = False,
    ):
        self.parsed_queries = parsed_queries
        self.analysis_enabled = analysis_enabled
        self.plans: List[WorkflowPlan] = []
        self.agents = agents
        self.simulation_input = simulation_input or {"present": False, "snippets": []}
        self.analysis_recommendation = analysis_recommendation or {}
        self.analysis_dependency_warnings: List[str] = []
        self.semantic_guardrails = semantic_guardrails
    
    def _build_planner_prompt(self) -> str:
        queries_json = json.dumps(self.parsed_queries, ensure_ascii=False, indent=2)


        agent_desc = """
    [Available Agents]

    Task management:
    - QueryAgent: already executed; parses user query.
    - WorkingAgent: you are this agent; you plan workflows.
    - ResponseAgent: generates human-readable answers.

    Simulation:
    - ZeoppAgent: surface area, pore volume.
    - RASPAAgent: Henry coefficient, gas uptake.
    - VASPAgent: DFT energy, band structure, binding energy, Bader charge, projected DOS.
    - LAMMPSAgent: diffusion coefficient, mean squared displacement, thermal expansion.

    Assistance:
    - AnalysisAgent: analyze outputs and run derived calculations.
    - ScreeningAgent: determine which tools to use for screening.
    - RagAgent: extract data from text and figures, identify key parameters.
    """

        schema = """
    You MUST return ONLY a JSON array of WorkflowPlan objects.

    WorkflowJob schema:
    {
    "job_id": "string",                   // e.g., "HKUST-1_binding_energy"
    "depends_on": ["string", ...],         // job-level dependency within the same plan
    }

    WorkflowPlan schema:
    {
    "job_name": "string",                 // unique plan identifier (used for plan-level dependencies)
    "agent": "string",                   // main agent for this plan
    "mof": "string",
    "guest": "string or null",
    "property": "string",
    "jobs": [WorkflowJob, ...],
    "depends_on_plans": ["string", ...],   // plan-level dependency (can be empty)
    }
    """

        planning_rules = """
Planning rules:

- CLOSED-WORLD HANDOFF CONTRACT: parsed queries are the complete and
  authoritative simulation inventory.
- Create exactly ONE non-final WorkflowPlan for EACH parsed-query element and
  create NO other non-final plan.
- Copy agent, MOF, guest, property, CIF path/directory, hMOF parameters, and
  metal filter from that parsed query. Do not substitute a related property.
- Do not add supporting simulations for interpretation or analysis. An
  analysis recommendation may consume the requested results, but it cannot
  expand the simulation inventory.
- NEVER output agent="RagAgent".
- Do NOT create any separate plan for patching/reproducing inputs; reproduction is handled inside the target simulation agent using context["simulation_input"].
- Parsed queries already express multi-MOF and multi-condition cardinality; do
  not expand or merge them.
- If MOF="database" appears in parsed queries, keep mof="database" in the plan as-is (do NOT expand to per-MOF plans). The batch iteration over CIFs is handled automatically at runtime.
- If MOF="hmof" appears in parsed queries, keep mof="hmof" in the plan as-is. hMOF CIF generation and simulation are handled automatically at runtime.
- Exactly ONE ResponseAgent is allowed and it MUST appear only in the final_response plan.

Final plans:
- If analysis_enabled is true, create exactly TWO final plans:
  1) Plan name: "final_analysis", agent: "AnalysisAgent"
     - jobs: [{"job_id":"final_analysis_job","depends_on": []}]
     - depends_on_plans: all non-final plans
  2) Plan name: "final_response", agent: "ResponseAgent"
     - jobs: [{"job_id":"final_response_job","depends_on": []}]
     - depends_on_plans: ["final_analysis"]
- If analysis_enabled is false, create exactly ONE final plan:
  - Plan name: "final_response", agent: "ResponseAgent"
    - jobs: [{"job_id":"final_response_job","depends_on": []}]
    - depends_on_plans: all non-final plans

Job naming:
- job_id MUST be globally unique and follow the format "<job_name>_job".
- Do NOT use semantic role names (e.g., run, mof, guest, complex) as job_id.
- Exception (VASP binding_energy): job_id MUST be:
  "<job_name>_mof", "<job_name>_guest", "<job_name>_complex".
- Exception (VASP bader_charge): job_id MUST be:
  "<job_name>_mof", "<job_name>_complex". (no guest)
- Exception (VASP projected_dos): job_id MUST be:
  "<job_name>_complex".

VASP workflows:
- VASP binding energy workflows MUST include three jobs (mof, guest, complex), ordered by dependency.
- For VASP binding_energy: "<job_name>_complex" depends_on MUST be ["<job_name>_mof"] only.
- DO NOT add "<job_name>_guest" to depends_on for the complex job. Guest runs independently.
- Bader charge workflows include only mof and complex jobs.
- bader_charge always depends_on_plans the matching binding_energy plan (same MOF/guest) and has no job-level depends_on.
- Projected DOS workflows include one static complex job.
- projected_dos always depends_on_plans the matching binding_energy plan (same MOF/guest) and reuses its optimized structures and restart files.

Screening → Simulation dependency rule:
- If a parsed query includes a ScreeningAgent plan for a MOF set (e.g., MOF="database" or multiple MOFs),
  AND there is any downstream simulation/analysis plan on the same MOF set (ZeoppAgent, RASPAAgent, VASPAgent, LAMMPSAgent),
  then the downstream plan MUST depend on the screening plan via depends_on_plans.

Property rule:
- WorkflowPlan.property MUST be a single property string (no commas, no lists).
- If multiple properties are requested (e.g., pore_volume and pore_limiting_diameter),
  create separate WorkflowPlans for each property (or use a single canonical combined property name only if it exists in ALLOWED_METHODS).

Dependencies (CRITICAL):
- Do NOT infer dependencies from textual order ("then", "after", etc.).
- Add depends_on ONLY when downstream tasks require explicit outputs/files from upstream tasks. (ex. binding energy of MOF + guest requires MOF DFT calculations)
- Independent computations using the same input structure MUST run in parallel.
- Use job-level depends_on for intra-plan prerequisites.
- Use plan-level depends_on_plans for cross-plan dependencies.

Example:
- Zeo++ pore analysis and LAMMPS diffusivity are independent and MUST NOT depend on each other.

Return ONLY a JSON array following this schema:
[
  {
    "job_name": "string",
    "agent": "string",
    "mof": "string",
    "guest": "string or null",
    "property": "string",
    "depends_on_plans": ["string", ...],
    "jobs": [
      {
        "job_id": "string",
        "depends_on": ["string", ...]
      }
    ]
  },
  ...
]
    """
    
        examples = """
    Before returning, compare the multiset of non-final plans to the parsed
    queries using (agent, property, MOF, guest). The two multisets must match
    exactly, including repeated queries for distinct conditions.
    """

        return f"""{agent_desc}
    {schema}

    You are the WorkingAgent.
    Given the parsed queries below, design WorkflowPlans.

    analysis_enabled: {self.analysis_enabled}

    {planning_rules}

    {examples}

    Parsed queries:
    {queries_json}
    """

    @staticmethod
    def _plan_identity(
        agent: str,
        mof: Optional[str],
        guest: Optional[str],
        property_name: str,
    ) -> tuple:
        property_name = canonical_property(property_name)
        normalized_guest = None
        if (
            agent != "ZeoppAgent"
            and property_name not in {"thermal_expansion", "youngs_modulus"}
        ):
            raw_guest = (
                str(guest or "")
                .upper()
                .replace("CO₂", "CO2")
                .replace("CH₄", "CH4")
                .replace("N₂", "N2")
                .replace("H₂", "H2")
            )
            normalized_guest = "/".join(
                sorted(part for part in re.split(r"[/,+\s]+", raw_guest) if part)
            )
        return (
            str(agent or ""),
            re.sub(r"[^a-z0-9]", "", str(mof or "").lower()),
            str(normalized_guest or ""),
            str(property_name or ""),
        )

    @staticmethod
    def _analysis_plan_name(
        mof: Optional[str],
        guest: Optional[str],
        method: str,
    ) -> str:
        parts = [str(mof or "unknown")]
        if guest:
            parts.append(str(guest))
        parts.append(str(method))
        raw = "_".join(parts)
        return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw)

    @staticmethod
    def _jobs_for_analysis_dependency(
        plan_name: str,
        method: str,
    ) -> List[WorkflowJob]:
        if method == "binding_energy":
            return [
                WorkflowJob(job_id=f"{plan_name}_mof", depends_on=[]),
                WorkflowJob(job_id=f"{plan_name}_guest", depends_on=[]),
                WorkflowJob(
                    job_id=f"{plan_name}_complex",
                    depends_on=[f"{plan_name}_mof"],
                ),
            ]
        if method == "bader_charge":
            return [
                WorkflowJob(job_id=f"{plan_name}_mof", depends_on=[]),
                WorkflowJob(job_id=f"{plan_name}_complex", depends_on=[]),
            ]
        if method == "projected_dos":
            return [
                WorkflowJob(job_id=f"{plan_name}_complex", depends_on=[]),
            ]
        return [WorkflowJob(job_id=f"{plan_name}_job", depends_on=[])]

    def _analysis_targets(self) -> List[Dict[str, Any]]:
        targets_by_mof: Dict[tuple, Dict[str, Any]] = {}
        for query in self.parsed_queries:
            if not isinstance(query, dict):
                continue
            key = (
                query.get("MOF"),
                query.get("CIFPath"),
                query.get("CIFDir"),
            )
            target = targets_by_mof.setdefault(
                key,
                {**query, "_analysis_guests": []},
            )
            guest = query.get("Guest")
            if guest and guest not in target["_analysis_guests"]:
                target["_analysis_guests"].append(guest)
        return list(targets_by_mof.values())

    @staticmethod
    def _analysis_request_requires_guest(agent: str, method: str) -> bool:
        if agent == "RASPAAgent":
            return True
        return method in {
            "binding_energy",
            "bader_charge",
            "projected_dos",
            "msd",
            "diffusivity",
        }

    def _ensure_analysis_dependency_plans(
        self,
        plans: List[WorkflowPlan],
    ) -> List[WorkflowPlan]:
        final_plans = [
            plan
            for plan in plans
            if plan.agent in {"AnalysisAgent", "ResponseAgent"}
        ]
        remaining = self._expected_simulation_counter()
        simulation_plans: List[WorkflowPlan] = []
        for plan in plans:
            if plan.agent in {"AnalysisAgent", "ResponseAgent"}:
                continue
            plan.property = canonical_property(plan.property)
            identity = self._plan_identity(
                plan.agent,
                plan.mof,
                plan.guest,
                plan.property,
            )
            if remaining[identity] <= 0:
                continue
            remaining[identity] -= 1
            simulation_plans.append(plan)

        dependency_names = [plan.job_name for plan in simulation_plans]
        for plan in final_plans:
            if plan.agent == "AnalysisAgent":
                plan.depends_on_plans = dependency_names
            elif plan.agent == "ResponseAgent":
                plan.depends_on_plans = (
                    ["final_analysis"] if self.analysis_enabled else dependency_names
                )

        return [*simulation_plans, *final_plans]

    def _expected_simulation_counter(self) -> Counter:
        return Counter(
            self._plan_identity(
                str(query.get("Agent") or ""),
                query.get("MOF"),
                query.get("Guest"),
                str(query.get("Property") or ""),
            )
            for query in self.parsed_queries
            if isinstance(query, dict)
        )

    def _actual_simulation_counter(
        self,
        plans: List[WorkflowPlan],
    ) -> Counter:
        return Counter(
            self._plan_identity(
                plan.agent,
                plan.mof,
                plan.guest,
                plan.property,
            )
            for plan in plans
            if plan.agent not in {"AnalysisAgent", "ResponseAgent"}
        )

    def _simulation_contract_matches(
        self,
        plans: List[WorkflowPlan],
    ) -> bool:
        return self._actual_simulation_counter(plans) == self._expected_simulation_counter()

    def _final_contract_matches(self, plans: List[WorkflowPlan]) -> bool:
        final = [plan for plan in plans if plan.agent in {"AnalysisAgent", "ResponseAgent"}]
        expected = (
            [("final_analysis", "AnalysisAgent"), ("final_response", "ResponseAgent")]
            if self.analysis_enabled
            else [("final_response", "ResponseAgent")]
        )
        return [(plan.job_name, plan.agent) for plan in final] == expected


    def plan(self) -> List[WorkflowPlan]:
        prompt = self._build_planner_prompt()

        llm_for_planner = AGENT_LLM_MAP.get("WorkingAgent", LLM_DEFAULT)

        set_llm_context("WorkingAgent", "workflow_planning")
        resp = llm_for_planner.invoke([
            SystemMessage(content="You are the WorkingAgent for MOF simulations."),
            HumanMessage(content=prompt),
        ])

        text = resp.content.strip()
        if text.startswith("```"):
            text = "\n".join(text.splitlines()[1:-1]).strip()

        data = json.loads(text)

        query_text = ""
        if self.parsed_queries and isinstance(self.parsed_queries, list):
            first = self.parsed_queries[0]
            if isinstance(first, dict):
                query_text = first.get("QueryText", "") or first.get("query_text", "") or ""

        cif_paths_by_mof = {}
        db_cif_dir = None
        hmof_params = None
        metal_filter = None
        for q in self.parsed_queries:
            if isinstance(q, dict):
                if q.get("CIFPath"):
                    cif_paths_by_mof[q.get("MOF")] = q["CIFPath"]
                if q.get("CIFDir") and not db_cif_dir:
                    db_cif_dir = q["CIFDir"]
                if q.get("HMOFParams") and not hmof_params:
                    hmof_params = q["HMOFParams"]
                if q.get("MetalFilter") and not metal_filter:
                    metal_filter = q["MetalFilter"]

        def _materialize(raw_data: Any) -> List[WorkflowPlan]:
            if not isinstance(raw_data, list):
                raise ValueError("WorkingAgent output must be a JSON array.")
            prepared = []
            for item in raw_data:
                if not isinstance(item, dict):
                    continue
                p = dict(item)
                p.setdefault("query_text", query_text)
                if not p.get("cif_path"):
                    p["cif_path"] = cif_paths_by_mof.get(p.get("mof"))
                if not p.get("cif_dir") and p.get("mof") == "database" and db_cif_dir:
                    p["cif_dir"] = db_cif_dir
                if not p.get("hmof_params") and p.get("mof") == "hmof" and hmof_params:
                    p["hmof_params"] = hmof_params
                if not p.get("metal_filter") and metal_filter:
                    p["metal_filter"] = metal_filter
                prepared.append(WorkflowPlan(**p))
            if self.semantic_guardrails:
                return self._ensure_analysis_dependency_plans(prepared)
            return prepared

        self.plans = _materialize(data)
        if self.semantic_guardrails and not (
            self._simulation_contract_matches(self.plans)
            and self._final_contract_matches(self.plans)
        ):
            repair_prompt = f"""
{prompt}

The previous response violated the closed-world handoff contract.
Repair it from scratch.

Success criteria:
- The multiset of non-final plans must equal the parsed-query multiset exactly.
- Preserve duplicate parsed queries because they represent distinct conditions.
- Add no simulation from analysis recommendations or scientific inference.
- Return the required final plan or plans exactly as specified.
- Return ONLY the corrected JSON array.
""".strip()
            set_llm_context("WorkingAgent", "workflow_planning_contract_repair")
            repair_resp = llm_for_planner.invoke(
                [
                    SystemMessage(
                        content=(
                            "You repair a MOF workflow plan to satisfy an exact "
                            "parsed-query handoff contract."
                        )
                    ),
                    HumanMessage(content=repair_prompt),
                ]
            )
            repair_text = repair_resp.content.strip()
            if repair_text.startswith("```"):
                repair_text = "\n".join(repair_text.splitlines()[1:-1]).strip()
            repaired = _materialize(json.loads(repair_text))
            if not (
                self._simulation_contract_matches(repaired)
                and self._final_contract_matches(repaired)
            ):
                raise ValueError(
                    "WorkingAgent could not satisfy the parsed-query handoff contract."
                )
            self.plans = repaired
        try:
            log_llm_decision("WorkingAgent", "workflow_planning",
                             [_model_dump(p) for p in self.plans])
        except Exception:
            pass

        from config import ask_user_confirmation

        plan_summary = "\n".join(
            f"  Plan '{p.job_name}' [{p.agent}]: jobs={[j.job_id for j in p.jobs]}"
            for p in self.plans
        )
        print(f"\n[WorkingAgent] Proposed simulation plan:\n{plan_summary}")

        def _reinvoke_working(instruction: str) -> str:
            revised_prompt = prompt + f"\n\nUser instruction: {instruction}\nRevise your plan accordingly."
            set_llm_context("WorkingAgent", "workflow_planning_revision")
            r = llm_for_planner.invoke([
                SystemMessage(content="You are the WorkingAgent for MOF simulations."),
                HumanMessage(content=revised_prompt),
            ])
            return r.content.strip()

        action, revised_text = ask_user_confirmation(
            "WorkingAgent", plan_summary, reinvoke_fn=_reinvoke_working, required=True
        )
        if action == "apply" and revised_text != plan_summary:
            try:
                t = revised_text
                if t.startswith("```"):
                    t = "\n".join(t.splitlines()[1:-1]).strip()
                data2 = json.loads(t)
                revised_plans = _materialize(data2)
                if (
                    self._simulation_contract_matches(revised_plans)
                    and self._final_contract_matches(revised_plans)
                ):
                    self.plans = revised_plans
                    print("[WorkingAgent] Plan updated per user instruction.")
            except Exception:
                pass

        return self.plans

    def _dump_context_job(self, ctx: Dict[str, Any], agent_name: str, when: str):
        
        work_dir = ctx.get("work_dir")
        if work_dir:
            base = Path(work_dir)
        else:
            
            
            plan_name = ctx.get("plan_name") or ctx.get("job_name") or "unknown_plan"
            base = Path(working_dir) / plan_name

        debug_dir = base / "_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        job_id = ctx.get("job_id", "unknown_job")
        out = debug_dir / f"context_{when}_{agent_name}_{job_id}_{os.getpid()}.json"

        try:
            with open(out, "w", encoding="utf-8") as f:
                json.dump(ctx, f, indent=2, ensure_ascii=False)
            ctx["latest_context_path"] = str(out)
        except Exception as e:
            print(f"[WorkingAgent] Warning: context dump failed: {e}")


    def _build_job_ctx(self, plan, job, results_by_plan):
        upstream_jobs = {}
        for dep in getattr(job, "depends_on", []) or []:
            upstream_jobs[dep] = results_by_plan.get(plan.job_name, {}).get(dep)

        upstream_plans = {}
        for dep_plan in getattr(plan, "depends_on_plans", []) or []:
            upstream_plans[dep_plan] = results_by_plan.get(dep_plan, {})

        root_dir = Path(working_dir)
        root_dir.mkdir(parents=True, exist_ok=True)

        plan_root = root_dir / f"{plan.job_name}_{os.getpid()}"
        plan_root.mkdir(parents=True, exist_ok=True)

        work_dir = plan_root

        raw_cif_dir = getattr(plan, "cif_dir", None)
        db_key_resolved = None
        if raw_cif_dir:
            resolved = resolve_cif_dir(raw_cif_dir)
            if resolved:
                db_key_resolved = raw_cif_dir
                cif_dir = resolved
            else:
                cif_dir = raw_cif_dir
        else:
            cif_dir = None

        plan_metal_filter = getattr(plan, "metal_filter", None)
        if cif_dir and plan_metal_filter:
            from core.mof_filter import apply_metal_filter
            cif_dir = apply_metal_filter(
                cif_dir=cif_dir,
                metals=plan_metal_filter,
                db_key=db_key_resolved,
            )

        ctx = {
            "plan_name": plan.job_name,
            "job_name": plan.job_name,
            "job_id": job.job_id,
            "agent": getattr(plan, "agent", None),
            "mof": getattr(plan, "mof", None),
            "guest": getattr(plan, "guest", None),
            "cif_path": getattr(plan, "cif_path", None),
            "cif_dir": cif_dir,
            "hmof_params": getattr(plan, "hmof_params", None),
            "metal_filter": getattr(plan, "metal_filter", None),
            "property": getattr(plan, "property", None),
            "query_text": getattr(plan, "query_text", ""),
            "simulation_input": self.simulation_input,
            "analysis_recommendation": self.analysis_recommendation,
            "results": {},
            "upstream_jobs": upstream_jobs,
            "upstream_plans": upstream_plans,
            "plan_root": str(plan_root),
            "work_dir": str(work_dir),
            "paths": {
                "root": str(root_dir),
            },
        }

        if ctx["agent"] == "VASPAgent":
            jid = job.job_id
            if jid.endswith("_mof"):
                role = "mof"
            elif jid.endswith("_guest"):
                role = "guest"
            elif jid.endswith("_complex"):
                role = "complex"
            else:
                role = "job"

            vasp_dir = plan_root / "vasp" / role
            vasp_dir.mkdir(parents=True, exist_ok=True)

            ctx["vasp_role"] = role
            ctx["vasp_dir"] = str(vasp_dir)
            ctx["vasp_label"] = jid
            ctx["vasp_system"] = {"dir": str(vasp_dir), "label": jid}
            ctx["paths"].setdefault("vasp", {})
            ctx["paths"]["vasp"]["run_dir"] = str(vasp_dir)

        if ctx["agent"] == "AnalysisAgent":
            ctx["interpret_only"] = True
            analysis_plan = self.analysis_recommendation.get("analysis_plan", {}) or {}
            ctx["analysis_requested_methods"] = [
                step.get("method")
                for step in analysis_plan.get("steps", [])
                if isinstance(step, dict) and step.get("method")
            ]

        return ctx




    def run(self,
            max_concurrency: int = 4,
            per_agent_limits: Optional[Dict[str, int]] = None
            ):
        return asyncio.run(self.run_async(max_concurrency=max_concurrency,
                                          per_agent_limits=per_agent_limits))

    async def run_async(
        self,
        max_concurrency: int = 4,
        per_agent_limits: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        if per_agent_limits is None:
            per_agent_limits = {}

        if not getattr(self, "plans", None):
            self.plans = self.plan()

        global_sem = asyncio.Semaphore(max_concurrency)
        agent_sems = {k: asyncio.Semaphore(v) for k, v in per_agent_limits.items()}

        results_by_plan: Dict[str, Dict[str, Any]] = defaultdict(dict)

        
        plan_map = {p.job_name: p for p in self.plans}

        
        plan_tasks: Dict[str, asyncio.Task] = {}

        async def run_one_job(plan, job):
            agent_name = getattr(plan, "agent", None)
            if not agent_name:
                raise ValueError(f"[{plan.job_name}] plan.agent is missing")

            agent = self.agents.get(agent_name)
            if agent is None:
                raise ValueError(f"Unknown agent: {agent_name}")

            
            ctx = self._build_job_ctx(plan, job, results_by_plan)
            record_job_event(ctx, "created", message="job context created")

            self._dump_context_job(ctx, agent_name=agent_name, when="pre")
            record_job_event(ctx, "running", message=f"{agent_name}.run started")

            try:
                async with global_sem:
                    with timer(
                        f"{agent_name}.run",
                        category="workflow_agent",
                        context=ctx,
                        extra={
                            "plan_agent": agent_name,
                        },
                    ):
                        _NON_BATCH_AGENTS = {"ScreeningAgent", "ResponseAgent", "AnalysisAgent", "RagAgent"}
                        is_hmof = (
                            ctx.get("mof") == "hmof"
                            and agent_name not in _NON_BATCH_AGENTS
                            and ctx.get("hmof_params")
                        )
                        is_batch = (
                            ctx.get("mof") == "database"
                            and agent_name not in _NON_BATCH_AGENTS
                            and ctx.get("cif_dir")
                        )
                        if is_hmof:
                            out = await asyncio.to_thread(_run_hmof_job, agent, ctx)
                        elif is_batch:
                            from batch.workflow import BatchWorkflow
                            out = await asyncio.to_thread(BatchWorkflow(agent).run, ctx)
                        elif inspect.iscoroutinefunction(agent.run):
                            out = await agent.run(ctx)
                        else:
                            out = await asyncio.to_thread(agent.run, ctx)
            except Exception as e:
                record_job_event(ctx, "failed", message=f"{agent_name}.run raised", last_error=str(e))
                raise

            
            if isinstance(out, dict):
                ctx.update(out)
            
            self._dump_context_job(ctx, agent_name=agent_name, when="post")
            record_job_event(ctx, "completed", message=f"{agent_name}.run completed")

            results_by_plan[plan.job_name][job.job_id] = ctx
            return ctx

        async def run_plan(plan_name: str):
            plan = plan_map[plan_name]

            
            for dep_plan_name in getattr(plan, "depends_on_plans", []) or []:
                dep_task = plan_tasks.get(dep_plan_name)
                if dep_task is None:
                    raise ValueError(f"Unknown depends_on_plans: {dep_plan_name}")
                await dep_task  

            
            await run_jobs_in_plan(plan, run_one_job)

        async def run_jobs_in_plan(plan, job_runner):
            jobs = list(plan.jobs)

            
            job_by_id = {j.job_id: j for j in jobs}

            
            indeg = {j.job_id: 0 for j in jobs}
            children = {j.job_id: [] for j in jobs}

            for j in jobs:
                for dep in getattr(j, "depends_on", []) or []:
                    if dep not in job_by_id:
                        raise ValueError(
                            f"[{plan.job_name}] job {j.job_id} depends_on unknown job_id: {dep}"
                        )
                    indeg[j.job_id] += 1
                    children[dep].append(j.job_id)

            
            ready = deque([jid for jid, d in indeg.items() if d == 0])

            running: Dict[str, asyncio.Task] = {}
            done: Set[str] = set()

            
            while ready or running:
                
                while ready:
                    jid = ready.popleft()
                    if jid in done or jid in running:
                        continue
                    task = asyncio.create_task(job_runner(plan, job_by_id[jid]))
                    running[jid] = task

                
                if not running:
                    break

                finished, _ = await asyncio.wait(
                    running.values(), return_when=asyncio.FIRST_COMPLETED
                )

                
                finished_ids = []
                for jid, t in running.items():
                    if t in finished:
                        finished_ids.append(jid)

                for jid in finished_ids:
                    
                    await running[jid]
                    del running[jid]
                    done.add(jid)

                    for ch in children[jid]:
                        indeg[ch] -= 1
                        if indeg[ch] == 0:
                            ready.append(ch)

            
            if len(done) != len(jobs):
                remaining = [jid for jid in indeg if jid not in done]
                raise RuntimeError(
                    f"[{plan.job_name}] not all jobs finished. Remaining: {remaining}"
                )

        
        for plan_name in plan_map:
            
            plan_tasks[plan_name] = asyncio.create_task(run_plan(plan_name))

        
        await asyncio.gather(*plan_tasks.values())

        return dict(results_by_plan)




def _run_hmof_job(agent, ctx: Dict[str, Any]) -> Dict[str, Any]:
    from pathlib import Path
    from structure.agent import StructureAgent

    params = ctx["hmof_params"]
    hmof_type = params.get("type", "random")
    n_mofs = int(params.get("n_mofs", 1))

    hmof_dir = Path(ctx["work_dir"]) / "hmof_cifs"
    hmof_dir.mkdir(parents=True, exist_ok=True)

    sa = StructureAgent()

    if hmof_type == "custom":
        out_path = hmof_dir / "hmof_custom.cif"
        cif = sa.make_custom_hmof(
            topology_name=params["topology"],
            node_bbs=params.get("nodes", {}),
            edge_bbs=params.get("edge_bbs"),
            out_path=str(out_path),
            optimize=params.get("optimize", False),
        )
        if n_mofs == 1:
            single_ctx = dict(ctx)
            single_ctx["mof"] = Path(cif).stem
            single_ctx["mof_path"] = str(cif)
            single_ctx["cif_path"] = str(cif)
            return agent.run(single_ctx)
        cif_paths = [cif]
        for i in range(1, n_mofs):
            out_i = hmof_dir / f"hmof_custom_{i}.cif"
            import shutil
            shutil.copy2(cif, out_i)
            cif_paths.append(out_i)
    else:
        cif_paths = sa.make_random_hmof(
            n_mofs=n_mofs,
            save_dir=str(hmof_dir),
            max_atoms=params.get("max_atoms", 1500),
            min_cell=params.get("min_cell", 4.5),
            max_cell=params.get("max_cell", 60.0),
            random_seed=params.get("random_seed"),
            optimize=params.get("optimize", False),
        )

    if n_mofs == 1 and cif_paths:
        single_ctx = dict(ctx)
        single_ctx["mof"] = Path(cif_paths[0]).stem
        single_ctx["mof_path"] = str(cif_paths[0])
        single_ctx["cif_path"] = str(cif_paths[0])
        return agent.run(single_ctx)

    batch_ctx = dict(ctx)
    batch_ctx["mof"] = "database"
    batch_ctx["cif_dir"] = str(hmof_dir)
    from batch.workflow import BatchWorkflow
    return BatchWorkflow(agent).run(batch_ctx)


if __name__ == "__main__":
    pass
