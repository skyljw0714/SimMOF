import json
import os
import asyncio
import inspect

from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional
from pydantic import BaseModel
from langchain.schema import SystemMessage, HumanMessage
from config import LLM_DEFAULT, AGENT_LLM_MAP, working_dir
from pathlib import Path
from collections import defaultdict, deque



class WorkflowJob(BaseModel):
    job_id: str                 
    depends_on: List[str] = []   

class WorkflowPlan(BaseModel):
    job_name: str
    agent: str
    mof: Optional[str] = None
    guest: Optional[str] = None
    property: str
    depends_on_plans: List[str] = []
    jobs: List[WorkflowJob] = []
    query_text: str = ""
        



class WorkingAgent:
    def __init__(
        self,
        parsed_queries: List[dict],
        analysis_enabled: bool = False,
        agents=None,
        simulation_input: Optional[dict] = None,
    ):
        self.parsed_queries = parsed_queries
        self.analysis_enabled = analysis_enabled
        self.plans: List[WorkflowPlan] = []
        self.agents = agents
        self.simulation_input = simulation_input or {"present": False, "snippets": []}
    
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
    - VASPAgent: DFT energy, band structure, binding energy.
    - LAMMPSAgent: diffusion coefficient, mean squared distance.

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

- NEVER output agent="RagAgent".
- Do NOT create any separate plan for patching/reproducing inputs; reproduction is handled inside the target simulation agent using context["simulation_input"].
- For multi-MOF queries, create per-MOF plans and exactly ONE final_response plan.
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

VASP workflows:
- VASP binding energy workflows MUST include three jobs (mof, guest, complex), ordered by dependency.
- For VASP binding_energy: "<job_name>_complex" depends_on MUST be ["<job_name>_mof"] only.
- DO NOT add "<job_name>_guest" to depends_on for the complex job. Guest runs independently.
- Bader charge workflows include only mof and complex jobs.
- bader_charge always depends_on_plans the matching binding_energy plan (same MOF/guest) and has no job-level depends_on.

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
        "depends_on": ["string", ...],
        "steps": [
          {"order": 1, "agent": "string"},
          {"order": 2, "agent": "string"},
        ]
      }
    ]
  },
  ...
]
    """
    
        examples = """
    Example A (single MOF Zeo++ surface area):
    Parsed queries:
    [
    {"Name":"UiO-66-SA","Agent":"ZeoppAgent","Property":"surface_area","MOF":"UiO-66","Guest":null}
    ]
    Output:
    [
    {
        "job_name": "UiO-66_surface_area",
        "agent": "ZeoppAgent",
        "mof": "UiO-66",
        "guest": null,
        "property": "surface_area",
        "depends_on_plans": [],
        "jobs": [
        {
            "job_id": "UiO-66_surface_area_job",
            "depends_on": [],
        }
        ]
    },
    {
        "job_name": "final_response",
        "agent": "ResponseAgent",
        "mof": "UiO-66",
        "guest": null,
        "property": "surface_area",
        "depends_on_plans": ["UiO-66_surface_area"],
        "jobs": [
        {
            "job_id": "final_response_1",
            "depends_on": [],
        }
        ]
    }
    ]

    Example B (HKUST-1 + CO2 binding energy, 2 plans)
    Parsed queries:
    [
    {
        "Name": "HKUST-1-CO2-binding_energy",
        "Agent": "VASPAgent",
        "Property": "binding_energy",
        "MOF": "HKUST-1",
        "Guest": "CO2"
    }
    ]

    Output:
    [
    {
        "job_name": "HKUST-1_CO2_binding_energy",
        "agent": "VASPAgent",
        "mof": "HKUST-1",
        "guest": "CO2",
        "property": "binding_energy",
        "depends_on_plans": [],
        "jobs": [
        {
            "job_id": "HKUST-1_CO2_binding_energy_mof",
            "depends_on": [],
        },
        {
            "job_id": "HKUST-1_CO2_binding_energy_guest",
            "depends_on": [],
        },
        {
            "job_id": "HKUST-1_CO2_binding_energy_complex",
            "depends_on": ["HKUST-1_CO2_binding_energy_mof"],
        }
        ]
    },
    {
        "job_name": "final_response",
        "agent": "ResponseAgent",
        "mof": "HKUST-1",
        "guest": "CO2",
        "property": "binding_energy",
        "depends_on_plans": ["HKUST-1_CO2_vasp"],
        "jobs": [
        {
            "job_id": "final_response_job",
            "depends_on": [],
        }
        ]
    }
    ]

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


    def plan(self) -> List[WorkflowPlan]:
        prompt = self._build_planner_prompt()

        llm_for_planner = AGENT_LLM_MAP.get("WorkingAgent", LLM_DEFAULT)

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

        for p in data:
            if isinstance(p, dict):
                p.setdefault("query_text", query_text)

        self.plans = [WorkflowPlan(**p) for p in data]
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
        out = debug_dir / f"context_{when}_{agent_name}_{job_id}.json"

        try:
            with open(out, "w", encoding="utf-8") as f:
                json.dump(ctx, f, indent=2, ensure_ascii=False)
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

        plan_root = root_dir / plan.job_name
        plan_root.mkdir(parents=True, exist_ok=True)

        work_dir = plan_root

        ctx = {
            "plan_name": plan.job_name,
            "job_name": plan.job_name,
            "job_id": job.job_id,
            "agent": getattr(plan, "agent", None),
            "mof": getattr(plan, "mof", None),
            "guest": getattr(plan, "guest", None),
            "property": getattr(plan, "property", None),
            "query_text": getattr(plan, "query_text", ""),
            "simulation_input": self.simulation_input,
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

            self._dump_context_job(ctx, agent_name=agent_name, when="pre")

            
            async with global_sem:
                
                if inspect.iscoroutinefunction(agent.run):
                    out = await agent.run(ctx)
                else:
                    out = await asyncio.to_thread(agent.run, ctx)

            
            if isinstance(out, dict):
                ctx.update(out)
            
            self._dump_context_job(ctx, agent_name=agent_name, when="post")

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




if __name__ == "__main__":
    pass