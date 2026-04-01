from __future__ import annotations

import json
import re
import time
import subprocess
from pathlib import Path
import numpy as np
from ase.io import read
from typing import Any, Dict, List, Optional, Union, Sequence, Tuple

from pydantic import BaseModel, Field, ValidationError
from langchain.schema import SystemMessage, HumanMessage
from config import LLM_DEFAULT, AGENT_LLM_MAP





class ExplanationGoalModel(BaseModel):
    goal: str = Field(...)


class HypothesisModel(BaseModel):
    hypothesis: str = Field(...)


class PlanStepModel(BaseModel):
    name: str
    method: str

class SimulationPlanModel(BaseModel):
    steps: List[PlanStepModel]


class InterpretationModel(BaseModel):
    summary: str
    key_findings: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)
    next_best_step: str = ""





ANALYSIS_METHODS: Dict[str, Dict[str, Any]] = {
    
    "bader_charge": {"engine": "VASP"},
    "binding_energy": {"engine": "VASP"},

    
    "henry_coefficient": {"engine": "RASPA"},
    "heat_of_adsorption": {"engine": "RASPA"},
    "uptake": {"engine": "RASPA"},
    "selectivity": {"engine": "RASPA"},

    
    "pore_size_distribution": {"engine": "Zeo++"},
    "pore_limiting_diameter": {"engine": "Zeo++"},
    "largest_cavity_diameter": {"engine": "Zeo++"},
    "pore_volume": {"engine": "Zeo++"},
    "surface_area": {"engine": "Zeo++"},

    
    "msd": {"engine": "LAMMPS"},
    "diffusivity": {"engine": "LAMMPS"},
}

ALLOWED_METHODS = list(ANALYSIS_METHODS.keys())

DEFAULT_SYSTEM = """You are an expert computational chemistry assistant.
You must respond in STRICT JSON only (no markdown, no commentary).
Do NOT invent results.
If you need numeric values, only use those explicitly provided in the input evidence/results.
If insufficient evidence exists, clearly state uncertainty and propose the single best next step.

Hard rules:
- Output must be valid JSON.
- Do NOT include keys not requested by the schema.

Scope rule:
- Use only the exact quantities explicitly requested in the user query.
- Do not introduce additional variants, subtypes, or related metrics unless the user explicitly asks for them.
- If the requested quantity is ambiguous, keep the goal/hypothesis/plans minimal and do not expand the scope.
"""


class AnalysisAgent:
    METAL_SPECIES = {
        
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        
        "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        
        "Al", "Ga", "In", "Sn", "Pb", "Bi",
    }

    def __init__(self, llm=None, agent_name: str = "AnalysisAgent"):
        self.agent_name = agent_name
        self.llm = llm if llm is not None else AGENT_LLM_MAP.get(agent_name, LLM_DEFAULT)


    @staticmethod
    def _match_atoms_by_distance(poscar_mof: Path, poscar_complex: Path, cutoff: float = 0.5):
        from ase.io import read

        atoms_mof = read(poscar_mof)
        atoms_complex = read(poscar_complex)

        N_mof = len(atoms_mof)
        N_complex = len(atoms_complex)

        combined = atoms_mof + atoms_complex

        mapping: Dict[int, int] = {}
        matched_complex: set[int] = set()

        for i_mof in range(N_mof):
            target_indices = list(range(N_mof, N_mof + N_complex))
            dists = combined.get_distances(i_mof, target_indices, mic=True)

            
            j_rel = int(np.argmin(dists))          
            d_min = float(dists[j_rel])

            j_complex0 = j_rel                     
            j_complex1 = int(j_complex0 + 1)       
            mof_idx1 = int(i_mof + 1)              

            if d_min > cutoff:
                print(
                    f"[Warning] Distance from MOF atom {mof_idx1} to the nearest complex atom = "
                    f"{d_min:.2f} Å (exceeds cutoff {cutoff} Å)"
                )

            mapping[mof_idx1] = j_complex1
            matched_complex.add(j_complex1)

        
        guest_indices: List[int] = [
            int(j) for j in range(1, N_complex + 1) if j not in matched_complex
        ]

        return mapping, guest_indices


    def _parse_poscar_idx_to_species(self, poscar_path: Path) -> Dict[int, str]:
        lines = poscar_path.read_text(encoding="utf-8", errors="ignore").splitlines()

        if len(lines) < 7:
            raise ValueError(f"POSCAR too short: {poscar_path}")

        species_line = lines[5].split()
        counts_line = lines[6].split()

        if not species_line:
            raise ValueError(f"POSCAR has empty species line: {poscar_path}")
        if not counts_line:
            raise ValueError(f"POSCAR has empty counts line: {poscar_path}")

        species = species_line
        counts = list(map(int, counts_line))

        if len(species) != len(counts):
            raise ValueError(
                f"POSCAR species/count mismatch in {poscar_path}:\n"
                f"  species={species}\n  counts={counts}"
            )

        idx_to_species: Dict[int, str] = {}
        idx = 1  

        for sp, cnt in zip(species, counts):
            for _ in range(cnt):
                idx_to_species[idx] = sp
                idx += 1

        return idx_to_species

    def _parse_acf_dat(self, acf_path: Path) -> Dict[int, float]:
        if not acf_path.exists():
            raise FileNotFoundError(f"ACF.dat not found at {acf_path}")

        idx_to_e: Dict[int, float] = {}

        lines = acf_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line in lines:
            if re.match(r"^\s*#", line):
                continue
            if re.match(r"^\s*-{3,}", line):
                continue
            if not line.strip():
                continue

            parts = line.split()
            
            if len(parts) >= 5 and parts[0].isdigit():
                idx = int(parts[0])
                e = float(parts[4])
                idx_to_e[idx] = e

        if not idx_to_e:
            raise ValueError(f"ACF.dat has no valid atom lines: {acf_path}")

        return idx_to_e

    def _summarize_delta_q(self, delta: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
        metal_sites = delta.get("metal_sites", {}) or {}
        guest = delta.get("guest", {}) or {}
        summ = delta.get("summary", {}) or {}

        
        metal_items = []
        metal_total = 0.0
        for idx_str, rec in metal_sites.items():
            try:
                idx = int(idx_str)
            except Exception:
                idx = idx_str
            dq = float(rec.get("delta_q", 0.0))
            metal_total += dq
            metal_items.append(
                {
                    "mof_index": idx,
                    "species": rec.get("species", "?"),
                    "delta_q": dq,
                    "q_mof": rec.get("q_mof"),
                    "q_complex": rec.get("q_complex"),
                }
            )

        metal_items_sorted = sorted(metal_items, key=lambda x: abs(float(x["delta_q"])), reverse=True)
        metal_top = metal_items_sorted[: max(1, int(top_k))]

        
        guest_items = []
        guest_charge_sum = 0.0
        guest_species_count = {}
        for idx_str, rec in guest.items():
            try:
                idx = int(idx_str)
            except Exception:
                idx = idx_str
            q = float(rec.get("q_complex", 0.0))
            sp = rec.get("species", "?")
            guest_charge_sum += q
            guest_items.append({"complex_index": idx, "species": sp, "q_complex": q})
            guest_species_count[sp] = guest_species_count.get(sp, 0) + 1

        guest_items_sorted = sorted(guest_items, key=lambda x: x["complex_index"])

        return {
            "definition": {
                "acf_column": "CHARGE (ACF parts[4])",
                "metal_delta_q": "sum over metal_sites of (CHARGE_complex - CHARGE_mof)",
                "co2_charge": "sum over guest atoms of CHARGE_complex (not a delta; no isolated CO2 baseline here)",
            },
            "counts": {
                "n_framework_atoms": summ.get("n_framework_atoms"),
                "n_guest_atoms": summ.get("n_guest_atoms"),
                "n_metal_sites": summ.get("n_metal_sites"),
                "metal_species_found": summ.get("metal_species_found"),
                "guest_species_count": guest_species_count,
            },
            "metal": {
                "delta_q_total": metal_total,
                "top_sites_by_abs_delta_q": metal_top,
            },
            "co2": {
                "guest_charge_sum_in_complex": guest_charge_sum,
                "guest_atoms": guest_items_sorted,
            },
        }
        
    def _build_bader_delta_q_for_mof_complex(self, mof_dir: Path, complex_dir: Path) -> Dict[str, Any]:

        mof_poscar = mof_dir / "POSCAR"
        complex_poscar = complex_dir / "POSCAR"
        acf_mof = mof_dir / "ACF.dat"
        acf_complex = complex_dir / "ACF.dat"

        
        mapping, guest_indices = self._match_atoms_by_distance(
            mof_poscar, complex_poscar, cutoff=0.5
        )


        
        idx_to_e_mof = self._parse_acf_dat(acf_mof)
        idx_to_e_complex = self._parse_acf_dat(acf_complex)

        
        idx_to_sp_mof = self._parse_poscar_idx_to_species(mof_poscar)
        idx_to_sp_complex = self._parse_poscar_idx_to_species(complex_poscar)

        
        framework = {}
        for mof_idx, complex_idx in mapping.items():
            if mof_idx not in idx_to_e_mof or complex_idx not in idx_to_e_complex:
                continue

            sp = idx_to_sp_mof.get(mof_idx, "?")
            q_mof = idx_to_e_mof[mof_idx]
            q_complex = idx_to_e_complex[complex_idx]
            dq = q_complex - q_mof

            framework[mof_idx] = {
                "species": sp,
                "q_mof": q_mof,
                "q_complex": q_complex,
                "delta_q": dq,
                "complex_index": complex_idx,
            }

        
        guest = {}
        for idx in guest_indices:
            if idx in idx_to_e_complex:
                sp = idx_to_sp_complex.get(idx, "?")
                q = idx_to_e_complex[idx]
                guest[idx] = {
                    "species": sp,
                    "q_complex": q,
                }

        
        metal_sites = {
            int(idx): rec
            for idx, rec in framework.items()
            if rec["species"] in self.METAL_SPECIES
        }

        summary = {
            "n_framework_atoms": int(len(framework)),
            "n_guest_atoms": int(len(guest)),
            "n_metal_sites": int(len(metal_sites)),
            "guest_indices": [int(i) for i in guest_indices],
            "metal_species_found": sorted(
                {v["species"] for v in framework.values() if v["species"] in self.METAL_SPECIES}
            ),
        }

        return {
            "framework": {int(k): v for k, v in framework.items()},
            "guest": {int(k): v for k, v in guest.items()},
            "metal_sites": metal_sites,
            "summary": summary,
        }


    def _extract_bader_dirs_any(self, context: Dict[str, Any]) -> Dict[str, Tuple[Path, Path]]:
        upstream = context.get("upstream_plans", {}) or {}
        out = {}

        for plan_name, plan_blob in upstream.items():
            if not isinstance(plan_blob, dict):
                continue
            if not plan_name.endswith("_bader_charge"):
                continue

            mof_job = plan_blob.get(f"{plan_name}_mof", {})
            complex_job = plan_blob.get(f"{plan_name}_complex", {})

            mof_dir = mof_job.get("results", {}).get("bader_charge", {}).get("bader_dir")
            complex_dir = complex_job.get("results", {}).get("bader_charge", {}).get("bader_dir")

            if mof_dir and complex_dir:
                out[plan_name] = (Path(mof_dir), Path(complex_dir))

        return out

    def _extract_zeopp_summaries_any(self, context: Dict[str, Any]) -> Dict[str, Any]:
        upstream = context.get("upstream_plans", {}) or {}
        out: Dict[str, Any] = {}

        for plan_name, plan_blob in upstream.items():
            if not isinstance(plan_blob, dict):
                continue

            for job_id, job in plan_blob.items():
                if not isinstance(job, dict):
                    continue

                mof = job.get("mof") or job.get("MOF")
                prop = job.get("property") or job.get("property_name") or job.get("simulation_property")
                res = job.get("results", {}) or {}
                zeopp = res.get("zeopp", {}) or {}
                raw = zeopp.get("raw", {}) or {}

                if not mof or not prop or not raw:
                    continue

                out.setdefault(mof, {})

                if prop == "pore_volume":
                    out[mof]["pore_volume"] = {
                        "AV_cm3_g": raw.get("AV_cm3_g"),
                        "AV_volume_fraction": raw.get("AV_Volume_fraction"),
                        "AV_A3": raw.get("AV_A3"),
                        "probe_radius_A": (job.get("zeopp_info") or {}).get("probe_radius"),
                        "note": "Zeo++ accessible volume (-ha -vol).",
                    }

                elif prop == "pore_limiting_diameter":
                    out[mof]["pore_diameter"] = {
                        "PLD_free_sphere_A": raw.get("free_sphere"),
                        "LCD_included_sphere_A": raw.get("included_sphere"),
                        "LCD_along_free_path_A": raw.get("included_sphere_along_free_path"),
                        "note": "Zeo++ pore diameters (-ha -res). free_sphere is commonly used as PLD.",
                    }

                elif prop == "largest_cavity_diameter":
                    
                    out[mof]["largest_cavity_diameter"] = {
                        "LCD_included_sphere_A": raw.get("included_sphere"),
                        "note": "Zeo++ largest cavity diameter (included_sphere).",
                    }

                elif prop == "surface_area":
                    out[mof]["surface_area"] = {
                        "ASA_m2_g": raw.get("ASA_m2_g"),
                        "NASA_m2_g": raw.get("NASA_m2_g"),
                        "note": "Zeo++ surface area if available.",
                    }

        return out

    def _extract_diffusivity_summaries_any(self, context: Dict[str, Any]) -> Dict[str, Any]:
        upstream = context.get("upstream_plans", {}) or {}
        out: Dict[str, Any] = {}

        for plan_name, plan_blob in upstream.items():
            if not isinstance(plan_blob, dict):
                continue

            for job_id, job in plan_blob.items():
                if not isinstance(job, dict):
                    continue

                mof = job.get("mof") or job.get("MOF")
                prop = job.get("property") or job.get("property_name") or job.get("simulation_property")
                guest = job.get("guest") or "guest"

                res = job.get("results", {}) or {}
                d = res.get("diffusivity", {}) or {}

                
                if prop != "diffusivity" or not mof or not d:
                    continue

                out.setdefault(mof, {})
                out[mof].setdefault(guest, {})
                out[mof][guest][str(job_id)] = {
                    "D_m2_per_s": d.get("D_m2_per_s"),
                    "r2": d.get("r2"),
                    "slope_A2_per_fs": d.get("slope_A2_per_fs"),
                    "std_err_slope": d.get("std_err_slope"),
                    "p_value": d.get("p_value"),
                    "time_range_fs": d.get("time_range_fs"),
                    "note": "From MSD linear fit (Einstein relation).",
                }

        return out

    def _extract_raspa_henry_summaries_any(self, context: Dict[str, Any]) -> Dict[str, Any]:
        upstream = context.get("upstream_plans", {}) or {}
        out: Dict[str, Any] = {}

        for plan_name, plan_blob in upstream.items():
            if not isinstance(plan_blob, dict):
                continue

            for job_id, job in plan_blob.items():
                if not isinstance(job, dict):
                    continue

                if (job.get("agent") != "RASPAAgent") or (job.get("property") != "henry_coefficient"):
                    continue

                mof = job.get("mof") or job.get("MOF")
                guest = job.get("guest") or "guest"
                res = job.get("results", {}) or {}

                henry = res.get("henry_constant")
                if henry is None:
                    continue

                out.setdefault(mof, {})
                out[mof][guest] = {
                    "henry_constant": henry,
                    "henry_error": res.get("henry_error"),
                    "henry_units": res.get("henry_units"),
                    "raspa_output_file": res.get("raspa_output_file"),
                    "note": "RASPA parsed henry_constant (Widom insertion).",
                }

        return out

    def _run_bader_summaries_any(self, context: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
        context.setdefault("analysis", {})
        context["analysis"].setdefault("bader_summary", {})

        pairs = self._extract_bader_dirs_any(context)
        for plan_name, (mof_dir, complex_dir) in pairs.items():
            delta = self._build_bader_delta_q_for_mof_complex(mof_dir, complex_dir)
            summary = self._summarize_delta_q(delta, top_k=top_k)
            context["analysis"]["bader_summary"][plan_name] = summary

        return context

    def _extract_raspa_uptake_summaries_any(self, context: Dict[str, Any]) -> Dict[str, Any]:
        upstream = context.get("upstream_plans", {}) or {}
        out: Dict[str, Any] = {}

        for plan_name, plan_blob in upstream.items():
            if not isinstance(plan_blob, dict):
                continue

            for job_id, job in plan_blob.items():
                if not isinstance(job, dict):
                    continue

                if (job.get("agent") != "RASPAAgent") or (job.get("property") != "uptake"):
                    continue

                mof = job.get("mof") or job.get("MOF")
                guest = job.get("guest")
                res = job.get("results", {}) or {}

                if not mof:
                    continue

                
                uptake = res.get("uptake_excess")
                units = res.get("uptake_units")

                if uptake is None:
                    continue

                out.setdefault(mof, {})
                out[mof][guest or "guest"] = {
                    "uptake_excess": uptake,
                    "uptake_units": units,
                    "raspa_output_file": res.get("raspa_output_file"),
                    "note": "RASPA parsed uptake_excess",
                }

        return out

    
    
    
    def run(
        self,
        context_or_contexts: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        
        if isinstance(context_or_contexts, dict):
            return self._run_single(context_or_contexts)

        
        
        batch_ctx = self._build_batch_context(context_or_contexts)
        return self._run_single(batch_ctx)

    def _run_single(self, context: Dict[str, Any]) -> Dict[str, Any]:
        context.setdefault("analysis", {})
        trace = context["analysis"].setdefault("trace", [])
        self._trace(trace, "start", {"agent": self.agent_name})
        

        interpret_only = bool(context.get("interpret_only") or context["analysis"].get("interpret_only"))
        try:
            if interpret_only:
                empty_plan = SimulationPlanModel(steps=[])
                evidence: Dict[str, Any] = {"mode": "interpret_only"}

                interp = self._step_interpretation(
                    context,
                    goal="",
                    hypothesis="",
                    plan=empty_plan,
                    evidence=evidence,
                )
                context["analysis"]["interpretation"] = interp.model_dump()
                self._trace(trace, "interpretation", interp.model_dump())
                self._trace(trace, "end", {"agent": self.agent_name})
                return context

            goal = self._step_goal(context)
            context["analysis"]["goal"] = goal.goal
            self._trace(trace, "goal", goal.model_dump())

            hyp = self._step_hypothesis(context, goal.goal)
            context["analysis"]["hypothesis"] = hyp.hypothesis
            self._trace(trace, "hypothesis", hyp.model_dump())

            plan = self._step_plan(context, goal.goal, hyp.hypothesis)
            context["analysis"]["plan"] = plan.model_dump()
            self._trace(trace, "plan", plan.model_dump())

            evidence: Dict[str, Any] = {}

            context["analysis"]["evidence"] = evidence

            interp = self._step_interpretation(context, goal.goal, hyp.hypothesis, plan, evidence)
            context["analysis"]["interpretation"] = interp.model_dump()
            self._trace(trace, "interpretation", interp.model_dump())

        except ValidationError as ve:
            context["analysis"]["error"] = {"type": "ValidationError", "message": str(ve)}
            self._trace(trace, "error", context["analysis"]["error"])
        except Exception as e:
            context["analysis"]["error"] = {"type": type(e).__name__, "message": str(e)}
            self._trace(trace, "error", context["analysis"]["error"])

        self._trace(trace, "end", {"agent": self.agent_name})
        return context

    def _build_batch_context(
        self,
        contexts: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not contexts:
            raise ValueError("AnalysisAgent: no contexts provided for batch mode")

        first = contexts[0]

        query_text = first.get("query_text") or first.get("QueryText") or ""
        job_name   = first.get("batch_job_name") or first.get("job_name", "")
        prop       = (
            first.get("property")
            or first.get("property_name")
            or first.get("simulation_property")
            or ""
        )
        guest      = first.get("guest")

        batch_results: Dict[str, Any] = {}
        per_mof_info: Dict[str, Any] = {}

        for idx, ctx in enumerate(contexts):
            results = ctx.get("results", {}) or {}

            
            mof_name = (
                ctx.get("mof")
                or ctx.get("MOF")
                or ctx.get("job_name")
                or f"system_{idx}"
            )

            batch_results[mof_name] = results

            per_mof_info[mof_name] = {
                "work_dir": ctx.get("work_dir"),
                "job_name": ctx.get("job_name"),
                "mof": ctx.get("mof", mof_name),
                "guest": ctx.get("guest", guest),
                "property": ctx.get("property", prop),
            }

        batch_context: Dict[str, Any] = {
            "job_name": job_name,
            "property": prop,
            "guest": guest,
            "query_text": query_text,
            "results": batch_results,
            "per_mof_info": per_mof_info,
        }

        return batch_context
    
    
    
    def _call_llm(self, messages: List[Any]) -> str:
        llm_obj = self.llm

        
        if callable(llm_obj) and not hasattr(llm_obj, "invoke"):
            try:
                candidate = llm_obj()
                if hasattr(candidate, "invoke"):
                    llm_obj = candidate
            except TypeError:
                
                pass

        if hasattr(llm_obj, "invoke"):
            resp = llm_obj.invoke(messages)
            return str(getattr(resp, "content", resp))
        else:
            resp = llm_obj(messages)
            if hasattr(resp, "content"):
                return str(resp.content)
            return str(resp)

    def _safe_json_loads(self, text: str) -> Dict[str, Any]:
        t = str(text).strip()
        if t.startswith("```"):
            lines = t.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            t = "\n".join(lines).strip()
        return json.loads(t)

    def _invoke_llm_json(self, prompt: str, model_cls):
        messages = [SystemMessage(content=DEFAULT_SYSTEM), HumanMessage(content=prompt)]
        raw = self._call_llm(messages)
        obj = self._safe_json_loads(raw)
        return model_cls.model_validate(obj)

    
    
    
    def _step_goal(self, context: Dict[str, Any]) -> ExplanationGoalModel:
        q = context.get("query_text", context.get("QueryText", ""))
        results = context.get("results", {})
        prompt = f"""{DEFAULT_SYSTEM}

Task: Define a concise analysis goal based on the user's question and any available results.

User question:
{q}

Available results keys:
{list(results.keys())}

Return JSON:
{{"goal":"..."}}"""
        return self._invoke_llm_json(prompt, ExplanationGoalModel)

    
    
    
    def _step_hypothesis(self, context: Dict[str, Any], goal: str) -> HypothesisModel:
        q = context.get("query_text", context.get("QueryText", ""))
        prompt = f"""{DEFAULT_SYSTEM}

Task: Propose exactly ONE testable hypothesis.

Goal:
{goal}

User question:
{q}

Guidelines for hypothesis (PRIORITY ORDER):

1. OVERRIDING RULE (highest priority):
- If the task studies temperature dependence for a SINGLE MOF, do NOT introduce any structural descriptors (pore size, PLD, LCD, flexibility, etc.).
- Temperature effects should be explained purely by thermal activation and molecular motion.

2. Diffusivity-related tasks (when multiple MOFs are compared):
- Hypotheses may involve pore structure descriptors such as pore size, pore limiting diameter, connectivity, flexibility, or free volume.

3. Binding energy tasks:
- Consider hypotheses involving Bader charge or charge transfer.

4. Adsorption/selectivity tasks:
- Prefer thermodynamic hypotheses testable by RASPA (henry_coefficient or heat_of_adsorption, plus uptake/selectivity).
- Do NOT mention diffusivity unless the user explicitly asks for diffusion/kinetics/transport.
- If the user asks for "selectivity" and it is an allowed method, DO NOT redefine it as single-component uptake ratio; use the selectivity method definition (x_A/x_B)/(y_A/y_B).

5. General rules:
- Hypotheses should identify physical factors that explain differences and should be testable by explicit computation.
- Do NOT introduce additional structural descriptors unless multiple MOFs are compared or the user explicitly requests activation energy or mechanistic analysis.

Return JSON:
{{"hypothesis":"..."}}"""
        return self._invoke_llm_json(prompt, HypothesisModel)

    
    
    
    def _step_plan(self, context: Dict[str, Any], goal: str, hypothesis: str) -> SimulationPlanModel:
        q = context.get("query_text", context.get("QueryText", ""))
        prompt = f"""{DEFAULT_SYSTEM}

Rules for Plan:

- You MUST include MAIN result steps that compute the property explicitly requested by the user.
- MAIN steps are the only REQUIRED steps.
- EXPLANATORY steps are OPTIONAL:
  - Include them ONLY if the hypothesis explicitly requires additional computable descriptors
    and those descriptors are meaningful for explaining differences.
  - Do NOT add EXPLANATORY steps for simple trends (e.g., temperature dependence in a single MOF).
- EXPLANATORY steps are OPTIONAL in general, BUT they become REQUIRED if the hypothesis explicitly states they are needed to test the hypothesis.
- Do NOT force a fixed number of steps.
- ANALYSIS steps MUST NOT be included if they are not computable by ALLOWED_METHODS.
  Interpretation and explanation should be handled in the final response instead.
- Every step's "method" MUST be one of: {ALLOWED_METHODS}.
- Every step's "method" MUST be one of: {ALLOWED_METHODS}.
- If the hypothesis mentions something not computable by {ALLOWED_METHODS}, do NOT include it as a step.
  Ignore it or replace it with the closest computable alternative from {ALLOWED_METHODS}.
- Do NOT add prerequisite or intermediate-output steps (e.g., MSD) if a direct method exists for the requested quantity in {ALLOWED_METHODS}.

Goal:
{goal}

Hypothesis:
{hypothesis}

User question:
{q}

Return JSON:
{{
  "steps": [
    {{"name": "...", "method": "..."}},
    {{"name": "...", "method": "..."}},
  ]
}}"""
        plan = self._invoke_llm_json(prompt, SimulationPlanModel)
        return plan

    
    
    
    def _trace(self, trace_list: List[Dict[str, Any]], event: str, data: Dict[str, Any]) -> None:
        trace_list.append({"t": time.time(), "event": event, "data": data})

    
    
    
    
    def _step_interpretation(
        self,
        context: Dict[str, Any],
        goal: str,
        hypothesis: str,
        plan: SimulationPlanModel,
        evidence: Dict[str, Any],
    ) -> InterpretationModel:
        q = context.get("query_text") or context.get("QueryText") or ""

        
        
        context = self._run_bader_summaries_any(context, top_k=5)

        analysis_blob = context.get("analysis", {}) or {}
        bader_summary = analysis_blob.get("bader_summary", {}) or {}

        
        upstream = context.get("upstream_plans", {}) or {}
        binding_energy_summary: Dict[str, Any] = {}

        for plan_name, plan_blob in upstream.items():
            if not isinstance(plan_blob, dict):
                continue
            if not plan_name.endswith("_binding_energy"):
                continue

            
            jm = plan_blob.get(f"{plan_name}_mof", {})
            jg = plan_blob.get(f"{plan_name}_guest", {})
            jc = plan_blob.get(f"{plan_name}_complex", {})

            try:
                Emof = float(jm.get("results", {}).get("vasp_energy_ev"))
                Eguest = float(jg.get("results", {}).get("vasp_energy_ev"))
                Ecomplex = float(jc.get("results", {}).get("vasp_energy_ev"))
            except (TypeError, ValueError):
                
                continue

            
            mof_name = jm.get("mof") or plan_blob.get("mof") or plan_name.split("_CO2_")[0]

            Ebind = Ecomplex - (Emof + Eguest)
            binding_energy_summary[mof_name] = {
                "E_bind_ev": Ebind,
                "E_mof_ev": Emof,
                "E_guest_ev": Eguest,
                "E_complex_ev": Ecomplex,
                "convention": "E_bind = E(MOF+Guest) - E(MOF) - E(Guest); more negative => stronger binding",
            }
        
        zeopp_summary = self._extract_zeopp_summaries_any(context)
        diff_summary = self._extract_diffusivity_summaries_any(context)
        uptake_summary = self._extract_raspa_uptake_summaries_any(context)
        henry_summary = self._extract_raspa_henry_summaries_any(context)

        
        results_for_prompt: Dict[str, Any] = {
            "henry": henry_summary,
            "uptake": uptake_summary,
            "binding_energy": binding_energy_summary,
            "bader_summary": bader_summary,
            "zeopp_summary": zeopp_summary,
            "diffusivity_summary": diff_summary,
        }

        
        raw_results = context.get("results", {}) or {}
        if (
            not henry_summary
            and not uptake_summary
            and not binding_energy_summary
            and not bader_summary
            and not zeopp_summary
            and not diff_summary
        ):
            if raw_results:
                results_for_prompt = raw_results
            else:
                results_for_prompt = {"upstream_plans": context.get("upstream_plans", {})}

        
        prompt = f"""{DEFAULT_SYSTEM}

    Task:
    - The following "User query" is the primary instruction.
    - Interpret the available results to answer the query.
    - Use ONLY the provided results. Do NOT fabricate numbers.
    - If results are insufficient, state uncertainties and propose the single best next step.

    User query (PRIMARY INSTRUCTION):
    {q}

    Available results:
    {json.dumps(results_for_prompt, indent=2, ensure_ascii=False)}

    Optional context (may be empty; do not depend on it):
    - goal: {goal}
    - hypothesis: {hypothesis}
    - plan: {json.dumps(plan.model_dump(), indent=2, ensure_ascii=False)}
    - evidence: {json.dumps(evidence, indent=2, ensure_ascii=False)}

    Notes:
    - Binding energy: more negative E_bind means stronger binding.
    - Bader summary:
    * Use metal.delta_q_total to discuss charge transfer at metal sites.
    * Use guest_charge_sum_in_complex to discuss guest polarization/interaction strength.
    * ACF "CHARGE" column is used as parsed by the code.

    Return JSON:
    {{
    "summary": "",
    "key_findings": [],
    "uncertainties": [],
    "next_best_step": ""
    }}
    """
        return self._invoke_llm_json(prompt, InterpretationModel)

    def recommend_analysis_tasks(self, context: Dict[str, Any]) -> str:
        
        goal_obj = self._step_goal(context)
        goal = goal_obj.goal
        print("goal:", goal)

        
        hyp_obj = self._step_hypothesis(context, goal)
        hypothesis = hyp_obj.hypothesis
        print("hypothesis:", hypothesis)

        
        plan = self._step_plan(context, goal, hypothesis)
        return plan