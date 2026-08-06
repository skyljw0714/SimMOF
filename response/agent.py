import json
import re
from typing import Dict, Any, Tuple

from config import LLM_DEFAULT
from langchain.schema import SystemMessage, HumanMessage


class ResponseAgent:

    def __init__(self, llm=None):
        self.llm = llm or LLM_DEFAULT

    
    
    
    @staticmethod
    def _extract_results_from_job_ctx(job_ctx: Any) -> Dict[str, Any]:
        if not isinstance(job_ctx, dict):
            return {}
        res = job_ctx.get("results")
        return res if isinstance(res, dict) else {}

    def _collect_upstream_results(self, context: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        merged_results: Dict[str, Any] = {}
        namespaced: Dict[str, Any] = {"plans": {}, "jobs": {}}

        
        up_plans = context.get("upstream_plans", {})
        if isinstance(up_plans, dict):
            for plan_name, plan_payload in up_plans.items():
                plan_bucket = {}
                if isinstance(plan_payload, dict):
                    for job_id, job_ctx in plan_payload.items():
                        job_res = self._extract_results_from_job_ctx(job_ctx)
                        plan_bucket[job_id] = job_res

                        
                        
                        merged_results.update(job_res)

                namespaced["plans"][plan_name] = plan_bucket

        
        up_jobs = context.get("upstream_jobs", {})
        if isinstance(up_jobs, dict):
            for job_id, job_ctx in up_jobs.items():
                job_res = self._extract_results_from_job_ctx(job_ctx)
                namespaced["jobs"][job_id] = job_res
                merged_results.update(job_res)

        return merged_results, namespaced

    @staticmethod
    def _merge_into_context_results(context: Dict[str, Any], merged_results: Dict[str, Any]) -> Dict[str, Any]:
        results = context.setdefault("results", {})
        if not isinstance(results, dict):
            results = {}
            context["results"] = results

        
        results.update(merged_results)
        return results

    @staticmethod
    def _walk(obj: Any):
        if isinstance(obj, dict):
            yield obj
            for v in obj.values():
                yield from ResponseAgent._walk(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from ResponseAgent._walk(v)

    @staticmethod
    def find_interpretation(context: Dict[str, Any]) -> Dict[str, Any]:
        
        analysis = context.get("analysis")
        if isinstance(analysis, dict):
            interp = analysis.get("interpretation")
            if isinstance(interp, dict) and interp:
                return interp

        
        for node in ResponseAgent._walk(context.get("upstream_plans", {})):
            a = node.get("analysis")
            if isinstance(a, dict):
                interp = a.get("interpretation")
                if isinstance(interp, dict) and interp:
                    return interp

        
        for node in ResponseAgent._walk(context.get("upstream_jobs", {})):
            a = node.get("analysis")
            if isinstance(a, dict):
                interp = a.get("interpretation")
                if isinstance(interp, dict) and interp:
                    return interp

        return {}

    @staticmethod
    def _collect_vasp_adsorption_summaries(
        context: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        jobs_by_plan: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for node in ResponseAgent._walk(context):
            if (node.get("property") or "").lower() != "binding_energy":
                continue
            role = str(
                node.get("vasp_role")
                or (node.get("vasp_system") or {}).get("role")
                or ""
            ).lower()
            if role not in {"mof", "guest", "complex"}:
                job_id = str(node.get("job_id") or "")
                role = next(
                    (
                        candidate
                        for candidate in ("mof", "guest", "complex")
                        if job_id.endswith(f"_{candidate}")
                    ),
                    "",
                )
            if role not in {"mof", "guest", "complex"}:
                continue
            plan_name = str(node.get("plan_name") or node.get("job_name") or "")
            if not plan_name:
                continue
            existing = jobs_by_plan.setdefault(plan_name, {}).get(role)
            energy = (node.get("results") or {}).get("vasp_energy_ev")
            if existing is None or (
                (existing.get("results") or {}).get("vasp_energy_ev") is None
                and energy is not None
            ):
                jobs_by_plan[plan_name][role] = node

        summaries: Dict[str, Dict[str, Any]] = {}
        for plan_name, jobs in jobs_by_plan.items():
            if not all(role in jobs for role in ("mof", "guest", "complex")):
                continue
            try:
                e_mof = float(jobs["mof"]["results"]["vasp_energy_ev"])
                e_guest = float(jobs["guest"]["results"]["vasp_energy_ev"])
                e_complex = float(jobs["complex"]["results"]["vasp_energy_ev"])
            except (KeyError, TypeError, ValueError):
                continue
            complex_results = jobs["complex"].get("results") or {}
            interaction = complex_results.get("interaction_energy") or {}
            deformation = complex_results.get("structure_deformation") or {}
            e_relaxed = e_complex - e_mof - e_guest
            summary: Dict[str, Any] = {
                "status": "ok",
                "plan_name": plan_name,
                "mof": jobs["complex"].get("mof") or jobs["mof"].get("mof"),
                "guest": jobs["complex"].get("guest") or jobs["guest"].get("guest"),
                "E_ads_relaxed_ev": e_relaxed,
                "E_complex_opt_ev": e_complex,
                "E_mof_opt_ev": e_mof,
                "E_guest_opt_ev": e_guest,
                "relaxed_equation": (
                    "E_ads_relaxed = E(MOF+guest,opt) - E(MOF,opt) - E(guest,opt)"
                ),
                "relaxed_energy_includes": [
                    "direct host-guest interaction",
                    "MOF deformation",
                    "guest deformation",
                    "cell deformation when the cell was relaxed",
                ],
                "structure_deformation": deformation,
                "deformation_threshold_exceeded": bool(
                    deformation.get("threshold_exceeded")
                ),
                "interaction_energy_status": interaction.get("status"),
            }
            if interaction.get("status") == "ok" and interaction.get("E_int_ev") is not None:
                e_int = float(interaction["E_int_ev"])
                summary.update(
                    {
                        "E_int_ev": e_int,
                        "interaction_equation": interaction.get("equation"),
                        "deformation_contribution_ev": e_relaxed - e_int,
                    }
                )
            elif interaction:
                summary["interaction_energy"] = interaction
            summaries[plan_name] = summary
        return summaries

    @staticmethod
    def _format_adsorption_notice(
        summaries: Dict[str, Dict[str, Any]],
        query_text: str,
    ) -> str:
        if not summaries:
            return ""
        korean = bool(re.search(r"[가-힣]", query_text or ""))
        lines = []
        for summary in summaries.values():
            target = "–".join(
                str(value)
                for value in (summary.get("mof"), summary.get("guest"))
                if value
            ) or summary.get("plan_name", "VASP adsorption")
            e_relaxed = summary.get("E_ads_relaxed_ev")
            if korean:
                lines.append(
                    f"{target}: relaxed adsorption energy(변형 효과 포함) = "
                    f"{float(e_relaxed):.6f} eV."
                )
            else:
                lines.append(
                    f"{target}: relaxed adsorption energy (including deformation effects) = "
                    f"{float(e_relaxed):.6f} eV."
                )

            deformation = summary.get("structure_deformation") or {}
            if summary.get("deformation_threshold_exceeded"):
                measured = float(deformation.get("overall_deformation_percent") or 0.0)
                threshold = float(deformation.get("threshold_percent") or 20.0)
                if korean:
                    lines.append(
                        f"구조 변형 경고: {measured:.2f}%로 설정 임계값 "
                        f"{threshold:.2f}% 이상입니다."
                    )
                else:
                    lines.append(
                        f"Structural-deformation warning: {measured:.2f}% is at or above "
                        f"the {threshold:.2f}% threshold."
                    )

            if summary.get("E_int_ev") is not None:
                if korean:
                    lines.append(
                        "최적화된 결합 구조의 frozen interaction energy = "
                        f"{float(summary['E_int_ev']):.6f} eV."
                    )
                else:
                    lines.append(
                        "Frozen interaction energy at the optimized complex geometry = "
                        f"{float(summary['E_int_ev']):.6f} eV."
                    )
            elif summary.get("deformation_threshold_exceeded"):
                status = summary.get("interaction_energy_status") or "not_available"
                if korean:
                    lines.append(
                        f"frozen interaction energy는 아직 제시할 수 없습니다(상태: {status})."
                    )
                else:
                    lines.append(
                        f"Frozen interaction energy is not available (status: {status})."
                    )
        return "\n".join(lines)

    
    
    
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        
        job_name = context.get("job_name") or context.get("plan_name", "")
        mof = context.get("mof", "")
        guest = context.get("guest", None)
        prop = context.get("property", "")
        query_text = context.get("query_text", "")

        
        upstream_merged, upstream_namespaced = self._collect_upstream_results(context)
        results = self._merge_into_context_results(context, upstream_merged)

        adsorption_summaries = self._collect_vasp_adsorption_summaries(context)
        if adsorption_summaries:
            results["vasp_adsorption_energies"] = adsorption_summaries

        
        
        results.setdefault("_upstream", upstream_namespaced)

        
        batch_summary = results.get("raspa_batch_summary")
        if isinstance(batch_summary, dict):
            ranked = batch_summary.get("ranked", [])
            total = batch_summary.get("total")
            success = batch_summary.get("success")
            top_n = batch_summary.get("top_n")

            summary_for_llm = {
                "job_name": job_name,
                "property": prop,
                "mof": mof,
                "guest": guest,
                "batch_summary": {
                    "total": total,
                    "success": success,
                    "top_n": top_n,
                    "ranked": ranked,
                },
            }

            system_prompt = (
                "You summarize batch RASPA simulation results for porous materials.\n"
                "Rules:\n"
                "- First, state what was computed (guest, temperature/pressure if known, property).\n"
                "- Then output a ranked Top-N list.\n"
                "  * Each line MUST be: 'rank) MOF_NAME: VALUE UNIT'\n"
                "  * Use the values exactly as provided in the JSON (do not recompute).\n"
                "- Do NOT invent numbers or MOF names.\n"
                "- Keep it concise.\n"
            )

            user_prompt = (
                f"User query:\n{query_text or '(no explicit query text)'}\n\n"
                f"Structured batch summary (JSON):\n"
                f"{json.dumps(summary_for_llm, ensure_ascii=False, indent=2)}\n\n"
                "Write a short, clear summary for the user."
            )

            from core.llm_logging import set_llm_context
            set_llm_context("ResponseAgent", "final_response_batch")
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])

            answer_text = response.content
            results["final_response"] = answer_text
            context["results"] = results

            print("\n=== ResponseAgent: final user-facing answer (BATCH) ===\n")
            print(answer_text)
            print("\n=== End of Response ===\n")
            return context

        
        analysis = context.setdefault("analysis", {})
        if not isinstance(analysis, dict):
            analysis = {}
            context["analysis"] = analysis

        interpretation = analysis.get("interpretation")
        if not (isinstance(interpretation, dict) and interpretation):
            interpretation = self.find_interpretation(context)
            analysis["interpretation"] = interpretation

        summary_for_llm = {
            "job_name": job_name,
            "property": prop,
            "mof": mof,
            "guest": guest,
            "results": results,
            "interpretation": interpretation,
        }

        system_prompt = (
            "You are an expert in molecular simulations and porous materials.\n"
            "Your job is to turn structured simulation results into a clear, concise explanation for the user.\n"
            "\n"
            "Guidelines:\n"
            "- First, state briefly what was computed (property, MOF, guest).\n"
            "- If an 'interpretation' object is provided, treat it as the primary source of truth:\n"
            "  * Base your answer mainly on 'summary' and 'key_findings'.\n"
            "  * Do NOT contradict or reinterpret the physical conclusions in 'key_findings'.\n"
            "  * Explicitly mention important numerical values that appear there.\n"
            "- Otherwise, fall back to the raw 'results' data.\n"
            "- If the status indicates failure or missing data, do NOT invent numbers; explain that it did not complete properly.\n"
            "- For VASP adsorption, call E_ads_relaxed the relaxed adsorption energy (including deformation effects), not a direct interaction energy.\n"
            "- If E_int_ev is available, explicitly report both E_ads_relaxed_ev and E_int_ev with their distinct meanings.\n"
            "- If deformation_threshold_exceeded is true, explicitly warn the user that structural deformation reached or exceeded 20%.\n"
            "- Keep the answer concise: at most 10 sentences in total.\n"
            "- Answer in the same language as the user query if it is clear; otherwise default to English.\n"
        )

        user_prompt = (
            f"User query:\n{query_text or '(no explicit query text)'}\n\n"
            f"Structured results (JSON):\n"
            f"{json.dumps(summary_for_llm, ensure_ascii=False, indent=2)}\n\n"
            "Write a short explanation for the user."
        )

        from core.llm_logging import set_llm_context
        set_llm_context("ResponseAgent", "final_response")
        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])

        mandatory_notice = self._format_adsorption_notice(
            adsorption_summaries,
            query_text,
        )
        answer_text = response.content
        if mandatory_notice:
            answer_text = f"{mandatory_notice}\n\n{answer_text}"
        results["final_response"] = answer_text
        context["results"] = results

        print("\n=== ResponseAgent: final user-facing answer ===\n")
        print(answer_text)
        print("\n=== End of Response ===\n")

        return context
