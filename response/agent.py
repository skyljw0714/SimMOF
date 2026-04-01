import json
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

    
    
    
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        
        job_name = context.get("job_name") or context.get("plan_name", "")
        mof = context.get("mof", "")
        guest = context.get("guest", None)
        prop = context.get("property", "")
        query_text = context.get("query_text", "")

        
        upstream_merged, upstream_namespaced = self._collect_upstream_results(context)
        results = self._merge_into_context_results(context, upstream_merged)

        
        
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
            "- Keep the answer concise: at most 10 sentences in total.\n"
            "- Answer in the same language as the user query if it is clear; otherwise default to English.\n"
        )

        user_prompt = (
            f"User query:\n{query_text or '(no explicit query text)'}\n\n"
            f"Structured results (JSON):\n"
            f"{json.dumps(summary_for_llm, ensure_ascii=False, indent=2)}\n\n"
            "Write a short explanation for the user."
        )

        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])

        answer_text = response.content
        results["final_response"] = answer_text
        context["results"] = results

        print("\n=== ResponseAgent: final user-facing answer ===\n")
        print(answer_text)
        print("\n=== End of Response ===\n")

        return context
