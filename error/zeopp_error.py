import json
from typing import Any, Dict, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from input.zeopp_input import ZeoppInputAgent
from Zeopp.runner import ZeoppRunner
from config import LLM_DEFAULT, working_dir

class ZeoppErrorAgent:
    
    def __init__(
        self,
        llm=None,
        max_retries: int = 2,
        zeopp_runner: Optional[ZeoppRunner] = None,
        zeopp_input_agent: Optional[ZeoppInputAgent] = None,
    ):
        self.llm = llm or LLM_DEFAULT
        self.max_retries = max_retries
        
        self.zeopp_runner = zeopp_runner or ZeoppRunner()
        self.zeopp_input_agent = zeopp_input_agent or ZeoppInputAgent(llm=self.llm)

    
    
    
    def _call_llm(self, context: Dict[str, Any]) -> Dict[str, Any]:
        results    = context.get("results", {})
        zeopp_info = context.get("zeopp_info", {})
        cmd        = context.get("zeopp_command", "")

        returncode = results.get("zeopp_returncode")
        stdout     = results.get("zeopp_stdout", "")
        stderr     = results.get("zeopp_stderr", "")

        mof        = context.get("mof")
        prop       = context.get("property")
        query_text = context.get("query_text")

        prompt = f"""
You are an expert in diagnosing and fixing Zeo++ (zeopp) command-line errors.

Here is the context:
- MOF: {mof}
- Target property: {prop}
- User query: {query_text}

- Original zeopp_info (JSON):
{json.dumps(zeopp_info, indent=2, ensure_ascii=False)}

- Executed command:
{cmd}

- Return code: {returncode}

- STDOUT:
{stdout}

- STDERR:
{stderr}

Analyze why the Zeo++ command failed and propose a fix if possible.

Respond ONLY with a JSON object in the following format:

{{
  "action": "retry" or "abort",
  "reason": "short explanation in English",
  "fixed_zeopp_info": (optional, JSON object with same schema as zeopp_info, or null),
  "fixed_command": (optional, full Zeo++ command string, or null)
}}

Guidelines:
- If the failure is due to missing or wrong parameters (e.g. probe radius, number of samples, wrong subcommand like -sa/-ha/-psd/-chan/-block), return "action": "retry" and a corrected "fixed_zeopp_info".
- If only a small tweak in the command string is needed, you can return "fixed_command".
- If the error cannot be reasonably fixed from this information, return "action": "abort".
"""

        messages = [
            SystemMessage(content="You are a strict Zeo++ error fixer. Output only the JSON object described in the prompt."),
            HumanMessage(content=prompt),
        ]
        resp = self.llm.invoke(messages)

        try:
            data = json.loads(resp.content)
        except Exception as e:
            print("[ZeoppErrorAgent] LLM parsing error:", e)
            print("Raw response:", resp.content)
            
            return {
                "action": "abort",
                "reason": "LLM JSON parsing failed",
                "fixed_zeopp_info": None,
                "fixed_command": None,
            }

        
        data.setdefault("action", "abort")
        data.setdefault("reason", "")
        data.setdefault("fixed_zeopp_info", None)
        data.setdefault("fixed_command", None)
        return data

    
    
    
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        results = context.setdefault("results", {})
        status  = results.get("zeopp_status")

        
        if status == "ok":
            print("[ZeoppErrorAgent] zeopp_status == ok → nothing to do.")
            return context

        
        if status in ("input_failed", "no_command"):
            print(f"[ZeoppErrorAgent] status = {status} → cannot fix here.")
            return context

        
        attempts = results.get("zeopp_attempts", 0)

        
        while attempts < self.max_retries:
            
            if status not in ("run_failed", "retry"):
                print(f"[ZeoppErrorAgent] status = {status} → nothing to fix.")
                return context

            print("\n=== ZeoppErrorAgent: analyzing Zeo++ error (attempt "
                  f"{attempts + 1}/{self.max_retries}) ===")
            results["zeopp_attempts"] = attempts + 1

            
            llm_result = self._call_llm(context)
            action     = llm_result.get("action", "abort")
            reason     = llm_result.get("reason")
            fixed_info = llm_result.get("fixed_zeopp_info")
            fixed_cmd  = llm_result.get("fixed_command")

            results["zeopp_error_reason"] = reason
            results["zeopp_error_action"] = action

            if action != "retry":
                
                print("[ZeoppErrorAgent] LLM suggests abort:", reason)
                results["zeopp_status"] = "run_failed"
                return context

            print("[ZeoppErrorAgent] LLM suggests retry with updated command/params.")
            
            if fixed_info:
                
                context["zeopp_info"] = fixed_info
                mof = context.get("mof")
                if mof:
                    context["zeopp_info"]["MOF"] = mof

                
                work_dir = context.get("work_dir", working_dir)
                new_cmd = self.zeopp_input_agent._get_zeopp_command(
                    context["zeopp_info"],
                    cif_dir=work_dir,
                )
                context["zeopp_command"] = new_cmd

            elif fixed_cmd:
                context["zeopp_command"] = fixed_cmd

            else:
                
                print("[ZeoppErrorAgent] action=retry but no fixed_info/fixed_cmd → abort.")
                results["zeopp_status"] = "run_failed"
                return context

            
            print("[ZeoppErrorAgent] Re-running Zeo++ with updated command...")
            context = self.zeopp_runner.run(context)
            results = context.setdefault("results", {})
            status  = results.get("zeopp_status")
            attempts = results.get("zeopp_attempts", attempts + 1)

            
            if status == "ok":
                print("[ZeoppErrorAgent] Retry succeeded.")
                return context

            if status in ("no_command", "input_failed"):
                print(f"[ZeoppErrorAgent] status changed to {status} → cannot continue retry.")
                return context

            

        
        print(f"[ZeoppErrorAgent] max_retries ({self.max_retries}) reached. Abort.")
        results["zeopp_status"] = "run_failed"
        return context
