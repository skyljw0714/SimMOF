import json
import os
import signal
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from input.zeopp_input import ZeoppInputAgent
from Zeopp.runner import ZeoppRunner
from Zeopp.validation import store_validation_report, validate_zeopp_postflight
from config import LLM_DEFAULT, working_dir
from core.job_manager import record_job_event
from error.structure_regeneration import request_structure_regeneration

class ZeoppErrorAgent:
    
    def __init__(
        self,
        llm=None,
        max_retries: int = 2,
        zeopp_runner: Optional[ZeoppRunner] = None,
        zeopp_input_agent: Optional[ZeoppInputAgent] = None,
        action_handlers: Optional[
            Dict[str, Callable[[Dict[str, Any], Dict[str, Any], str], str]]
        ] = None,
    ):
        self.llm = llm or LLM_DEFAULT
        self.max_retries = max_retries
        
        self.zeopp_runner = zeopp_runner or ZeoppRunner()
        self.zeopp_input_agent = zeopp_input_agent or ZeoppInputAgent(llm=self.llm)
        self.action_handlers = {
            "retry": self._handle_retry,
            "regenerate_structure": self._handle_structure_regeneration,
            "abort": self._handle_abort,
        }
        if action_handlers:
            self.action_handlers.update(action_handlers)

        self.action_descriptions = {
            "retry": (
                "Apply a corrected zeopp_info or command and execute Zeo++ again."
            ),
            "regenerate_structure": (
                "Reacquire or clean the structure, rebuild the Zeo++ input, and rerun."
            ),
            "abort": (
                "Stop without changing the command or structure when no controlled "
                "recovery is justified."
            ),
        }

    
    
    
    def _call_llm(self, context: Dict[str, Any]) -> Dict[str, Any]:
        results    = context.get("results", {})
        zeopp_info = context.get("zeopp_info", {})
        cmd        = context.get("zeopp_command", "")

        returncode = results.get("zeopp_returncode")
        stdout     = results.get("zeopp_stdout", "")
        stderr     = results.get("zeopp_stderr", "")
        validation = {
            "stage": results.get("zeopp_validation_stage"),
            "detector_findings": results.get(
                "zeopp_validation_issues",
                results.get("zeopp_validation_errors", []),
            ),
            "runtime_observations": results.get(
                "zeopp_validation_observations",
                [],
            ),
            "evidence": results.get("zeopp_validation_evidence", []),
        }

        mof        = context.get("mof")
        prop       = context.get("property")
        query_text = context.get("query_text")
        action_catalog = {
            name: self.action_descriptions.get(
                name,
                "Execute the registered controlled recovery handler.",
            )
            for name in self.action_handlers
        }

        prompt = f"""
You are an expert in diagnosing and fixing Zeo++ (zeopp) command-line errors. This simulation uses Zeo++ (zeopp).

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

- SimMOF deterministic checks:
{json.dumps(validation, indent=2, ensure_ascii=False)}

Analyze the complete execution and validation evidence. Determine why the result
is not trustworthy and select the most appropriate registered recovery operation.
The runtime_observations are direct measurements, not Zeo++ messages or diagnoses.
Infer their meaning from the observed fields instead of treating them as named
error categories.

Registered recovery operations:
{json.dumps(action_catalog, indent=2, ensure_ascii=False)}

Respond ONLY with a JSON object in the following format:

{{
  "action": "<one registered recovery operation name>",
  "reason": "short explanation in English",
  "evidence": ["exact line(s) from STDOUT or STDERR, or exact observed fields from the deterministic checks"],
  "fixed_zeopp_info": (optional, JSON object with same schema as zeopp_info, or null),
  "fixed_command": (optional, full Zeo++ command string, or null)
}}

Guidelines:
- Base the decision on the complete evidence and context, not an isolated keyword
  or a pre-defined keyword-to-action mapping.
- Select an operation by matching the required recovery to the registered
  operation capabilities above.
- Preserve parameters that are unrelated to the reported failure.
- Do not invent files, paths, radii, or other resources that are not present in
  the supplied context.
"""

        messages = [
            SystemMessage(content="You are a strict Zeo++ error fixer. Output only the JSON object described in the prompt."),
            HumanMessage(content=prompt),
        ]
        resp = self.llm.invoke(messages)

        try:
            data = json.loads(resp.content)
            if not isinstance(data, dict):
                raise TypeError("LLM response is not a JSON object")
        except Exception as e:
            print("[ZeoppErrorAgent] LLM parsing error:", e)
            print("Raw response:", resp.content)
            
            return {
                "action": "abort",
                "reason": "LLM JSON parsing failed",
                "evidence": [],
                "fixed_zeopp_info": None,
                "fixed_command": None,
            }

        
        action = str(data.get("action", "abort")).strip()
        data["action"] = action
        if action not in self.action_handlers:
            data["action"] = "abort"
            data["reason"] = "LLM returned an unsupported action"
        data.setdefault("reason", "")
        if not isinstance(data.get("evidence"), list):
            data["evidence"] = []
        data.setdefault("fixed_zeopp_info", None)
        data.setdefault("fixed_command", None)
        return data

    
    
    
    def pre_run_review(self, context: Dict[str, Any]) -> Dict[str, Any]:
        zeopp_info = context.get("zeopp_info")
        zeopp_cmd  = context.get("zeopp_command", "")
        mof        = context.get("mof", "unknown MOF")
        prop       = context.get("property", "")

        if not zeopp_cmd:
            return context

        prompt = f"""You are a Zeo++ command reviewer for MOF porous analysis. This simulation uses Zeo++.
Check the following Zeo++ command and parameters for physics-level issues BEFORE running.

MOF: {mof}
Property: {prop}
Command: {zeopp_cmd}
zeopp_info: {json.dumps(zeopp_info, indent=2) if zeopp_info else "N/A"}

The command is generated from a validated internal template. The system default probe radius is 1.2 Å (H2) when no specific probe gas is requested by the user — do NOT change it unless the user explicitly specified a different probe gas (e.g., N2, CO2).

Focus on:
- Wrong probe radius only when user explicitly requested a specific probe gas (e.g., N2: 1.82 Å, He: 1.3 Å, H2: 1.2 Å)
- Missing required flags for the property (e.g., -sa for surface area, -vol for pore volume, -chan for channel analysis)
- Incorrect number of samples (too low → inaccurate)
- Mismatched property vs subcommand

If no issues: reply with exactly 'OK' and nothing else.
If issues exist: reply with ONLY a JSON object:
{{"fixed_command": "<corrected full command string>", "reason": "<short explanation>"}}
"""
        messages = [
            SystemMessage(content="You are a Zeo++ command reviewer. Output only OK or the JSON object."),
            HumanMessage(content=prompt),
        ]

        print(f"\n[ZeoppErrorAgent] Pre-run review for {mof} / {prop} ...")
        resp = self.llm.invoke(messages).content.strip()

        if resp.upper() == "OK":
            print("[ZeoppErrorAgent] Pre-run review: no issues found.")
            return context

        print("[ZeoppErrorAgent] Pre-run review found issues. Proposed fix:\n")
        print(resp)

        zeopp_system = "You are a Zeo++ command reviewer. Output only OK or the JSON object."

        def _reinvoke_zeopp(instruction: str) -> str:
            new_prompt = prompt + f"\n\nUser instruction: {instruction}\nRevise accordingly."
            return self.llm.invoke([
                SystemMessage(content=zeopp_system),
                HumanMessage(content=new_prompt),
            ]).content.strip()

        from config import ask_user_confirmation
        action, resp = ask_user_confirmation("ZeoppErrorAgent", resp, reinvoke_fn=_reinvoke_zeopp)
        if action == "skip":
            return context

        try:
            if resp.startswith("```"):
                resp = "\n".join(resp.splitlines()[1:-1]).strip()
            data = json.loads(resp)
            if data.get("fixed_command"):
                context["zeopp_command"] = data["fixed_command"]
                print(f"[ZeoppErrorAgent] Command updated: {data['fixed_command']}")
        except Exception as e:
            print(f"[ZeoppErrorAgent] Pre-run review parse error: {e}")

        return context

    def _read_text_file(self, path: Optional[str]) -> str:
        if not path:
            return ""
        try:
            return Path(path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

    def _read_returncode(self, path: Optional[str]) -> Optional[int]:
        text = self._read_text_file(path).strip()
        if not text:
            return None
        try:
            return int(text.split()[0])
        except Exception:
            return None

    def _poll_zeopp_process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        results = context.setdefault("results", {})
        work_dir = context.get("work_dir", working_dir)
        done_marker = results.get("zeopp_done_marker") or os.path.join(work_dir, "DONE")
        failed_marker = results.get("zeopp_failed_marker") or os.path.join(work_dir, "FAILED")
        stdout_path = results.get("zeopp_stdout_path") or os.path.join(work_dir, "zeopp.stdout")
        stderr_path = results.get("zeopp_stderr_path") or os.path.join(work_dir, "zeopp.stderr")
        returncode_path = results.get("zeopp_returncode_path") or os.path.join(work_dir, "zeopp.returncode")
        pid = results.get("zeopp_pid")
        poll_interval_sec = int(context.get("zeopp_poll_interval_sec", 10))
        timeout_sec = int(context.get("zeopp_timeout_sec", 3600))
        start_time = time.time()

        record_job_event(
            context,
            "polling",
            message="Zeo++ job-status tracking started by ZeoppErrorAgent",
            metadata={
                "pid": pid,
                "poll_interval_sec": poll_interval_sec,
                "timeout_sec": timeout_sec,
                "done_marker": done_marker,
                "failed_marker": failed_marker,
            },
        )

        while True:
            if os.path.exists(failed_marker):
                stdout = self._read_text_file(stdout_path)
                stderr = self._read_text_file(stderr_path)
                returncode = self._read_returncode(returncode_path)
                results["zeopp_returncode"] = 1 if returncode is None else returncode
                results["zeopp_stdout"] = stdout
                results["zeopp_stderr"] = stderr
                results["zeopp_status"] = "run_failed"
                context["zeopp_submitted"] = False
                print(f"[ZeoppErrorAgent] WARNING: Zeo++ exited with code {results['zeopp_returncode']}")
                record_job_event(
                    context,
                    "failed",
                    message="Zeo++ command failed",
                    metadata={"returncode": results["zeopp_returncode"], "pid": pid},
                    last_error=(stderr or stdout or "")[:4000],
                )
                return context

            if os.path.exists(done_marker):
                stdout = self._read_text_file(stdout_path)
                stderr = self._read_text_file(stderr_path)
                returncode = self._read_returncode(returncode_path)
                results["zeopp_returncode"] = returncode
                results["zeopp_stdout"] = stdout
                results["zeopp_stderr"] = stderr
                results["zeopp_process_status"] = "completed"
                context["zeopp_submitted"] = False

                if returncode is None:
                    report = {
                        "ok": False,
                        "stage": "post_run",
                        "issues": [
                            {
                                "code": "process_returncode_missing",
                                "message": (
                                    "The Zeo++ process completed without a readable "
                                    "return-code record."
                                ),
                            }
                        ],
                        "evidence": [
                            "[SimMOF validation:process_returncode_missing] "
                            "The Zeo++ process completed without a readable "
                            "return-code record."
                        ],
                        "metadata": {},
                    }
                    store_validation_report(
                        context,
                        report,
                        key="zeopp_postflight_validation",
                    )
                    record_job_event(
                        context,
                        "validation_failed",
                        message="Zeo++ return-code validation failed",
                        metadata={"issues": report["issues"], "pid": pid},
                        last_error=report["evidence"][0],
                    )
                    return context

                if returncode != 0:
                    results["zeopp_status"] = "run_failed"
                    print(f"[ZeoppErrorAgent] WARNING: Zeo++ exited with code {returncode}")
                    record_job_event(
                        context,
                        "failed",
                        message="Zeo++ command failed",
                        metadata={"returncode": returncode, "pid": pid},
                        last_error=(stderr or stdout or "")[:4000],
                    )
                    return context

                postflight = validate_zeopp_postflight(context)
                store_validation_report(
                    context,
                    postflight,
                    key="zeopp_postflight_validation",
                )
                if not postflight["ok"]:
                    record_job_event(
                        context,
                        "validation_failed",
                        message=(
                            "Zeo++ exited with code 0 but semantic output "
                            "validation failed"
                        ),
                        metadata={
                            "returncode": returncode,
                            "pid": pid,
                            "issues": postflight["issues"],
                        },
                        last_error="\n".join(postflight["evidence"])[:4000],
                    )
                    return context

                results["zeopp_status"] = "ok"
                record_job_event(
                    context,
                    "done_ok",
                    message="Zeo++ process and semantic output validation completed",
                    metadata={"returncode": returncode, "pid": pid},
                )
                return context

            elapsed = time.time() - start_time
            if elapsed >= timeout_sec:
                if pid:
                    try:
                        os.killpg(int(pid), signal.SIGTERM)
                    except Exception:
                        try:
                            os.kill(int(pid), signal.SIGTERM)
                        except Exception:
                            pass
                stdout = self._read_text_file(stdout_path)
                stderr = self._read_text_file(stderr_path)
                results["zeopp_returncode"] = None
                results["zeopp_stdout"] = stdout
                results["zeopp_stderr"] = stderr
                results["zeopp_status"] = "run_failed"
                results["zeopp_error_kind"] = "timeout"
                context["zeopp_submitted"] = False
                record_job_event(
                    context,
                    "timeout",
                    message="Zeo++ job-status tracking timeout",
                    metadata={"elapsed_sec": elapsed, "pid": pid},
                    last_error=(stderr or stdout or "")[:4000],
                )
                return context

            results["zeopp_status"] = "running"
            record_job_event(
                context,
                "running",
                message="Zeo++ process still running",
                metadata={"elapsed_sec": elapsed, "pid": pid, "poll_interval_sec": poll_interval_sec},
            )
            time.sleep(poll_interval_sec)

    def _handle_structure_regeneration(
        self,
        context: Dict[str, Any],
        decision: Dict[str, Any],
        error_text: str,
    ) -> str:
        results = context.setdefault("results", {})
        reason = decision.get("reason")
        evidence = decision.get("evidence", [])
        request_structure_regeneration(
            context,
            software="zeopp",
            reason=reason or "LLM selected structure regeneration",
            action="refetch_or_clean_cif_and_regenerate_zeopp_command",
            metadata={
                "decision_source": "llm",
                "evidence": evidence,
                "error_text": error_text[-4000:],
                "validation_evidence": results.get(
                    "zeopp_validation_evidence",
                    [],
                ),
                "validation_observations": results.get(
                    "zeopp_validation_observations",
                    [],
                ),
            },
        )
        results["zeopp_status"] = "needs_structure_regeneration"
        return "return"

    def _handle_abort(
        self,
        context: Dict[str, Any],
        decision: Dict[str, Any],
        error_text: str,
    ) -> str:
        del error_text
        reason = decision.get("reason")
        print("[ZeoppErrorAgent] LLM selected abort:", reason)
        context.setdefault("results", {})["zeopp_status"] = "run_failed"
        return "return"

    def _handle_retry(
        self,
        context: Dict[str, Any],
        decision: Dict[str, Any],
        error_text: str,
    ) -> str:
        del error_text
        results = context.setdefault("results", {})
        fixed_info = decision.get("fixed_zeopp_info")
        fixed_command = decision.get("fixed_command")
        print("[ZeoppErrorAgent] LLM selected retry.")

        try:
            if fixed_info:
                context["zeopp_info"] = dict(fixed_info)
                mof = context.get("mof")
                if mof:
                    context["zeopp_info"]["MOF"] = mof
                work_dir = context.get("work_dir", working_dir)
                context["zeopp_command"] = self.zeopp_input_agent._get_zeopp_command(
                    context["zeopp_info"],
                    cif_dir=work_dir,
                )
            elif fixed_command:
                context["zeopp_command"] = fixed_command
            else:
                results["zeopp_status"] = "run_failed"
                results["zeopp_error_reason"] = (
                    "LLM selected retry without corrected zeopp_info or command"
                )
                return "return"
        except Exception as exc:
            results["zeopp_status"] = "run_failed"
            results["zeopp_error_reason"] = (
                f"Could not construct the LLM-proposed retry command: {exc}"
            )
            return "return"

        print("[ZeoppErrorAgent] Re-running Zeo++ with the registered retry handler...")
        context = self.zeopp_runner.run(context)
        retry_status = context.setdefault("results", {}).get("zeopp_status")
        if retry_status in ("submitted", "running") or context.get("zeopp_submitted"):
            context = self._poll_zeopp_process(context)
            retry_status = context.setdefault("results", {}).get("zeopp_status")

        if retry_status == "ok":
            print("[ZeoppErrorAgent] Retry succeeded and passed validation.")
            return "return"
        if retry_status in ("no_command", "input_failed"):
            return "return"
        return "continue"

    def _dispatch_action(
        self,
        context: Dict[str, Any],
        decision: Dict[str, Any],
        error_text: str,
    ) -> str:
        action = decision.get("action", "abort")
        handler = self.action_handlers.get(action, self.action_handlers["abort"])
        disposition = handler(context, decision, error_text)
        return disposition if disposition in {"return", "continue"} else "return"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        results = context.setdefault("results", {})
        status  = results.get("zeopp_status")

        if status in ("submitted", "running") or context.get("zeopp_submitted"):
            context = self._poll_zeopp_process(context)
            results = context.setdefault("results", {})
            status = results.get("zeopp_status")

        
        if status == "ok":
            print("[ZeoppErrorAgent] zeopp_status == ok → nothing to do.")
            return context

        
        if status in ("input_failed", "no_command"):
            print(f"[ZeoppErrorAgent] status = {status} → cannot fix here.")
            return context

        
        attempts = results.get("zeopp_attempts", 0)

        
        while attempts < self.max_retries:
            
            if status not in ("run_failed", "validation_failed", "retry"):
                print(f"[ZeoppErrorAgent] status = {status} → nothing to fix.")
                return context

            print("\n=== ZeoppErrorAgent: analyzing Zeo++ error (attempt "
                  f"{attempts + 1}/{self.max_retries}) ===")
            results["zeopp_attempts"] = attempts + 1

            error_text = "\n".join(
                str(results.get(key, ""))
                for key in ("zeopp_stdout", "zeopp_stderr", "zeopp_returncode")
            )
            
            llm_result = self._call_llm(context)
            action     = llm_result.get("action", "abort")
            reason     = llm_result.get("reason")
            evidence   = llm_result.get("evidence", [])

            results["zeopp_error_reason"] = reason
            results["zeopp_error_action"] = action
            results["zeopp_error_evidence"] = evidence

            disposition = self._dispatch_action(context, llm_result, error_text)
            results = context.setdefault("results", {})
            status = results.get("zeopp_status")
            attempts = results.get("zeopp_attempts", attempts + 1)
            if disposition == "return":
                return context

        
        print(f"[ZeoppErrorAgent] max_retries ({self.max_retries}) reached. Abort.")
        results["zeopp_status"] = "run_failed"
        return context
