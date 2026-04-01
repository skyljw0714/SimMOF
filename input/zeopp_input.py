import os
import json
import re
from typing import Dict, Any, Optional

from config import LLM_DEFAULT, ZEOPP_BIN as ZEOPP_BIN_PATH, working_dir, zeo_dir
from langchain.schema import HumanMessage, SystemMessage

from .zeopp.prompt import ZEOPP_DESCRIPTION, ZEOPP_EXAMPLES


def _pick_snippet(simulation_input: dict, software: str) -> str:
    if not simulation_input:
        return ""
    for s in (simulation_input.get("snippets") or []):
        if (s.get("software") == software) and (s.get("text") or "").strip():
            return s["text"].strip()
    return ""


ZEOPP_REPRO_PATCH_SYSTEM = """You are a careful text editor for Zeo++ (zeopp) command lines.
Return ONLY the patched command line text. No markdown. No explanations."""

ZEOPP_REPRO_PATCH_USER = """Patch the original Zeo++ command by applying ONLY the required replacement below.

HARD RULES:
1) MINIMAL CHANGE: Do not alter any tokens except where needed to apply REQUIRED REPLACEMENT.
2) Preserve all flags/options exactly as-is (probe radius, samples, etc.).
3) The command MUST remain a single shell command line.
4) Replace the CIF path argument so that it points to the new target CIF file.
   - If the original command contains a .cif path, replace ONLY that path.
   - If the original command contains no .cif path, append the target CIF path at the end.

REQUIRED REPLACEMENT (JSON):
{replacements_json}

ORIGINAL COMMAND:
<<<{original_text}>>>
"""


class ZeoppInputAgent:
    
    def __init__(self, llm=None):
        self.llm = llm or LLM_DEFAULT


    def _get_zeopp_rag_hints(self, context: Dict[str, Any], top_files: int = 5) -> str:
        try:
            from rag.agent import RagAgent

            rag_ctx = {
                "job_name": context.get("job_name") or "",
                "mof": context.get("mof") or "",
                "guest": context.get("guest") or "",
                "property": context.get("property") or "",
                "query_text": context.get("user_query") or context.get("query_text") or "",
            }

            agent = RagAgent(agent_name="RagAgent")
            r = agent.run_for_zeopp(rag_ctx, top_files=top_files)

            hints = (r.get("zeopp_hints") or "").strip()
            if hints:
                print("[RAG] Zeopp hints enabled")
            else:
                print("[RAG] no relevant Zeopp hints")
            return hints

        except Exception as e:
            print(f"[RAG] Zeopp hints disabled due to error: {e}")
            return ""


    def _get_zeopp_command(self, zeopp_info: dict, cif_dir: str) -> str:
        mof = zeopp_info["MOF"]
        command = zeopp_info["command"]
        probe_radius = zeopp_info.get("probe_radius")
        num_samples = zeopp_info.get("num_samples")

        cif_file = os.path.join(cif_dir, f"{mof}.cif") if cif_dir else f"{mof}.cif"

        cmd = [os.path.join(zeo_dir, "network")] + command.split()

        if probe_radius is not None:
            radius = str(probe_radius)
            if num_samples is not None:
                cmd += [radius, radius, str(num_samples)]
            elif ("block" in command) or ("chan" in command):
                cmd += [radius]
        elif num_samples is not None:
            cmd += [str(num_samples)]

        cmd.append(cif_file)
        return " ".join(cmd)

    def _validate_zeopp_info(self, zeopp_info: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(zeopp_info, dict):
            return {}

        mof = zeopp_info.get("MOF")
        command = zeopp_info.get("command")

        if not mof or not isinstance(mof, str):
            return {}
        if not command or not isinstance(command, str):
            return {}

        if "probe_radius" in zeopp_info and zeopp_info["probe_radius"] is not None:
            try:
                zeopp_info["probe_radius"] = float(zeopp_info["probe_radius"])
            except Exception:
                zeopp_info["probe_radius"] = None

        if "num_samples" in zeopp_info and zeopp_info["num_samples"] is not None:
            try:
                zeopp_info["num_samples"] = int(float(zeopp_info["num_samples"]))
            except Exception:
                zeopp_info["num_samples"] = None

        return zeopp_info

    def _get_zeopp_info(self, raw_query: str, rag_hints: str = "") -> dict:
        prompt = f"""
You are an expert in Zeo++ (zeopp) command-line usage for MOF analysis.
{ZEOPP_DESCRIPTION}

Below are some examples of how to convert user queries to Zeo++ command parameters:
{ZEOPP_EXAMPLES}

Optional RAG_HINTS (may be irrelevant; use only if clearly applicable; keep defaults safe):
{rag_hints}

Now, given the following user query, extract the necessary parameters and generate a Zeo++ command and arguments in structured JSON format.
User query: "{raw_query}"

Strict output rules:
- Return ONLY a JSON object (no markdown, no extra keys beyond examples).
- Use safe defaults if unsure; do NOT invent exotic flags.
- If a parameter is not needed, set it to null or omit it following the example style.
"""
        messages = [
            SystemMessage(content="You are a Zeo++ command expert. Output only the JSON object."),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)
        try:
            zeopp_info = json.loads(response.content)
        except Exception as e:
            print("[ZeoppInputAgent] LLM parsing error:", e)
            print("Raw response:", response.content)
            return {}

        return self._validate_zeopp_info(zeopp_info)

    
    
    
    def _llm_patch_zeopp_command(self, original_text: str, target_cif_path: str) -> str:
        if self.llm is None:
            raise ValueError("LLM is required for Zeopp reproduce patching (self.llm is None).")

        replacements = {"target_cif_path": target_cif_path}
        rep_json = json.dumps(replacements, ensure_ascii=False, indent=2)

        resp = self.llm.invoke([
            SystemMessage(content=ZEOPP_REPRO_PATCH_SYSTEM),
            HumanMessage(content=ZEOPP_REPRO_PATCH_USER.format(
                replacements_json=rep_json,
                original_text=original_text
            )),
        ])

        out = (resp.content or "").strip()

        
        if out.startswith("```"):
            lines = out.splitlines()
            if lines and lines[0].lstrip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            out = "\n".join(lines).strip()

        if not out:
            raise ValueError("LLM returned empty patched Zeopp command.")

        
        out = " ".join(out.splitlines()).strip()
        return out

    def _looks_like_zeopp_cmd(self, s: str) -> bool:
        s = (s or "").strip()
        if not s:
            return False
        
        return ("network" in s) or ("zeo++" in s) or ("zeopp" in s)

    def _build_plan_query(self, mof: str, prop: str, guest: Optional[str] = None) -> str:
        prop_text = prop.replace("_", " ")
        q = f"Calculate {prop_text} of {mof}"
        if guest:
            q += f" with guest {guest}"
        return q

    
    
    
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        work_dir = context.get("work_dir", working_dir)
        mof = context.get("mof")
        prop = context.get("property")
        guest = context.get("guest")

        simulation_input = context.get("simulation_input") or {}
        snippet = _pick_snippet(simulation_input, "Zeopp")

        mode = "standard"
        if snippet and self._looks_like_zeopp_cmd(snippet):
            mode = "reproduce"

        
        if not mof:
            
            mof = context.get("mof_name") or context.get("framework_name") or "MOF"
            context["mof"] = mof

        target_cif_path = os.path.join(work_dir, f"{mof}.cif")

        
        if mode == "reproduce":
            print("[ZeoppInputAgent] reproduce mode detected")
            print("[RAG] skipped (reproduce mode)")

            try:
                patched_cmd = self._llm_patch_zeopp_command(
                    original_text=snippet,
                    target_cif_path=target_cif_path,
                )

                zeopp_bin = str(ZEOPP_BIN_PATH)
                toks = patched_cmd.split()
                if toks and toks[0] == "network":
                    toks[0] = zeopp_bin
                patched_cmd = " ".join(toks)

                cmd_middle = " ".join(toks[1:-1]) if len(toks) >= 3 else ""

                context["zeopp_command"] = patched_cmd
                context["zeopp_info"] = {
                    "MOF": mof,
                    "command": cmd_middle,
                    "probe_radius": None,
                    "num_samples": None,
                }
                context.setdefault("results", {})["zeopp_status"] = "ok"
                context.setdefault("results", {})["zeopp_mode"] = "reproduce"
                print(f"[ZeoppInputAgent] Zeo++ command (patched): {patched_cmd}")
                return context

            except Exception as e:
                print(f"[ZeoppInputAgent] reproduce patch failed -> fallback to standard: {e}")
                

        
        single_query = self._build_plan_query(mof=mof, prop=prop, guest=guest)

        rag_hints = self._get_zeopp_rag_hints(context, top_files=5)

        zeopp_info = self._get_zeopp_info(single_query, rag_hints=rag_hints)
        if not zeopp_info:
            print("[ZeoppInputAgent] ERROR: zeopp_info is empty.")
            context.setdefault("results", {})["zeopp_status"] = "input_failed"
            context.setdefault("results", {})["zeopp_mode"] = "standard"
            return context

        
        zeopp_info["MOF"] = mof

        zeopp_command = self._get_zeopp_command(zeopp_info, cif_dir=work_dir)
        print(f"[ZeoppInputAgent] Zeo++ command: {zeopp_command}")

        context["zeopp_info"] = zeopp_info
        context["zeopp_command"] = zeopp_command
        context.setdefault("results", {})["zeopp_status"] = "ok"
        context.setdefault("results", {})["zeopp_mode"] = "standard"
        return context
