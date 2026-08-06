from pathlib import Path
from typing import Any, Dict, Optional

from langchain.schema import HumanMessage, SystemMessage

from config import INTERACTION_MODE, LLM_DEFAULT, ask_user_confirmation
from core.llm_logging import log_llm_decision, set_llm_context


FINAL_INPUT_REVIEW_SYSTEM = """You are a careful simulation input file editor.
Return ONLY the complete revised input file text. No markdown. No explanations."""

FINAL_INPUT_REVIEW_USER = """Revise the generated {software} input file according to the user instruction.

Rules:
1) Preserve all existing settings unless the user instruction explicitly requires a change.
2) Keep the file valid for {software}.
3) Do not add prose, markdown fences, or explanations.

Context:
{context_summary}

Current input file:
<<<{input_text}>>>

User instruction:
{instruction}
"""


def maybe_interactive_review_input_file(
    *,
    software: str,
    path: str,
    context: Optional[Dict[str, Any]] = None,
    llm: Any = None,
    label: Optional[str] = None,
) -> str:
    if INTERACTION_MODE != "interactive":
        return path

    input_path = Path(path)
    try:
        input_text = input_path.read_text()
    except Exception as exc:
        print(f"[{label or software}] Final input review skipped: cannot read {input_path} ({exc})")
        return path

    print(f"\n[{label or software}] Generated input file for final review: {input_path}")
    print(input_text)

    context = context or {}
    context_summary = "\n".join(
        f"{key}: {context.get(key)}"
        for key in ("job_name", "mof", "guest", "property", "vasp_stage", "vasp_calc_type")
        if context.get(key) is not None
    ) or "(None)"

    def _reinvoke(instruction: str) -> str:
        set_llm_context(label or f"{software}InputAgent", "final_input_review")
        model = llm or LLM_DEFAULT
        response = model.invoke([
            SystemMessage(content=FINAL_INPUT_REVIEW_SYSTEM),
            HumanMessage(content=FINAL_INPUT_REVIEW_USER.format(
                software=software,
                context_summary=context_summary,
                input_text=input_text,
                instruction=instruction,
            )),
        ])
        revised = (response.content or "").strip()
        if revised.startswith("```"):
            lines = revised.splitlines()
            if lines and lines[0].lstrip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            revised = "\n".join(lines).strip()
        return revised

    action, revised = ask_user_confirmation(
        label or f"{software}InputAgent",
        f"Generated {software} input file: {input_path}",
        reinvoke_fn=_reinvoke,
        required=False,
    )
    if action == "apply" and revised != f"Generated {software} input file: {input_path}":
        if revised.strip():
            input_path.write_text(revised.rstrip() + "\n")
            print(f"[{label or software}] Final input updated per user instruction: {input_path}")
            try:
                log_llm_decision(
                    label or f"{software}InputAgent",
                    "final_input_review",
                    {"input_file": str(input_path), "updated": True, "preview": revised[:1000]},
                    context,
                )
            except Exception:
                pass
    return path
