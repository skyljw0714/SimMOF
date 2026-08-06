from __future__ import annotations

import json
import os
import time
from pathlib import Path
from threading import Lock, local
from typing import Any, Dict, Optional

_LOG_DIR = Path(os.getenv("SIMMOF_WORKING_DIR", Path(__file__).resolve().parents[1] / "working_dir"))
_PID = os.getpid()
_CALLS_PATH     = _LOG_DIR / f"llm_calls_{_PID}.jsonl"
_DECISIONS_PATH = _LOG_DIR / f"llm_decisions_{_PID}.jsonl"
_lock = Lock()
_ctx = local()


def set_llm_context(agent: Optional[str] = None, label: Optional[str] = None) -> None:
    _ctx.agent = agent
    _ctx.label = label


def clear_llm_context() -> None:
    _ctx.agent = None
    _ctx.label = None


def log_llm_call(
    model: str,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
) -> None:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "agent": getattr(_ctx, "agent", None),
        "label": getattr(_ctx, "label", None),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
    with _lock:
        with open(_CALLS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def log_llm_decision(
    agent: str,
    label: str,
    decision: Any,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    ctx = context or {}
    entry = {
        "ts":       time.strftime("%Y-%m-%d %H:%M:%S"),
        "agent":    agent,
        "label":    label,
        "mof":      ctx.get("mof") or ctx.get("MOF"),
        "guest":    ctx.get("guest") or ctx.get("Guest"),
        "property": ctx.get("property") or ctx.get("Property"),
        "plan":     ctx.get("plan_name") or ctx.get("job_name"),
        "job":      ctx.get("job_id"),
        "decision": decision,
    }
    with _lock:
        with open(_DECISIONS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
