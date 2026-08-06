
import json
import os
import time
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, Any, Optional

from config import working_dir


TIMING_LOG_PATH = Path(working_dir) / f"timing_log_{os.getpid()}.jsonl"


def log_timing(event: Dict[str, Any]):
    TIMING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(TIMING_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


@contextmanager
def timer(
    label: str,
    category: str = "block",
    context: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
):
    start = time.perf_counter()
    wall_start = time.strftime("%Y-%m-%d %H:%M:%S")
    status = "ok"

    try:
        yield

    except Exception:
        status = "error"
        raise

    finally:
        end = time.perf_counter()
        elapsed = end - start
        wall_end = time.strftime("%Y-%m-%d %H:%M:%S")

        event = {
            "category": category,
            "label": label,
            "elapsed_sec": elapsed,
            "status": status,
            "started_at": wall_start,
            "ended_at": wall_end,
        }

        if context:
            event.update({
                "plan": context.get("plan_name") or context.get("job_name"),
                "job": context.get("job_id"),
                "agent": context.get("agent"),
                "mof": context.get("mof"),
                "guest": context.get("guest"),
                "property": context.get("property"),
                "work_dir": context.get("work_dir"),
            })

        if extra:
            event.update(extra)

        print(f"[TIMER] {category} | {label}: {elapsed:.3f} sec | status={status}")
        log_timing(event)


def timed_call(
    label: str,
    fn,
    *args,
    category: str = "call",
    context: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    start = time.perf_counter()
    wall_start = time.strftime("%Y-%m-%d %H:%M:%S")
    status = "ok"

    try:
        return fn(*args, **kwargs)

    except Exception:
        status = "error"
        raise

    finally:
        end = time.perf_counter()
        elapsed = end - start
        wall_end = time.strftime("%Y-%m-%d %H:%M:%S")

        event = {
            "category": category,
            "label": label,
            "elapsed_sec": elapsed,
            "status": status,
            "started_at": wall_start,
            "ended_at": wall_end,
        }

        if context:
            event.update({
                "plan": context.get("plan_name") or context.get("job_name"),
                "job": context.get("job_id"),
                "agent": context.get("agent"),
                "mof": context.get("mof"),
                "guest": context.get("guest"),
                "property": context.get("property"),
                "work_dir": context.get("work_dir"),
            })

        if extra:
            event.update(extra)

        print(f"[TIMER] {category} | {label}: {elapsed:.3f} sec | status={status}")
        log_timing(event)