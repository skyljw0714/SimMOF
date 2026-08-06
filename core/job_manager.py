from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = Path(os.getenv("SIMMOF_WORKING_DIR", PROJECT_ROOT / "working_dir")) / f"job_state_{os.getpid()}.sqlite3"


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _json_loads(value: str | None) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        data = json.loads(value)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _merge_dicts(old: Dict[str, Any], new: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(old or {})
    if new:
        merged.update(new)
    return merged


def parse_scheduler_job_id(stdout: str) -> Optional[str]:
    text = (stdout or "").strip()
    if not text:
        return None
    match = re.search(r"\b(\d+(?:\.[A-Za-z0-9_.-]+)?)\b", text)
    if match:
        return match.group(1)
    return text.split()[0]


@dataclass
class SchedulerStatus:
    state: str
    raw: str = ""
    returncode: int = 0


class JobStateStore:

    def __init__(self, db_path: Optional[str | Path] = None):
        self.db_path = Path(db_path or DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    plan_name TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    agent TEXT,
                    software TEXT,
                    mof TEXT,
                    guest TEXT,
                    property TEXT,
                    work_dir TEXT,
                    qsub_path TEXT,
                    scheduler_job_id TEXT,
                    status TEXT NOT NULL,
                    retry_count INTEGER DEFAULT 0,
                    submit_returncode INTEGER,
                    submit_stdout TEXT,
                    submit_stderr TEXT,
                    context_path TEXT,
                    metadata_json TEXT,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (plan_name, job_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS job_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plan_name TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )

    def upsert_job(
        self,
        *,
        plan_name: str,
        job_id: str,
        status: str,
        agent: Optional[str] = None,
        software: Optional[str] = None,
        mof: Optional[str] = None,
        guest: Optional[str] = None,
        property: Optional[str] = None,
        work_dir: Optional[str] = None,
        qsub_path: Optional[str] = None,
        scheduler_job_id: Optional[str] = None,
        retry_count: Optional[int] = None,
        submit_returncode: Optional[int] = None,
        submit_stdout: Optional[str] = None,
        submit_stderr: Optional[str] = None,
        context_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        last_error: Optional[str] = None,
        message: str = "",
    ) -> None:
        now = _now()
        with self._connect() as conn:
            previous = conn.execute(
                "SELECT metadata_json FROM jobs WHERE plan_name = ? AND job_id = ?",
                (plan_name, job_id),
            ).fetchone()
            previous_metadata = _json_loads(previous["metadata_json"]) if previous else {}
            merged_metadata = _merge_dicts(previous_metadata, metadata)
            metadata_json = _json_dumps(merged_metadata)
            conn.execute(
                """
                INSERT INTO jobs (
                    plan_name, job_id, agent, software, mof, guest, property,
                    work_dir, qsub_path, scheduler_job_id, status, retry_count,
                    submit_returncode, submit_stdout, submit_stderr,
                    context_path, metadata_json, last_error, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(plan_name, job_id) DO UPDATE SET
                    agent=COALESCE(excluded.agent, jobs.agent),
                    software=COALESCE(excluded.software, jobs.software),
                    mof=COALESCE(excluded.mof, jobs.mof),
                    guest=COALESCE(excluded.guest, jobs.guest),
                    property=COALESCE(excluded.property, jobs.property),
                    work_dir=COALESCE(excluded.work_dir, jobs.work_dir),
                    qsub_path=COALESCE(excluded.qsub_path, jobs.qsub_path),
                    scheduler_job_id=COALESCE(excluded.scheduler_job_id, jobs.scheduler_job_id),
                    status=excluded.status,
                    retry_count=COALESCE(excluded.retry_count, jobs.retry_count),
                    submit_returncode=COALESCE(excluded.submit_returncode, jobs.submit_returncode),
                    submit_stdout=COALESCE(excluded.submit_stdout, jobs.submit_stdout),
                    submit_stderr=COALESCE(excluded.submit_stderr, jobs.submit_stderr),
                    context_path=COALESCE(excluded.context_path, jobs.context_path),
                    metadata_json=excluded.metadata_json,
                    last_error=COALESCE(excluded.last_error, jobs.last_error),
                    updated_at=excluded.updated_at
                """,
                (
                    plan_name,
                    job_id,
                    agent,
                    software,
                    mof,
                    guest,
                    property,
                    work_dir,
                    qsub_path,
                    scheduler_job_id,
                    status,
                    retry_count,
                    submit_returncode,
                    submit_stdout,
                    submit_stderr,
                    context_path,
                    metadata_json,
                    last_error,
                    now,
                    now,
                ),
            )
            conn.execute(
                """
                INSERT INTO job_events (
                    plan_name, job_id, status, message, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (plan_name, job_id, status, message, metadata_json, now),
            )

    def get_job(self, plan_name: str, job_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE plan_name = ? AND job_id = ?",
                (plan_name, job_id),
            ).fetchone()
        return dict(row) if row else None


def _marker_elapsed_metadata(context: Dict[str, Any], status: str) -> Dict[str, Any]:
    if status not in {"done_ok", "failed", "timeout"}:
        return {}

    work_dir = context.get("work_dir")
    if not work_dir:
        return {}

    work_path = Path(work_dir)
    start_path = work_path / "START"
    terminal_path = None
    if status == "done_ok" and (work_path / "DONE").exists():
        terminal_path = work_path / "DONE"
    elif (work_path / "FAILED").exists():
        terminal_path = work_path / "FAILED"

    if not (start_path.exists() and terminal_path and terminal_path.exists()):
        return {}

    elapsed = max(0.0, terminal_path.stat().st_mtime - start_path.stat().st_mtime)
    return {
        "marker_elapsed_sec": elapsed,
        "marker_start_path": str(start_path),
        "marker_terminal_path": str(terminal_path),
    }


class JobManager:
    def __init__(self, store: Optional[JobStateStore] = None):
        self.store = store or JobStateStore()

    @staticmethod
    def _ids(context: Dict[str, Any]) -> tuple[str, str]:
        return (
            str(context.get("plan_name") or context.get("job_name") or "unknown_plan"),
            str(context.get("job_id") or "unknown_job"),
        )

    @staticmethod
    def _software(context: Dict[str, Any]) -> str:
        agent = str(context.get("agent") or "")
        if agent.endswith("Agent"):
            return agent[:-5]
        return agent or str(context.get("software") or "")

    def record(
        self,
        context: Dict[str, Any],
        status: str,
        *,
        message: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        last_error: Optional[str] = None,
    ) -> None:
        plan_name, job_id = self._ids(context)
        event_metadata = _merge_dicts(metadata or {}, _marker_elapsed_metadata(context, status))
        self.store.upsert_job(
            plan_name=plan_name,
            job_id=job_id,
            status=status,
            agent=context.get("agent"),
            software=self._software(context),
            mof=context.get("mof"),
            guest=context.get("guest"),
            property=context.get("property"),
            work_dir=context.get("work_dir"),
            qsub_path=context.get("qsub_script")
            or context.get("qsub_path")
            or context.get("input_file"),
            scheduler_job_id=context.get("scheduler_job_id")
            or context.get("pbs_job_id")
            or context.get("raspa_job_id")
            or context.get("vasp_job_id"),
            retry_count=context.get("retry_count")
            or context.get("raspa_retry")
            or context.get("vasp_retry"),
            context_path=context.get("latest_context_path"),
            metadata=event_metadata,
            last_error=last_error,
            message=message,
        )

    def record_submission(
        self,
        context: Dict[str, Any],
        *,
        qsub_path: str,
        returncode: int,
        stdout: str,
        stderr: str,
        status: str,
        scheduler_job_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        plan_name, job_id = self._ids(context)
        jid = scheduler_job_id or parse_scheduler_job_id(stdout)
        if jid:
            context["scheduler_job_id"] = jid
        self.store.upsert_job(
            plan_name=plan_name,
            job_id=job_id,
            status=status,
            agent=context.get("agent"),
            software=self._software(context),
            mof=context.get("mof"),
            guest=context.get("guest"),
            property=context.get("property"),
            work_dir=context.get("work_dir"),
            qsub_path=qsub_path,
            scheduler_job_id=jid,
            submit_returncode=returncode,
            submit_stdout=stdout,
            submit_stderr=stderr,
            retry_count=context.get("retry_count")
            or context.get("raspa_retry")
            or context.get("vasp_retry"),
            context_path=context.get("latest_context_path"),
            metadata=metadata,
            message=f"submitted via {qsub_path}",
        )

    def poll_scheduler(self, scheduler_job_id: str) -> SchedulerStatus:
        try:
            proc = subprocess.run(
                ["qstat", "-f", scheduler_job_id],
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception as exc:
            return SchedulerStatus(state="scheduler_unavailable", raw=str(exc), returncode=1)

        raw = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0:
            return SchedulerStatus(state="not_in_scheduler", raw=raw, returncode=proc.returncode)

        match = re.search(r"job_state\s*=\s*(\S+)", raw)
        if not match:
            return SchedulerStatus(state="unknown", raw=raw, returncode=proc.returncode)

        code = match.group(1)
        mapping = {
            "Q": "queued",
            "R": "running",
            "E": "exiting",
            "C": "completed",
            "H": "held",
            "W": "waiting",
        }
        return SchedulerStatus(state=mapping.get(code, code), raw=raw, returncode=proc.returncode)


_DEFAULT_MANAGER: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    global _DEFAULT_MANAGER
    if _DEFAULT_MANAGER is None:
        _DEFAULT_MANAGER = JobManager()
    return _DEFAULT_MANAGER


def record_job_event(
    context: Dict[str, Any],
    status: str,
    *,
    message: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    last_error: Optional[str] = None,
) -> None:
    try:
        get_job_manager().record(
            context,
            status,
            message=message,
            metadata=metadata,
            last_error=last_error,
        )
    except Exception as exc:
        print(f"[JobManager] warning: failed to record {status}: {exc}")


def record_scheduler_status(context: Dict[str, Any]) -> None:
    scheduler_job_id = (
        context.get("scheduler_job_id")
        or context.get("pbs_job_id")
        or context.get("raspa_job_id")
        or context.get("vasp_job_id")
    )
    if not scheduler_job_id:
        return
    try:
        manager = get_job_manager()
        status = manager.poll_scheduler(str(scheduler_job_id))
        manager.record(
            context,
            status.state,
            message=f"qstat state for {scheduler_job_id}: {status.state}",
            metadata={
                "scheduler_job_id": scheduler_job_id,
                "scheduler_returncode": status.returncode,
                "scheduler_raw_excerpt": status.raw[:1000],
            },
        )
    except Exception as exc:
        print(f"[JobManager] warning: scheduler poll failed: {exc}")
