import csv
import json
import traceback
from pathlib import Path
from typing import Any, Dict

from config import BATCH_WORK_ROOT


class BatchWorkflow:
    def __init__(self, agent, max_workers: int = 4):
        self.agent = agent
        self.max_workers = max_workers

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        cif_dir = Path(ctx.get("cif_dir") or "")
        if not cif_dir.exists():
            raise ValueError(f"[BatchWorkflow] cif_dir not found: {cif_dir}")

        cif_files = sorted(cif_dir.glob("*.cif"))
        if not cif_files:
            raise ValueError(f"[BatchWorkflow] No CIF files found in: {cif_dir}")

        batch_root = Path(ctx.get("work_dir") or str(BATCH_WORK_ROOT)) / ctx.get("job_name", "batch")
        batch_root.mkdir(parents=True, exist_ok=True)

        print(f"[BatchWorkflow] Running {ctx.get('agent')} on {len(cif_files)} CIFs in {cif_dir}")

        rows = []
        failed = []

        for cif_path in cif_files:
            mof_name = cif_path.stem
            mof_work_dir = batch_root / mof_name
            mof_work_dir.mkdir(parents=True, exist_ok=True)

            mof_ctx = {
                **ctx,
                "mof": mof_name,
                "mof_path": str(cif_path),
                "cif_path": str(cif_path),
                "work_dir": str(mof_work_dir),
                "plan_root": str(mof_work_dir),
                "job_name": f"{ctx.get('job_name', 'batch')}_{mof_name}",
                "job_id": f"{ctx.get('job_id', 'batch_job')}_{mof_name}",
                "results": {},
            }

            try:
                out = self.agent.run(mof_ctx)
                row = {"mof": mof_name, "cif": str(cif_path), "status": "ok"}
                if isinstance(out, dict):
                    row.update(_flatten_result(out))
                rows.append(row)
                print(f"  [OK] {mof_name}")
            except Exception as e:
                print(f"  [FAIL] {mof_name}: {e}")
                failed.append({"mof": mof_name, "error": str(e), "traceback": traceback.format_exc()})
                rows.append({"mof": mof_name, "cif": str(cif_path), "status": "failed", "error": str(e)})

        summary_csv = batch_root / "batch_results.csv"
        if rows:
            all_keys = list(dict.fromkeys(k for r in rows for k in r))
            with open(summary_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)

        if failed:
            fail_log = batch_root / "batch_failed.json"
            with open(fail_log, "w", encoding="utf-8") as f:
                json.dump(failed, f, indent=2, ensure_ascii=False)

        n_ok = sum(1 for r in rows if r.get("status") == "ok")
        print(f"[BatchWorkflow] Done: {n_ok}/{len(cif_files)} succeeded. Results: {summary_csv}")

        return {
            "batch_results": rows,
            "batch_summary_csv": str(summary_csv),
            "batch_root": str(batch_root),
            "n_total": len(cif_files),
            "n_ok": n_ok,
            "n_failed": len(failed),
        }


def _flatten_result(out: dict) -> dict:
    flat = {}
    skip_keys = {"results", "upstream_jobs", "upstream_plans", "simulation_input",
                 "paths", "context", "debug", "batch_results"}
    for k, v in out.items():
        if k in skip_keys:
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            flat[k] = v
        elif isinstance(v, dict):
            for sk, sv in v.items():
                if isinstance(sv, (str, int, float, bool)) or sv is None:
                    flat[f"{k}.{sk}"] = sv
    return flat
