from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_REGISTRY: dict = {
    "coremof2024": {
        "display_name": "CoRE MOF 2024",
        "aliases": ["coremof", "core-mof", "core mof", "coremof2024"],
        "cif_dir": os.getenv(
            "SIMMOF_DB_COREMOF2024",
            str(PROJECT_ROOT / "CSD-modified" / "CSD-modified" / "cifs" / "CR" / "FSR"),
        ),
        "description": "CSD-modified CoRE MOF 2024 dataset",
    },
}


def list_databases() -> list:
    return list(_REGISTRY.keys())


def resolve_db(name: str) -> Optional[dict]:
    key = name.strip().lower().replace("-", "").replace(" ", "")
    for db_key, info in _REGISTRY.items():
        aliases = [a.lower().replace("-", "").replace(" ", "") for a in info["aliases"]]
        if key == db_key or key in aliases:
            return {"key": db_key, **info}
    return None


def resolve_cif_dir(name: str) -> Optional[str]:
    entry = resolve_db(name)
    if entry:
        return entry["cif_dir"]
    return None


def db_summary() -> str:
    lines = ["Available MOF databases:"]
    for key, info in _REGISTRY.items():
        cif_dir = info["cif_dir"]
        exists = Path(cif_dir).exists()
        n = len(list(Path(cif_dir).glob("*.cif"))) if exists else 0
        lines.append(f"  [{key}] {info['display_name']} — {n} CIFs — {cif_dir}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(db_summary())
