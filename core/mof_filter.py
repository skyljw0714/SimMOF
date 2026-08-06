from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Optional

import csv



_DB_METADATA: dict = {
    "coremof2024": str(
        Path(__file__).resolve().parent.parent
        / "CSD-modified" / "CSD-modified" / "CR_data_CSD_modified_20250227.csv"
    ),
}

_METAL_COL = "Metal Types"



def apply_metal_filter(
    cif_dir: str,
    metals: List[str],
    db_key: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> str:
    cif_dir = Path(cif_dir)
    metals_norm = [m.strip().capitalize() for m in metals]
    tag = "_".join(metals_norm)

    if output_dir is None:
        output_dir = cif_dir.parent / f"{cif_dir.name}_filtered_{tag}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for old in output_dir.glob("*.cif"):
        old.unlink()

    if db_key and db_key in _DB_METADATA:
        matched = _filter_by_csv(cif_dir, metals_norm, db_key)
    else:
        matched = _filter_by_ase(cif_dir, metals_norm)

    if not matched:
        raise ValueError(
            f"No CIFs found matching metals {metals_norm} in {cif_dir}"
        )

    for src in matched:
        dst = output_dir / src.name
        if not dst.exists():
            try:
                os.symlink(src.resolve(), dst)
            except OSError:
                shutil.copy2(src, dst)

    print(f"[mof_filter] {len(matched)} CIFs matched {metals_norm} → {output_dir}")
    return str(output_dir)



def _filter_by_csv(cif_dir: Path, metals: List[str], db_key: str) -> List[Path]:
    csv_path = _DB_METADATA[db_key]
    if not Path(csv_path).exists():
        print(f"[mof_filter] metadata CSV not found: {csv_path}, falling back to ASE")
        return _filter_by_ase(cif_dir, metals)

    matched_ids = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metal_str = row.get(_METAL_COL, "") or ""
            if any(m in metal_str for m in metals):
                coreid = row.get("coreid", "").strip()
                if coreid:
                    matched_ids.add(coreid)

    results = []
    for cif in sorted(cif_dir.glob("*.cif")):
        stem = cif.stem
        if stem in matched_ids or any(stem.startswith(cid) or cid in stem for cid in matched_ids):
            results.append(cif)

    return results



def _filter_by_ase(cif_dir: Path, metals: List[str]) -> List[Path]:
    try:
        import ase.io
    except ImportError:
        raise ImportError("ASE is required for directory-based metal filtering.")

    metals_cap = {m.capitalize() for m in metals}
    results = []
    cifs = sorted(cif_dir.glob("*.cif"))
    print(f"[mof_filter] ASE scan: {len(cifs)} CIFs in {cif_dir}")

    for cif in cifs:
        try:
            atoms = ase.io.read(str(cif))
            symbols = set(atoms.get_chemical_symbols())
            if symbols & metals_cap:
                results.append(cif)
        except Exception:
            continue

    return results
