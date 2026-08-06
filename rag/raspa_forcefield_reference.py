from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import RASPA_DIR


def _raspa_share_dir(raspa_dir: Optional[str] = None) -> Path:
    return Path(raspa_dir or RASPA_DIR) / "share" / "raspa"


def _read_lines(path: Path) -> List[str]:
    if not path.is_file():
        return []
    return path.read_text(errors="ignore").splitlines()


def _comment_summary(paths: List[Path], limit: int = 3) -> List[str]:
    comments: List[str] = []
    seen = set()
    for path in paths:
        for raw in _read_lines(path):
            stripped = raw.strip()
            if not stripped.startswith("#"):
                continue
            text = stripped.lstrip("#").strip()
            if not text or text.lower().startswith(("number of", "type ")):
                continue
            if text not in seen:
                comments.append(text)
                seen.add(text)
            if len(comments) >= limit:
                return comments
    return comments


def _count_data_entries(path: Path) -> int:
    count = 0
    for raw in _read_lines(path):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if len(line.split()) == 1:
            continue
        count += 1
    return count


def load_forcefield_catalog(raspa_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    root = _raspa_share_dir(raspa_dir) / "forcefield"
    if not root.is_dir():
        return []

    catalog: List[Dict[str, Any]] = []
    for directory in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name):
        pseudo = directory / "pseudo_atoms.def"
        mixing = directory / "force_field_mixing_rules.def"
        specific = directory / "force_field.def"
        catalog.append(
            {
                "name": directory.name,
                "path": str(directory),
                "has_pseudo_atoms": pseudo.is_file(),
                "has_mixing_rules": mixing.is_file(),
                "has_specific_pairs": specific.is_file(),
                "pseudo_atom_entries": _count_data_entries(pseudo),
                "mixing_rule_entries": _count_data_entries(mixing),
                "specific_pair_entries": _count_data_entries(specific),
                "file_comments": _comment_summary([pseudo, mixing, specific]),
            }
        )
    return catalog


def load_molecule_definition_catalog(
    molecule_name: Optional[str],
    raspa_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if not molecule_name:
        return []
    root = _raspa_share_dir(raspa_dir) / "molecules"
    if not root.is_dir():
        return []

    catalog: List[Dict[str, Any]] = []
    for family in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name):
        definition = family / f"{molecule_name}.def"
        if not definition.is_file():
            continue
        nonempty = [line.strip() for line in _read_lines(definition) if line.strip()]
        catalog.append(
            {
                "family": family.name,
                "path": str(definition),
                "comments": _comment_summary([definition], limit=4),
                "preview": nonempty[:12],
            }
        )
    return catalog


def _parse_mixing_entries(path: Path) -> Dict[str, Dict[str, Any]]:
    entries: Dict[str, Dict[str, Any]] = {}
    for line_no, raw in enumerate(_read_lines(path), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2 or parts[0].isdigit():
            continue
        if parts[0] in {"Lorentz-Berthelot", "Jorgensen", "WaldmanHagler"}:
            continue
        entries[parts[0]] = {
            "atom_type": parts[0],
            "line": raw,
            "line_no": line_no,
            "path": str(path),
        }
    return entries


def _parse_pseudo_entries(path: Path) -> Dict[str, Dict[str, Any]]:
    entries: Dict[str, Dict[str, Any]] = {}
    for line_no, raw in enumerate(_read_lines(path), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3 or parts[0].isdigit():
            continue
        entries[parts[0]] = {
            "atom_type": parts[0],
            "element": parts[2],
            "line": raw,
            "line_no": line_no,
            "path": str(path),
        }
    return entries


def parse_missing_vdw_pairs(text: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    pattern = re.compile(
        r"WARNING:\s+THERE ARE ATOM-PAIRS WITH NO VDW INTERACTION\s+(.+?)(?:\s+\(maximum|\n|$)",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(text or ""):
        unresolved: List[str] = []
        found_hyphen_pair = False
        for token in match.group(1).split():
            if "-" not in token:
                unresolved.append(token.strip())
                continue
            left, right = token.split("-", 1)
            if left.strip() and right.strip():
                pairs.append((left.strip(), right.strip()))
                found_hyphen_pair = True
        if not found_hyphen_pair:
            pairs.extend(
                (unresolved[index], unresolved[index + 1])
                for index in range(0, len(unresolved) - 1, 2)
                if unresolved[index] and unresolved[index + 1]
            )
    return pairs


def _element_hint(atom_type: str) -> Optional[str]:
    match = re.match(r"([A-Z][a-z]?)", re.sub(r"[^A-Za-z]", "", atom_type or ""))
    return match.group(1) if match else None


def load_missing_vdw_candidates(
    error_msg: str,
    selected_forcefield: Optional[str],
    raspa_dir: Optional[str] = None,
) -> Dict[str, Any]:
    share_dir = _raspa_share_dir(raspa_dir)
    ff_root = share_dir / "forcefield"
    pairs = parse_missing_vdw_pairs(error_msg)
    atom_types = sorted({atom_type for pair in pairs for atom_type in pair})
    result: Dict[str, Any] = {
        "selected_forcefield": selected_forcefield,
        "pairs": pairs,
        "atom_types": {},
    }
    if not pairs or not ff_root.is_dir():
        return result

    for atom_type in atom_types:
        element = _element_hint(atom_type)
        candidates: List[Dict[str, Any]] = []
        for ff_dir in sorted((path for path in ff_root.iterdir() if path.is_dir()), key=lambda path: path.name):
            mixing = _parse_mixing_entries(ff_dir / "force_field_mixing_rules.def")
            pseudo = _parse_pseudo_entries(ff_dir / "pseudo_atoms.def")
            for source_type, mixing_entry in mixing.items():
                pseudo_entry = pseudo.get(source_type)
                exact = source_type == atom_type
                same_element = bool(
                    element
                    and pseudo_entry
                    and str(pseudo_entry.get("element") or "").lower() == element.lower()
                )
                if not exact and not same_element:
                    continue
                candidates.append(
                    {
                        "source_forcefield": ff_dir.name,
                        "source_type": source_type,
                        "match": "exact_type" if exact else "same_element",
                        "mixing_rule": mixing_entry,
                        "pseudo_atom": pseudo_entry,
                    }
                )
        exact_candidates = [item for item in candidates if item["match"] == "exact_type"]
        if exact_candidates:
            candidates = exact_candidates
        result["atom_types"][atom_type] = candidates
    return result


def format_raspa_local_reference_evidence(
    *,
    error_msg: str,
    selected_forcefield: Optional[str],
    molecule_name: Optional[str],
    max_chars: int = 34000,
) -> str:
    lines = ["Installed RASPA file-backed recovery candidates:"]
    missing_vdw = load_missing_vdw_candidates(error_msg, selected_forcefield)
    if missing_vdw["pairs"]:
        lines.append(
            f"Missing-VDW source candidates for selected Forcefield={selected_forcefield!r}; "
            f"reported pairs={missing_vdw['pairs']}:"
        )
        for target_type, candidates in missing_vdw["atom_types"].items():
            if not candidates:
                lines.append(f"- target_type={target_type}: no installed file-backed candidate")
                continue
            grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
            for candidate in candidates:
                mixing = candidate["mixing_rule"]
                mixing_value = re.sub(r"^\s*\S+\s*", "", mixing["line"]).strip()
                group_key = (candidate["source_type"], mixing_value)
                grouped.setdefault(group_key, []).append(candidate)

            for group in grouped.values():
                candidate = group[0]
                mixing = candidate["mixing_rule"]
                pseudo = candidate.get("pseudo_atom")
                source_forcefields = ",".join(item["source_forcefield"] for item in group)
                pseudo_ref = (
                    f"{pseudo['path']}:{pseudo['line_no']} line={pseudo['line']}"
                    if pseudo
                    else "no matching pseudo_atoms.def entry"
                )
                lines.append(
                    f"- target_type={target_type}; source_forcefields={source_forcefields}; "
                    f"source_type={candidate['source_type']}; match={candidate['match']}; "
                    f"mixing={mixing['path']}:{mixing['line_no']} line={mixing['line']}; "
                    f"pseudo={pseudo_ref}"
                )

    molecule_catalog = load_molecule_definition_catalog(molecule_name)
    lines.append(f"Molecule-definition candidates for MoleculeName={molecule_name!r} (no preference order):")
    if not molecule_catalog:
        lines.append("- none found in the installed RASPA molecule families")
    for item in molecule_catalog:
        comments = " | ".join(item["comments"]) or "no descriptive comments found"
        preview = " / ".join(item["preview"][:6])
        lines.append(
            f"- {item['family']}: path={item['path']}; comments={comments}; file_preview={preview}"
        )

    lines.append("Force fields (no preference order; choose by file availability, scope, comments, and compatibility):")
    for item in load_forcefield_catalog():
        comments = " | ".join(item["file_comments"]) or "no descriptive comments found"
        lines.append(
            f"- {item['name']}: pseudo_atoms={item['has_pseudo_atoms']}"
            f"({item['pseudo_atom_entries']} entries), mixing_rules={item['has_mixing_rules']}"
            f"({item['mixing_rule_entries']} entries), force_field.def={item['has_specific_pairs']}"
            f"({item['specific_pair_entries']} entries); path={item['path']}; comments={comments}"
        )

    text = "\n".join(lines)
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "\n[truncated]"
    return text
