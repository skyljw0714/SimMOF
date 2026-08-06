import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set

from config import RASPA_DIR


KBOLTZ_KCAL_PER_MOL_K = 0.0019872041
DEFAULT_ELEMENTS = ("C", "O", "H", "N", "S", "Zn")


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def _raspa_uff_dir(raspa_dir: Optional[str] = None) -> Path:
    return Path(raspa_dir or RASPA_DIR) / "share" / "raspa" / "forcefield" / "UFF"


def _read_lines(path: Path):
    if not path.exists():
        return []
    return path.read_text(errors="ignore").splitlines()


def load_raspa_uff_reference(raspa_dir: Optional[str] = None) -> Dict[str, Any]:
    uff_dir = _raspa_uff_dir(raspa_dir)
    pseudo_path = uff_dir / "pseudo_atoms.def"
    mixing_path = uff_dir / "force_field_mixing_rules.def"

    pseudo_by_type: Dict[str, Dict[str, Any]] = {}
    for line_no, raw in enumerate(_read_lines(pseudo_path), start=1):
        line = _strip_comment(raw)
        if not line:
            continue
        parts = line.split()
        if len(parts) < 6 or parts[0].lower() in {"number", "type"}:
            continue
        try:
            mass = float(parts[5])
        except ValueError:
            continue
        atom_type = parts[0]
        element = parts[2]
        pseudo_by_type[atom_type] = {
            "atom_type": atom_type,
            "element": element,
            "mass": mass,
            "pseudo_atoms_path": str(pseudo_path),
            "pseudo_atoms_line": line_no,
        }

    lj_by_type: Dict[str, Dict[str, Any]] = {}
    mixing_rule = None
    mixing_rule_line = None
    for line_no, raw in enumerate(_read_lines(mixing_path), start=1):
        line = _strip_comment(raw)
        if not line:
            continue
        parts = line.split()
        if len(parts) == 1 and "-" in parts[0]:
            mixing_rule = parts[0]
            mixing_rule_line = line_no
            continue
        if len(parts) < 4 or parts[1].lower() != "lennard-jones":
            continue
        try:
            epsilon_kelvin = float(parts[2])
            sigma_angstrom = float(parts[3])
        except ValueError:
            continue
        atom_type = parts[0]
        lj_by_type[atom_type] = {
            "atom_type": atom_type,
            "epsilon_kelvin": epsilon_kelvin,
            "epsilon_kcal_per_mol": epsilon_kelvin * KBOLTZ_KCAL_PER_MOL_K,
            "sigma_angstrom": sigma_angstrom,
            "mixing_rules_path": str(mixing_path),
            "mixing_rules_line": line_no,
        }

    by_element: Dict[str, Dict[str, Any]] = {}
    for atom_type, pseudo in pseudo_by_type.items():
        element = str(pseudo.get("element") or "")
        if atom_type != f"{element}_":
            continue
        lj = lj_by_type.get(atom_type)
        if not lj:
            continue
        by_element[element] = {
            **pseudo,
            **lj,
            "source_forcefield": "RASPA_UFF",
            "mixing_rule": mixing_rule or "Lorentz-Berthelot",
            "mixing_rule_line": mixing_rule_line,
        }

    return {
        "by_element": by_element,
        "pseudo_atoms_path": str(pseudo_path),
        "force_field_mixing_rules_path": str(mixing_path),
        "mixing_rule": mixing_rule or "Lorentz-Berthelot",
        "mixing_rule_line": mixing_rule_line,
    }


def infer_elements_from_lammps_inputs(error_msg: str, file_dict: Dict[str, str]) -> Set[str]:
    elements: Set[str] = set()
    text = "\n".join([error_msg or ""] + [content or "" for content in file_dict.values()])

    for match in re.finditer(r"#\s*([A-Z][a-z]?)\b", text):
        elements.add(match.group(1))

    for match in re.finditer(r"\b([A-Z][a-z]?)[A-Za-z0-9_+-]*\b", text):
        token = match.group(1)
        if token in DEFAULT_ELEMENTS:
            elements.add(token)

    masses_text = file_dict.get("system.data", "")
    in_masses = False
    for raw in masses_text.splitlines():
        stripped = raw.strip()
        if re.match(r"^Masses\b", stripped, flags=re.IGNORECASE):
            in_masses = True
            continue
        if in_masses and re.match(r"^[A-Za-z]", stripped):
            break
        if not in_masses:
            continue
        body, _, comment = raw.partition("#")
        parts = body.split()
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        label = comment.strip()
        for match in re.finditer(r"\b([A-Z][a-z]?)\b", label):
            elements.add(match.group(1))

    return elements


def format_lammps_forcefield_reference_evidence(
    error_msg: str = "",
    file_dict: Optional[Dict[str, str]] = None,
    elements: Optional[Iterable[str]] = None,
    max_chars: int = 3500,
) -> str:
    reference = load_raspa_uff_reference()
    by_element = reference.get("by_element", {})

    requested = set(elements or DEFAULT_ELEMENTS)
    if file_dict:
        requested.update(infer_elements_from_lammps_inputs(error_msg, file_dict))
    requested = {element for element in requested if element in by_element}
    if not requested:
        return ""

    ordered = [element for element in DEFAULT_ELEMENTS if element in requested]
    ordered.extend(sorted(requested.difference(ordered)))

    lines = [
        "File-backed force-field reference for LAMMPS repairs:",
        f"- Source force field: RASPA UFF",
        f"- Mass source: {reference['pseudo_atoms_path']}",
        f"- LJ source: {reference['force_field_mixing_rules_path']}",
        "- RASPA UFF epsilon is stored in Kelvin; for LAMMPS `units real`, use epsilon_kcal_per_mol.",
        f"- Cross LJ mixing rule: {reference.get('mixing_rule', 'Lorentz-Berthelot')}",
        "- Available elemental entries:",
    ]
    for element in ordered:
        item = by_element[element]
        lines.append(
            f"  {element}: atom_type={item['atom_type']}; mass={item['mass']:.6g} "
            f"(pseudo_atoms.def:{item['pseudo_atoms_line']}); "
            f"epsilon_K={item['epsilon_kelvin']:.6g}; "
            f"epsilon_kcal_per_mol={item['epsilon_kcal_per_mol']:.6f}; "
            f"sigma_angstrom={item['sigma_angstrom']:.6g} "
            f"(force_field_mixing_rules.def:{item['mixing_rules_line']})"
        )

    text = "\n".join(lines)
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "\n[truncated]"
    return text
