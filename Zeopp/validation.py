from __future__ import annotations

import math
import re
import shlex
import stat
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import ase.io

from config import working_dir
from output.zeopp_output import ZeoppOutputAgent


NUMERIC_ARGUMENTS: Dict[str, Tuple[str, ...]] = {
    "-sa": ("float", "float", "int"),
    "-vol": ("float", "float", "int"),
    "-volpo": ("float", "float", "int"),
    "-psd": ("float", "float", "int"),
    "-ray": ("float", "float", "int"),
    "-chan": ("float",),
    "-block": ("float",),
    "-axs": ("float",),
}


def _issue(code: str, message: str, **details: Any) -> Dict[str, Any]:
    item: Dict[str, Any] = {"code": code, "message": message}
    if details:
        item["details"] = details
    return item


def _report(
    stage: str,
    issues: List[Dict[str, Any]],
    *,
    metadata: Optional[Dict[str, Any]] = None,
    observations: Optional[List[Dict[str, Any]]] = None,
    ok: Optional[bool] = None,
) -> Dict[str, Any]:
    evidence = [
        f"[SimMOF validation:{item['code']}] {item['message']}"
        for item in issues
    ]
    return {
        "ok": not issues if ok is None else ok,
        "stage": stage,
        "issues": issues,
        "evidence": evidence,
        "observations": observations or [],
        "metadata": metadata or {},
    }


def _command_tokens(command: str) -> Tuple[List[str], Optional[str]]:
    try:
        return shlex.split(command or ""), None
    except ValueError as exc:
        return [], str(exc)


def _resolve_cif_path(context: Dict[str, Any], tokens: Sequence[str]) -> Optional[Path]:
    work_dir = Path(context.get("work_dir", working_dir))
    cif_tokens = [token for token in tokens if token.lower().endswith(".cif")]
    if cif_tokens:
        path = Path(cif_tokens[-1])
        return path if path.is_absolute() else work_dir / path

    mof_path = context.get("mof_path")
    if mof_path:
        return Path(mof_path)

    mof = context.get("mof") or (context.get("zeopp_info") or {}).get("MOF")
    return work_dir / f"{mof}.cif" if mof else None


def _validate_command(command: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    issues: List[Dict[str, Any]] = []
    tokens, parse_error = _command_tokens(command)
    if parse_error:
        issues.append(
            _issue(
                "command_tokenization_failed",
                f"Could not tokenize the Zeo++ command: {parse_error}",
            )
        )
        return issues, tokens

    for option, expected_types in NUMERIC_ARGUMENTS.items():
        if option not in tokens:
            continue
        start = tokens.index(option) + 1
        values = tokens[start : start + len(expected_types)]
        if len(values) != len(expected_types):
            issues.append(
                _issue(
                    "command_argument_count_invalid",
                    (
                        f"{option} requires {len(expected_types)} numeric arguments, "
                        f"but only {len(values)} were present."
                    ),
                    option=option,
                    values=values,
                )
            )
            continue

        for offset, (value, expected_type) in enumerate(zip(values, expected_types), start=1):
            try:
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError("value is not finite")
                if expected_type == "int" and not number.is_integer():
                    raise ValueError("value is not an integer")
            except (TypeError, ValueError) as exc:
                issues.append(
                    _issue(
                        "command_numeric_argument_invalid",
                        (
                            f"{option} argument {offset} must be a finite "
                            f"{expected_type}, but received {value!r}."
                        ),
                        option=option,
                        argument_index=offset,
                        value=value,
                        error=str(exc),
                    )
                )

        sample_value = values[-1] if expected_types and expected_types[-1] == "int" else None
        if sample_value is not None:
            try:
                if int(float(sample_value)) <= 0:
                    issues.append(
                        _issue(
                            "command_sample_count_invalid",
                            f"{option} sample count must be greater than zero.",
                            option=option,
                            value=sample_value,
                        )
                    )
            except (TypeError, ValueError):
                pass
        break

    return issues, tokens


def _cif_tokens(line: str) -> List[str]:
    try:
        return shlex.split(line, comments=True, posix=True)
    except ValueError:
        return line.split()


def _inspect_atom_loop(text: str) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    lines = text.splitlines()
    issues: List[Dict[str, Any]] = []
    atom_row_count: Optional[int] = None
    index = 0

    while index < len(lines):
        if lines[index].strip().lower() != "loop_":
            index += 1
            continue

        index += 1
        tags: List[str] = []
        while index < len(lines):
            stripped = lines[index].strip()
            if not stripped or stripped.startswith("#"):
                index += 1
                continue
            if not stripped.startswith("_"):
                break
            tags.append(stripped.split()[0])
            index += 1

        if not any(tag.lower().startswith("_atom_site_") for tag in tags):
            continue

        values: List[str] = []
        while index < len(lines):
            stripped = lines[index].strip()
            lowered = stripped.lower()
            if (
                lowered == "loop_"
                or lowered.startswith("data_")
                or lowered.startswith("save_")
                or stripped.startswith("_")
            ):
                break
            if stripped and not stripped.startswith("#"):
                values.extend(_cif_tokens(stripped))
            index += 1

        if not tags:
            issues.append(
                _issue(
                    "cif_atom_loop_has_no_columns",
                    "The CIF atom loop does not declare any _atom_site_ columns.",
                )
            )
            return issues, atom_row_count

        remainder = len(values) % len(tags)
        if remainder:
            issues.append(
                _issue(
                    "cif_loop_value_count_mismatch",
                    (
                        "The CIF atom loop value count is not divisible by its "
                        "column count; at least one atom row is incomplete."
                    ),
                    column_count=len(tags),
                    value_count=len(values),
                    remainder=remainder,
                )
            )
        else:
            atom_row_count = len(values) // len(tags)

        lowered_tags = {tag.lower() for tag in tags}
        if not {
            "_atom_site_label",
            "_atom_site_type_symbol",
        }.intersection(lowered_tags):
            issues.append(
                _issue(
                    "cif_atom_identity_missing",
                    "The CIF atom loop has neither atom labels nor element-type symbols.",
                )
            )

        fractional = {
            "_atom_site_fract_x",
            "_atom_site_fract_y",
            "_atom_site_fract_z",
        }
        cartesian = {
            "_atom_site_cartn_x",
            "_atom_site_cartn_y",
            "_atom_site_cartn_z",
        }
        if not (fractional.issubset(lowered_tags) or cartesian.issubset(lowered_tags)):
            issues.append(
                _issue(
                    "cif_atom_coordinates_missing",
                    "The CIF atom loop does not contain a complete coordinate triplet.",
                )
            )
        return issues, atom_row_count

    issues.append(
        _issue(
            "cif_atom_loop_missing",
            "No _atom_site_ loop was found in the CIF.",
        )
    )
    return issues, atom_row_count


def _validate_cif(
    cif_path: Optional[Path],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]], bool]:
    issues: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
    observations: List[Dict[str, Any]] = []
    if cif_path is None:
        observations.append(
            {
                "source": "input",
                "operation": "resolve_cif_path",
                "result": {
                    "succeeded": False,
                    "resolved_path": None,
                },
            }
        )
        return issues, metadata, observations, False

    metadata["cif_path"] = str(cif_path)
    try:
        stat_result = cif_path.stat()
        stat_observation = {
            "source": "filesystem",
            "operation": "stat",
            "path": str(cif_path),
            "result": {
                "succeeded": True,
                "exists": True,
                "is_file": stat.S_ISREG(stat_result.st_mode),
            },
        }
    except OSError as exc:
        stat_observation = {
            "source": "filesystem",
            "operation": "stat",
            "path": str(cif_path),
            "result": {
                "succeeded": False,
                "exists": False if isinstance(exc, FileNotFoundError) else None,
                "is_file": False if isinstance(exc, FileNotFoundError) else None,
                "error_type": type(exc).__name__,
                "errno": exc.errno,
                "error": str(exc),
            },
        }
    observations.append(stat_observation)
    if (
        not stat_observation["result"]["succeeded"]
        or not stat_observation["result"]["is_file"]
    ):
        return issues, metadata, observations, False

    try:
        text = cif_path.read_text(encoding="utf-8", errors="replace")
        observations.append(
            {
                "source": "filesystem",
                "operation": "read_text",
                "path": str(cif_path),
                "result": {
                    "succeeded": True,
                    "character_count": len(text),
                },
            }
        )
    except OSError as exc:
        observations.append(
            {
                "source": "filesystem",
                "operation": "read_text",
                "path": str(cif_path),
                "result": {
                    "succeeded": False,
                    "error_type": type(exc).__name__,
                    "errno": exc.errno,
                    "error": str(exc),
                },
            }
        )
        return issues, metadata, observations, False

    loop_issues, atom_row_count = _inspect_atom_loop(text)
    issues.extend(loop_issues)
    if atom_row_count is not None:
        metadata["cif_atom_row_count"] = atom_row_count

    try:
        atoms = ase.io.read(str(cif_path))
        atom_count = len(atoms)
        metadata["expected_atom_count"] = atom_count
        if atom_count < 1:
            issues.append(
                _issue(
                    "cif_has_no_atoms",
                    "The CIF parser returned no atoms.",
                )
            )

        lengths = [float(value) for value in atoms.cell.lengths()]
        angles = [float(value) for value in atoms.cell.angles()]
        volume = float(atoms.cell.volume)
        metadata["cell_lengths"] = lengths
        metadata["cell_angles"] = angles
        metadata["cell_volume"] = volume

        if any(not math.isfinite(value) or value <= 0.0 for value in lengths):
            issues.append(
                _issue(
                    "cif_cell_length_invalid",
                    "The CIF has a non-finite or non-positive unit-cell length.",
                    cell_lengths=lengths,
                )
            )
        if any(not math.isfinite(value) or not 0.0 < value < 180.0 for value in angles):
            issues.append(
                _issue(
                    "cif_cell_angle_invalid",
                    "The CIF has a non-finite or invalid unit-cell angle.",
                    cell_angles=angles,
                )
            )
        if not math.isfinite(volume) or volume <= 0.0:
            issues.append(
                _issue(
                    "cif_cell_volume_invalid",
                    "The CIF unit-cell volume is non-finite or non-positive.",
                    cell_volume=volume,
                )
            )

        positions = atoms.get_positions()
        if any(not math.isfinite(float(value)) for row in positions for value in row):
            issues.append(
                _issue(
                    "cif_coordinates_nonfinite",
                    "The CIF contains non-finite atom coordinates.",
                )
            )
    except Exception as exc:
        issues.append(
            _issue(
                "cif_parse_failed",
                f"ASE could not parse the CIF: {exc}",
                path=str(cif_path),
            )
        )

    return issues, metadata, observations, True


def validate_zeopp_preflight(context: Dict[str, Any]) -> Dict[str, Any]:
    command = context.get("zeopp_command", "")
    command_issues, tokens = _validate_command(command)
    cif_path = _resolve_cif_path(context, tokens)
    cif_issues, cif_metadata, observations, cif_ready = _validate_cif(cif_path)
    issues = command_issues + cif_issues
    return _report(
        "pre_run",
        issues,
        metadata={"command": command, **cif_metadata},
        observations=observations,
        ok=not issues and cif_ready,
    )


def _iter_numbers(value: Any, path: str = "result") -> Iterable[Tuple[str, float]]:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield path, float(value)
        return
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _iter_numbers(child, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _iter_numbers(child, f"{path}[{index}]")


def _parse_expected_output(context: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
    work_dir = str(context.get("work_dir", working_dir))
    info = context.get("zeopp_info") or {}
    mof = info.get("MOF") or context.get("mof")
    command = info.get("command") or context.get("zeopp_command", "")
    if not mof:
        raise ValueError("MOF name is missing from zeopp_info and context")

    output = ZeoppOutputAgent()
    if "-psd" in command:
        return "pore_size_distribution", output._read_psd_file(mof, work_dir)
    if "-res" in command:
        return "pore_diameter", output._read_res_file(mof, work_dir)
    if "-vol" in command:
        return "accessible_volume", output._read_vol_file(mof, work_dir)
    if "-sa" in command:
        return "surface_area", output._read_sa_file(mof, work_dir)
    return None, {}


def validate_zeopp_postflight(context: Dict[str, Any]) -> Dict[str, Any]:
    results = context.setdefault("results", {})
    issues: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}

    preflight = results.get("zeopp_preflight_validation") or {}
    expected_atom_count = (preflight.get("metadata") or {}).get("expected_atom_count")
    if expected_atom_count is not None:
        metadata["expected_atom_count"] = expected_atom_count

    stdout = str(results.get("zeopp_stdout", ""))
    particle_counts = [
        int(value)
        for value in re.findall(r"Total particles\s*=\s*(\d+)", stdout)
    ]
    if particle_counts:
        metadata["zeopp_particle_counts"] = particle_counts
    if expected_atom_count is not None and any(
        count != expected_atom_count for count in particle_counts
    ):
        issues.append(
            _issue(
                "zeopp_particle_count_mismatch",
                (
                    f"The validated CIF contains {expected_atom_count} atoms, but "
                    f"Zeo++ reported particle count(s) {particle_counts}."
                ),
                expected_atom_count=expected_atom_count,
                zeopp_particle_counts=particle_counts,
            )
        )

    try:
        property_type, parsed = _parse_expected_output(context)
        metadata["property_type"] = property_type
        if property_type and not parsed:
            issues.append(
                _issue(
                    "zeopp_output_empty",
                    f"The {property_type} output was present but contained no parsed values.",
                )
            )
        for value_path, number in _iter_numbers(parsed):
            if not math.isfinite(number):
                issues.append(
                    _issue(
                        "zeopp_output_nonfinite",
                        f"The Zeo++ output contains a non-finite value at {value_path}.",
                        value_path=value_path,
                        value=number,
                    )
                )

        if property_type == "surface_area":
            for key, value in parsed.items():
                if float(value) < 0.0:
                    issues.append(
                        _issue(
                            "zeopp_surface_area_negative",
                            f"Surface-area value {key} is negative.",
                            key=key,
                            value=value,
                        )
                    )
        elif property_type == "accessible_volume":
            fraction = parsed.get("AV_Volume_fraction")
            if fraction is not None and not 0.0 <= float(fraction) <= 1.0:
                issues.append(
                    _issue(
                        "zeopp_volume_fraction_out_of_range",
                        "Accessible-volume fraction is outside the range [0, 1].",
                        value=fraction,
                    )
                )
        elif property_type == "pore_diameter":
            for key, value in parsed.items():
                if float(value) < 0.0:
                    issues.append(
                        _issue(
                            "zeopp_pore_diameter_negative",
                            f"Pore-diameter value {key} is negative.",
                            key=key,
                            value=value,
                        )
                    )
    except (OSError, ValueError, IndexError, RuntimeError) as exc:
        issues.append(
            _issue(
                "zeopp_output_parse_failed",
                f"The expected Zeo++ output could not be parsed: {exc}",
            )
        )

    return _report("post_run", issues, metadata=metadata)


def store_validation_report(
    context: Dict[str, Any],
    report: Dict[str, Any],
    *,
    key: str,
) -> Dict[str, Any]:
    results = context.setdefault("results", {})
    results[key] = report
    results["zeopp_validation_stage"] = report.get("stage")
    results["zeopp_validation_issues"] = list(report.get("issues", []))
    results["zeopp_validation_errors"] = [
        item.get("message", "") for item in report.get("issues", [])
    ]
    results["zeopp_validation_evidence"] = list(report.get("evidence", []))
    results["zeopp_validation_observations"] = list(
        report.get("observations", [])
    )
    if not report.get("ok"):
        results["zeopp_status"] = "validation_failed"
        results["zeopp_error_kind"] = "semantic_validation"
    return context
