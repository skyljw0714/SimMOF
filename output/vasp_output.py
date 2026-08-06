import os
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
from ase.io import read

from VASP.adsorption import (
    DEFAULT_DEFORMATION_THRESHOLD_PERCENT,
    analyze_structure_deformation,
)


class VASPOutputAgent:
    def __init__(self) -> None:
        pass

    def _parse_energy_from_outcar(self, outcar_path: str) -> Optional[float]:
        if not os.path.exists(outcar_path):
            return None

        try:
            with open(outcar_path, "r") as f:
                lines = f.readlines()
        except Exception as e:
            print(f"[VASPOutputAgent] ERROR reading OUTCAR: {outcar_path} ({e})")
            return None

        for line in reversed(lines):
            s = line.strip()
            if "free  energy   TOTEN" in s:
                try:
                    return float(s.split("=")[1].split()[0])
                except:
                    pass

        for line in reversed(lines):
            s = line.strip()
            if "energy  without entropy" in s:
                try:
                    return float(s.split("=")[1].split()[0])
                except:
                    pass

        for line in reversed(lines):
            s = line.strip()
            if "next E" in s:
                try:
                    return float(s.split("=")[1].split()[0])
                except:
                    pass

        return None

    

    def _is_dos_job(self, context: Dict[str, Any]) -> bool:
        prop = (context.get("property") or "").lower()
        stage = (context.get("vasp_stage") or "").lower()
        calc = (context.get("vasp_calc_type") or "").lower()
        return (
            stage == "dos"
            or calc == "dos"
            or stage == "projected_dos"
            or calc == "projected_dos"
            or prop in {
                "dos",
                "density_of_states",
                "electronic_density_of_states",
                "projected_dos",
            }
        )

    def _is_projected_dos_job(self, context: Dict[str, Any]) -> bool:
        prop = (context.get("property") or "").lower()
        stage = (context.get("vasp_stage") or "").lower()
        calc = (context.get("vasp_calc_type") or "").lower()
        return "projected_dos" in {prop, stage, calc}

    def _parse_doscar_header(self, doscar_path: str) -> Optional[Dict[str, Any]]:
        if not os.path.exists(doscar_path):
            return None

        try:
            with open(doscar_path, "r") as f:
                
                header = [next(f) for _ in range(6)]
        except Exception as e:
            print(f"[VASPOutputAgent] ERROR reading DOSCAR header: {doscar_path} ({e})")
            return None

        
        try:
            parts = header[5].split()
            
            emax = float(parts[0])
            emin = float(parts[1])
            nedos = int(float(parts[2]))
            efermi = float(parts[3])
            return {
                "e_max_ev": emax,
                "e_min_ev": emin,
                "n_energies": nedos,
                "fermi_ev": efermi,
            }
        except Exception:
            
            return {"raw_header_line6": header[5].strip()}

    def _peek_total_dos_columns(self, first_data_line: str) -> int:
        return len(first_data_line.split())

    def _parse_total_dos_preview(
        self,
        doscar_path: str,
        nedos: int,
        max_points: int = 200,
    ) -> Optional[Dict[str, Any]]:
        if not os.path.exists(doscar_path):
            return None

        try:
            with open(doscar_path, "r") as f:
                
                for _ in range(6):
                    next(f)

                
                first = next(f).strip()
                if not first:
                    return None
                ncol = self._peek_total_dos_columns(first)

                
                data_lines = [first]
                
                for _ in range(min(nedos - 1, max_points - 1)):
                    data_lines.append(next(f).strip())

        except StopIteration:
            
            pass
        except Exception as e:
            print(f"[VASPOutputAgent] ERROR parsing DOSCAR: {doscar_path} ({e})")
            return None

        
        parsed: List[List[float]] = []
        for ln in data_lines:
            if not ln:
                continue
            try:
                parsed.append([float(x) for x in ln.split()])
            except:
                continue

        return {
            "n_columns": ncol,
            "n_points_preview": len(parsed),
            "preview": parsed,  
        }

    @staticmethod
    def _incar_uses_noncollinear_spin(incar_path: str) -> bool:
        if not os.path.exists(incar_path):
            return False
        try:
            with open(incar_path, "r", errors="ignore") as handle:
                text = handle.read()
        except OSError:
            return False
        for line in text.splitlines():
            content = line.split("!", 1)[0].split("#", 1)[0].strip()
            if not content or "=" not in content:
                continue
            key, value = content.split("=", 1)
            if key.strip().upper() == "LNONCOLLINEAR":
                return value.strip().upper() in {".TRUE.", "TRUE", "T", "1"}
        return False

    @staticmethod
    def _orbital_group_slices(n_orbitals: int) -> Dict[str, Tuple[int, int]]:
        if n_orbitals >= 16:
            return {"s": (0, 1), "p": (1, 4), "d": (4, 9), "f": (9, 16)}
        if n_orbitals >= 9:
            return {"s": (0, 1), "p": (1, 4), "d": (4, 9)}
        if n_orbitals == 4:
            return {"s": (0, 1), "p": (1, 2), "d": (2, 3), "f": (3, 4)}
        if n_orbitals == 3:
            return {"s": (0, 1), "p": (1, 2), "d": (2, 3)}
        if n_orbitals == 1:
            return {"s": (0, 1)}
        raise ValueError(
            f"Unsupported number of projected orbital channels: {n_orbitals}"
        )

    def _parse_projected_doscar(
        self,
        doscar_path: str,
        structure_path: str,
        output_path: str,
        incar_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not os.path.exists(doscar_path):
            return {
                "status": "missing_doscar",
                "doscar": doscar_path,
            }
        if not os.path.exists(structure_path):
            return {
                "status": "missing_structure",
                "structure": structure_path,
                "doscar": doscar_path,
            }

        try:
            symbols = read(structure_path).get_chemical_symbols()
        except Exception as exc:
            return {
                "status": "structure_parse_failed",
                "structure": structure_path,
                "error": str(exc),
            }

        try:
            with open(doscar_path, "r", errors="ignore") as handle:
                first_header = next(handle).split()
                atom_count = int(float(first_header[0]))
                for _ in range(4):
                    next(handle)
                total_header = next(handle).split()
                e_max = float(total_header[0])
                e_min = float(total_header[1])
                n_energies = int(float(total_header[2]))
                efermi = float(total_header[3])

                total_rows = [
                    [float(value) for value in next(handle).split()]
                    for _ in range(n_energies)
                ]
                if not total_rows:
                    raise ValueError("DOSCAR contains no total DOS rows")

                if self._incar_uses_noncollinear_spin(incar_path or ""):
                    component_names = ["charge", "mx", "my", "mz"]
                elif len(total_rows[0]) == 5:
                    component_names = ["up", "down"]
                else:
                    component_names = ["total"]
                n_components = len(component_names)

                atom_blocks: List[np.ndarray] = []
                energies = None
                n_orbitals = None
                for atom_index in range(atom_count):
                    projected_header = next(handle).split()
                    if len(projected_header) < 3:
                        raise ValueError(
                            f"Malformed projected DOS header for atom {atom_index + 1}"
                        )
                    block = np.asarray(
                        [
                            [float(value) for value in next(handle).split()]
                            for _ in range(n_energies)
                        ],
                        dtype=float,
                    )
                    if block.ndim != 2 or block.shape[1] < 2:
                        raise ValueError(
                            f"Malformed projected DOS block for atom {atom_index + 1}"
                        )
                    if energies is None:
                        energies = block[:, 0]
                    elif not np.allclose(energies, block[:, 0], atol=1e-6):
                        raise ValueError(
                            f"Inconsistent energy grid in atom block {atom_index + 1}"
                        )

                    projected_columns = block.shape[1] - 1
                    if projected_columns % n_components:
                        raise ValueError(
                            "Projected DOS columns are incompatible with the "
                            f"{n_components} spin/magnetization components"
                        )
                    current_n_orbitals = projected_columns // n_components
                    self._orbital_group_slices(current_n_orbitals)
                    if n_orbitals is None:
                        n_orbitals = current_n_orbitals
                    elif n_orbitals != current_n_orbitals:
                        raise ValueError("Projected DOS atom blocks have different widths")

                    atom_blocks.append(
                        block[:, 1:].reshape(
                            n_energies,
                            current_n_orbitals,
                            n_components,
                        )
                    )
        except StopIteration:
            return {
                "status": "parse_failed",
                "doscar": doscar_path,
                "error": (
                    "DOSCAR ended before all atom-projected blocks were read; "
                    "the calculation likely did not enable LORBIT."
                ),
            }
        except (OSError, ValueError) as exc:
            return {
                "status": "parse_failed",
                "doscar": doscar_path,
                "error": str(exc),
            }

        if atom_count != len(symbols):
            return {
                "status": "atom_count_mismatch",
                "doscar_atoms": atom_count,
                "structure_atoms": len(symbols),
                "doscar": doscar_path,
                "structure": structure_path,
            }

        raw = np.asarray(atom_blocks, dtype=np.float32)
        grouped = np.zeros(
            (atom_count, n_energies, 4, n_components),
            dtype=np.float32,
        )
        group_names = ["s", "p", "d", "f"]
        group_slices = self._orbital_group_slices(int(n_orbitals or 0))
        for group_index, group_name in enumerate(group_names):
            if group_name not in group_slices:
                continue
            start, stop = group_slices[group_name]
            grouped[:, :, group_index, :] = raw[:, :, start:stop, :].sum(axis=2)

        relative_energies = np.asarray(energies, dtype=np.float64) - efermi
        np.savez_compressed(
            output_path,
            energies_ev=relative_energies,
            densities=grouped,
            atom_symbols=np.asarray(symbols, dtype="<U3"),
            orbital_groups=np.asarray(group_names, dtype="<U1"),
            component_names=np.asarray(component_names, dtype="<U8"),
            efermi_ev=np.asarray(efermi),
        )
        return {
            "status": "ok",
            "doscar": doscar_path,
            "structure": structure_path,
            "artifact": output_path,
            "n_atoms": atom_count,
            "n_energies": n_energies,
            "orbital_groups": group_names,
            "components": component_names,
            "spin_polarized": component_names == ["up", "down"],
            "fermi_ev": efermi,
            "energy_reference": "E - E_F",
            "energy_range_ev": [
                float(relative_energies.min()),
                float(relative_energies.max()),
            ],
            "doscar_energy_range_ev": [e_min, e_max],
        }

    def _is_bandgap_job(self, context: Dict[str, Any]) -> bool:
        prop = (context.get("property") or "").lower()
        stage = (context.get("vasp_stage") or "").lower()
        calc = (context.get("vasp_calc_type") or "").lower()
        return (
            prop in ["band_gap", "bandgap", "electronic_band_gap"]
            or stage in ["band_gap", "bandgap"]
            or calc in ["band_gap", "bandgap"]
        )

    def _parse_bandgap_from_eigenval(self, eigenval_path: str) -> Optional[Dict[str, Any]]:
        if not os.path.exists(eigenval_path):
            return None

        try:
            with open(eigenval_path, "r", errors="ignore") as f:
                raw_lines = [ln.rstrip("\n") for ln in f]
        except Exception:
            return None

        lines = [ln.strip() for ln in raw_lines]

        def _is_float4(s: str) -> bool:
            p = s.split()
            if len(p) < 4:
                return False
            try:
                float(p[0]); float(p[1]); float(p[2]); float(p[3])
                return True
            except Exception:
                return False

        def _band_line_parts(s: str):
            p = s.split()
            if len(p) < 3:
                return None
            try:
                bi = int(float(p[0]))
                en = float(p[1])
                occs = [float(x) for x in p[2:]]
                return bi, en, occs
            except Exception:
                return None

        def _next_nonempty(i: int) -> int:
            while i < len(lines) and not lines[i]:
                i += 1
            return i

        
        kp0 = None
        for i in range(len(lines)):
            if not lines[i]:
                continue
            if _is_float4(lines[i]):
                j = _next_nonempty(i + 1)
                bl = _band_line_parts(lines[j]) if j < len(lines) else None
                if bl and bl[0] == 1:
                    kp0 = i
                    break
        if kp0 is None:
            return None

        
        i = _next_nonempty(kp0 + 1)
        max_bi = 0
        while i < len(lines):
            if not lines[i]:
                i += 1
                continue
            if _is_float4(lines[i]):
                j = _next_nonempty(i + 1)
                bl = _band_line_parts(lines[j]) if j < len(lines) else None
                if bl and bl[0] == 1:
                    break
            bl = _band_line_parts(lines[i])
            if bl:
                bi, _, _ = bl
                if bi > max_bi:
                    max_bi = bi
            i += 1
        nbands = max_bi
        if nbands <= 0:
            return None

        
        nkpt = None
        nelect = None
        for h in range(max(0, kp0 - 50), kp0):
            p = lines[h].split()
            if len(p) >= 3:
                try:
                    a = int(float(p[0])); b = int(float(p[1])); c = int(float(p[2]))
                except Exception:
                    continue
                
                if c == nbands and b >= 1:
                    nkpt = b
                    nelect = float(a)
                    break

        
        occ_tol = 1e-4
        vbm = None
        cbm = None
        blocks_seen = 0

        i = kp0
        while i < len(lines):
            if not lines[i]:
                i += 1
                continue

            if _is_float4(lines[i]):
                j = _next_nonempty(i + 1)
                bl0 = _band_line_parts(lines[j]) if j < len(lines) else None
                if not bl0 or bl0[0] != 1:
                    i += 1
                    continue

                blocks_seen += 1
                i = j

                read_bands = 0
                while i < len(lines) and read_bands < nbands:
                    if not lines[i]:
                        i += 1
                        continue
                    bl = _band_line_parts(lines[i])
                    i += 1
                    if bl is None:
                        continue
                    _, e, occs = bl
                    read_bands += 1

                    occ = occs[0] if occs else 0.0
                    if occ >= 1.0 - occ_tol:
                        if vbm is None or e > vbm:
                            vbm = e
                    elif occ <= occ_tol:
                        if cbm is None or e < cbm:
                            cbm = e

                if nkpt is not None and blocks_seen >= nkpt:
                    break
                continue

            i += 1

        if vbm is None or cbm is None:
            return None

        gap = cbm - vbm
        return {
            "status": "ok",
            "source": eigenval_path,
            "nelect": nelect,
            "nkpt": nkpt if nkpt is not None else blocks_seen,
            "nbands": nbands,
            "vbm_ev": vbm,
            "cbm_ev": cbm,
            "gap_ev": gap,
        }



    

    def _get_single_system_info(self, context: Dict[str, Any]) -> Dict[str, Any]:
        sys_info = context.get("vasp_system")
        if not (isinstance(sys_info, dict) and sys_info.get("dir")):
            vasp_dir = context.get("vasp_dir")
            if not vasp_dir:
                raise RuntimeError("[VASPOutputAgent] missing vasp_system or vasp_dir in context")

            sys_info = {
                "dir": vasp_dir,
                "label": context.get("vasp_label") or context.get("mof") or "vasp_job",
                "role": context.get("vasp_role"),
            }

        sys_info.setdefault("label", context.get("vasp_label") or context.get("mof") or "vasp_job")
        sys_info.setdefault("role", context.get("vasp_role"))

        context["vasp_system"] = sys_info
        context["vasp_dir"] = sys_info["dir"]
        context["vasp_label"] = sys_info["label"]
        if sys_info.get("role"):
            context["vasp_role"] = sys_info["role"]

        return sys_info

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        results: Dict[str, Any] = context.setdefault("results", {})

        if context.get("vasp_status") == "needs_structure_from_user":
            results["vasp_output_status"] = "blocked_missing_structure"
            return context

        try:
            sys_info = self._get_single_system_info(context)
        except Exception as e:
            print(f"[VASPOutputAgent] ERROR: {e}")
            results["vasp_output_status"] = "failed_no_system"
            return context

        system_dir = sys_info["dir"]
        label = sys_info.get("label")
        role = sys_info.get("role")

        
        outcar_path = os.path.join(system_dir, "OUTCAR")
        if not os.path.exists(outcar_path):
            print(f"[VASPOutputAgent] OUTCAR not found: {outcar_path}")
            results["vasp_output_status"] = "missing_outcar"
            results["vasp_energy_ev"] = None
            results["vasp_label"] = label
            results["vasp_role"] = role
            results["vasp_outcar"] = outcar_path
            e = None
        else:
            e = self._parse_energy_from_outcar(outcar_path)
            if e is None:
                print(f"[VASPOutputAgent] Failed to parse energy: {label} ({outcar_path})")
                results["vasp_output_status"] = "parse_failed"
                results["vasp_energy_ev"] = None
            else:
                print(f"[VASPOutputAgent] {label}: E = {e:.6f} eV")
                results["vasp_output_status"] = "ok"
                results["vasp_energy_ev"] = e

        context["vasp_energy"] = {
            "label": label,
            "role": role,
            "dir": system_dir,
            "outcar": outcar_path,
            "energy_ev": e,
            "status": results["vasp_output_status"],
        }

        results["vasp_label"] = label
        results["vasp_role"] = role
        results["vasp_outcar"] = outcar_path

        threshold = context.get(
            "structure_deformation_threshold_percent",
            DEFAULT_DEFORMATION_THRESHOLD_PERCENT,
        )
        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            threshold = DEFAULT_DEFORMATION_THRESHOLD_PERCENT
        deformation = analyze_structure_deformation(
            os.path.join(system_dir, "POSCAR"),
            os.path.join(system_dir, "CONTCAR"),
            threshold_percent=threshold,
        )
        deformation["role"] = role
        deformation["label"] = label
        results["structure_deformation"] = deformation
        context["structure_deformation"] = deformation
        if deformation.get("threshold_exceeded"):
            print(
                "[VASPOutputAgent] WARNING: structural deformation "
                f"{deformation.get('overall_deformation_percent', 0.0):.2f}% "
                f">= {threshold:.2f}% for {label}"
            )

        
        if self._is_dos_job(context):
            doscar_path = os.path.join(system_dir, "DOSCAR")
            if not os.path.exists(doscar_path):
                
                results["dos"] = {
                    "status": "missing_doscar",
                    "doscar": doscar_path,
                    "vasp_dir": system_dir,
                }
            else:
                header = self._parse_doscar_header(doscar_path) or {}
                
                nedos = header.get("n_energies")
                preview = None
                if isinstance(nedos, int) and nedos > 0:
                    preview = self._parse_total_dos_preview(doscar_path, nedos=nedos, max_points=200)

                results["dos"] = {
                    "status": "ok",
                    "doscar": doscar_path,
                    "vasp_dir": system_dir,
                    **header,
                }
                if preview:
                    results["dos"]["total_dos_preview"] = preview

        if self._is_projected_dos_job(context):
            structure_path = os.path.join(system_dir, "POSCAR")
            artifact_path = os.path.join(system_dir, "projected_dos.npz")
            projected = self._parse_projected_doscar(
                os.path.join(system_dir, "DOSCAR"),
                structure_path,
                artifact_path,
                incar_path=os.path.join(system_dir, "INCAR"),
            )
            projected.update(
                {
                    "role": context.get("projected_dos_role") or role,
                    "vasp_dir": system_dir,
                }
            )
            results["projected_dos"] = projected

        
        if self._is_bandgap_job(context):
            eigenval_path = os.path.join(system_dir, "EIGENVAL")
            bg = self._parse_bandgap_from_eigenval(eigenval_path)

            if bg is None:
                results["band_gap"] = {
                    "status": "parse_failed",
                    "eigenval": eigenval_path,
                    "vasp_dir": system_dir,
                }
            else:
                results["band_gap"] = bg

        
        if results.get("vasp_run_status") is None:
            results["vasp_run_status"] = "done"

        return context
