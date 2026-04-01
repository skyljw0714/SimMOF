import os
from typing import Dict, Any, Optional, List, Tuple


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
            or prop in {"dos", "density_of_states", "electronic_density_of_states"}
        )

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
        if isinstance(sys_info, dict) and sys_info.get("dir"):
            sys_info.setdefault("label", context.get("vasp_label") or context.get("mof") or "vasp_job")
            sys_info.setdefault("role", context.get("vasp_role"))
            return sys_info

        vasp_dir = context.get("vasp_dir")
        if not vasp_dir:
            raise RuntimeError("[VASPOutputAgent] missing vasp_system or vasp_dir in context")

        label = context.get("vasp_label") or context.get("mof") or "vasp_job"
        role = context.get("vasp_role")

        return {"dir": vasp_dir, "label": label, "role": role}

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        results: Dict[str, Any] = context.setdefault("results", {})

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
