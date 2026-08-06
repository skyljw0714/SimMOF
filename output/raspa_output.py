import re
import json
import csv

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from config import LLM_DEFAULT, AGENT_LLM_MAP


class RASPAOutputAgent:
    FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[Ee][+-]?\d+)?")
    NUM_TOKEN_RE = re.compile(r"[-+]?(?:\d*\.?\d+(?:[Ee][+-]?\d+)?|nan|inf)", re.IGNORECASE)
    TARGET_UNIT_SUBSTR = "cm^3 (STP)/cm^3 framework"
    HENRY_RE = re.compile(
        r"\s*\[[^\]]+\]\s*Average Henry coefficient:\s*([0-9Ee\+\-\.]+)\s*(?:\+/-|±)\s*([0-9Ee\+\-\.]+)\s*\[([^\]]+)\]"
    )
    ENTHALPY_ADS_RE = re.compile(
        r"^\s*([-+0-9Ee\.]+)\s*\+/-\s*([-+0-9Ee\.]+)\s*\[KJ/MOL\]\s*$",
        re.IGNORECASE
    )

    COMPONENT_HEADER_RE = re.compile(r"^\s*Component\s+(\d+)\s+\[([^\]]+)\](?:.*)$", re.IGNORECASE)
    LOADING_LINE_RE = re.compile(
        r"^\s*Average loading (absolute|excess)\s*\[([^\]]+)\]\s*([-+0-9Ee\.]+)\s*\+/-\s*([-+0-9Ee\.]+)",
        re.IGNORECASE
    )
    AVG_ENERGY_RE = re.compile(
        r"^\s*Average\s+([-+0-9Ee\.]+)\s+Van der Waals:\s*([-+0-9Ee\.]+)\s+Coulomb:\s*([-+0-9Ee\.]+)\s*\[K\]",
        re.IGNORECASE
    )
    ERR_ENERGY_RE = re.compile(
        r"^\s*\+/-\s*([-+0-9Ee\.]+)\s+\+/-\s*([-+0-9Ee\.]+)\s+\+/-\s*([-+0-9Ee\.]+)\s*\[K\]",
        re.IGNORECASE
    )



    def __init__(self, llm=None):
        self.llm = llm or AGENT_LLM_MAP.get("RASPAOutputAgent", LLM_DEFAULT)

    def _to_float(self, value: str):
        try:
            parsed = float(value)
        except Exception:
            return None
        if parsed != parsed or parsed in (float("inf"), float("-inf")):
            return None
        return parsed

    def _extract_first_number(self, line: str):
        m = self.NUM_TOKEN_RE.search(line)
        if not m:
            return None
        return self._to_float(m.group(0))

    def _parse_value_error_unit_line(self, line: str):
        nums = self.NUM_TOKEN_RE.findall(line)
        if len(nums) < 2:
            return None, None, None
        unit_m = re.search(r"\[([^\]]+)\]\s*$", line)
        return self._to_float(nums[0]), self._to_float(nums[1]), unit_m.group(1).strip() if unit_m else None

    def _parse_uptake_from_data(self, text: str):
        for line in text.splitlines():
            if "Average loading excess" not in line:
                continue
            if self.TARGET_UNIT_SUBSTR not in line:
                continue

            units_match = re.search(r"Average loading excess\s*\[([^\]]+)\]", line)
            units = units_match.group(1).strip() if units_match else None

            pos = line.find("]")
            num_region = line[pos + 1:] if pos != -1 else line
            num_match = self.FLOAT_RE.search(num_region)
            if num_match:
                try:
                    val = float(num_match.group(0))
                except ValueError:
                    val = None
                return val, units

        return None, None

    def _parse_loading_table(self, text: str):
        loadings: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
        cur_name = None

        for line in text.splitlines():
            m = self.COMPONENT_HEADER_RE.match(line)
            if m:
                cur_name = m.group(2).strip()
                loadings.setdefault(cur_name, {"absolute": {}, "excess": {}})
                continue

            if cur_name is None:
                continue

            m2 = self.LOADING_LINE_RE.match(line)
            if not m2:
                continue

            mode = m2.group(1).lower().strip()
            unit = m2.group(2).strip()
            val = self._to_float(m2.group(3))
            err = self._to_float(m2.group(4))
            if val is None:
                continue
            loadings.setdefault(cur_name, {"absolute": {}, "excess": {}})
            loadings[cur_name][mode][unit] = {"value": val, "error": err}

        return loadings

    def _parse_conditions_and_components(self, text: str):
        conditions: Dict[str, Any] = {}
        components: Dict[str, Dict[str, Any]] = {}
        lines = text.splitlines()
        cur_name = None
        pending_pressure = None
        pending_fugacity = None

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("External temperature:"):
                conditions["temperature_K"] = self._extract_first_number(stripped)
            elif stripped.startswith("External Pressure:"):
                p_pa = self._extract_first_number(stripped)
                conditions["external_pressure_Pa"] = p_pa
                if p_pa is not None:
                    conditions["external_pressure_bar"] = p_pa / 100000.0

            m = self.COMPONENT_HEADER_RE.match(line)
            if m:
                cur_name = m.group(2).strip()
                components.setdefault(cur_name, {})
                pending_pressure = None
                pending_fugacity = None
                continue

            if cur_name is None:
                continue

            if stripped.startswith("MoleculeDefinitions:"):
                components[cur_name]["force_field_model"] = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("MolFraction:"):
                components[cur_name]["mol_fraction"] = self._extract_first_number(stripped)
            elif stripped.startswith("Compressibility:"):
                components[cur_name]["compressibility"] = self._extract_first_number(stripped)
            elif stripped.startswith("Density of the bulk fluid phase:"):
                components[cur_name]["bulk_density_kg_m3"] = self._extract_first_number(stripped)
            elif stripped.startswith("Fugacity coefficient:"):
                components[cur_name]["fugacity_coefficient"] = self._extract_first_number(stripped)
            elif stripped.startswith("Partial pressure:"):
                pending_pressure = {"Pa": self._extract_first_number(stripped)}
                pending_fugacity = None
                components[cur_name]["partial_pressure"] = pending_pressure
            elif stripped.startswith("Partial fugacity:"):
                pending_pressure = None
                pending_fugacity = {"Pa": self._extract_first_number(stripped)}
                components[cur_name]["partial_fugacity"] = pending_fugacity
            elif pending_pressure is not None:
                val = self._extract_first_number(stripped)
                if val is not None:
                    if "[Torr]" in stripped:
                        pending_pressure["Torr"] = val
                    elif "[bar]" in stripped:
                        pending_pressure["bar"] = val
                    elif "[atm]" in stripped:
                        pending_pressure["atm"] = val
            elif pending_fugacity is not None:
                val = self._extract_first_number(stripped)
                if val is not None:
                    if "[Torr]" in stripped:
                        pending_fugacity["Torr"] = val
                    elif "[bar]" in stripped:
                        pending_fugacity["bar"] = val
                    elif "[atm]" in stripped:
                        pending_fugacity["atm"] = val

        return conditions, components

    def _parse_properties_computed(self, text: str):
        props: Dict[str, bool] = {}
        in_block = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped == "Properties computed":
                in_block = True
                continue
            if in_block and stripped.startswith("VTK"):
                break
            if not in_block or ":" not in stripped:
                continue
            key, value = stripped.rsplit(":", 1)
            value = value.strip().lower()
            if value in ("yes", "no"):
                props[key.strip()] = (value == "yes")
        return props

    def _parse_average_energy_block(self, text: str, header: str):
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.strip() != header:
                continue
            for j in range(i + 1, min(i + 20, len(lines))):
                avg_m = self.AVG_ENERGY_RE.match(lines[j])
                if not avg_m:
                    continue
                result = {
                    "total_K": self._to_float(avg_m.group(1)),
                    "vdw_K": self._to_float(avg_m.group(2)),
                    "coulomb_K": self._to_float(avg_m.group(3)),
                }
                if j + 1 < len(lines):
                    err_m = self.ERR_ENERGY_RE.match(lines[j + 1])
                    if err_m:
                        result.update({
                            "total_error_K": self._to_float(err_m.group(1)),
                            "vdw_error_K": self._to_float(err_m.group(2)),
                            "coulomb_error_K": self._to_float(err_m.group(3)),
                        })
                return result
        return None

    def _parse_scalar_average_block(self, text: str, header: str):
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.strip() != header:
                continue
            for j in range(i + 1, min(i + 25, len(lines))):
                stripped = lines[j].strip()
                if stripped.startswith("Average"):
                    val, err, unit = self._parse_value_error_unit_line(stripped)
                    if val is not None:
                        return {"value": val, "error": err, "unit": unit}
        return None

    def _parse_widom_summary(self, text: str):
        widom: Dict[str, Dict[str, Any]] = {}
        patterns = [
            ("rosenbluth_weight", re.compile(r"\[([^\]]+)\]\s+Average Widom Rosenbluth-weight:\s*([-+0-9Ee\.]+)\s*\+/-\s*([-+0-9Ee\.]+)", re.IGNORECASE), None),
            ("chemical_potential_K", re.compile(r"\[([^\]]+)\]\s+Average chemical potential:\s*([-+0-9Ee\.]+)\s*\+/-\s*([-+0-9Ee\.]+)\s*\[K\]", re.IGNORECASE), "K"),
            ("ideal_gas_chemical_potential_K", re.compile(r"\[([^\]]+)\]\s+Average Widom Ideal-gas chemical potential:\s*([-+0-9Ee\.]+)\s*\+/-\s*([-+0-9Ee\.]+)", re.IGNORECASE), "K"),
            ("excess_chemical_potential_K", re.compile(r"\[([^\]]+)\]\s+Average Widom excess chemical potential:\s*([-+0-9Ee\.]+)\s*\+/-\s*([-+0-9Ee\.]+)", re.IGNORECASE), "K"),
            ("henry_coefficient", re.compile(r"\[([^\]]+)\]\s+Average Henry coefficient:\s*([-+0-9Ee\.]+)\s*\+/-\s*([-+0-9Ee\.]+)\s*\[([^\]]+)\]", re.IGNORECASE), None),
            ("adsorption_energy_widom", re.compile(r"\[([^\]]+)\]\s+Average\s+<U_gh>_1-<U_h>_0:\s*([-+0-9Ee\.]+)\s*\+/-\s*([-+0-9Ee\.]+)\s*\[K\]\s*\(\s*([-+0-9Ee\.]+)\s*\+/-\s*([-+0-9Ee\.]+)\s*kJ/mol\)", re.IGNORECASE), None),
        ]

        for line in text.splitlines():
            for key, pattern, unit in patterns:
                m = pattern.search(line)
                if not m:
                    continue
                comp = m.group(1).strip()
                widom.setdefault(comp, {})
                if key == "henry_coefficient":
                    widom[comp][key] = {
                        "value": self._to_float(m.group(2)),
                        "error": self._to_float(m.group(3)),
                        "unit": m.group(4).strip(),
                    }
                elif key == "adsorption_energy_widom":
                    widom[comp][key] = {
                        "value_K": self._to_float(m.group(2)),
                        "error_K": self._to_float(m.group(3)),
                        "value_kJ_mol": self._to_float(m.group(4)),
                        "error_kJ_mol": self._to_float(m.group(5)),
                    }
                else:
                    widom[comp][key] = {
                        "value": self._to_float(m.group(2)),
                        "error": self._to_float(m.group(3)),
                        "unit": unit,
                    }
        return widom

    def _parse_simulation_diagnostics(self, text: str):
        diagnostics: Dict[str, Any] = {}
        m = re.search(r"Simulation finished,\s*(\d+)\s+warnings", text, re.IGNORECASE)
        if m:
            diagnostics["warnings"] = int(m.group(1))
        m = re.search(r"Total energy-drift:\s*([-+0-9Ee\.]+)", text, re.IGNORECASE)
        if m:
            diagnostics["total_energy_drift"] = self._to_float(m.group(1))
        return diagnostics

    def _parse_raspa_summary_from_data(self, text: str):
        conditions, components = self._parse_conditions_and_components(text)
        enthalpy_value, enthalpy_error, enthalpy_unit = self._parse_enthalpy_adsorption_kjmol(text)
        loadings = self._parse_loading_table(text)
        properties = self._parse_properties_computed(text)

        target_uptake = None
        target_uptake_error = None
        for comp, comp_loadings in loadings.items():
            target = comp_loadings.get("excess", {}).get(self.TARGET_UNIT_SUBSTR)
            if target:
                target_uptake = target.get("value")
                target_uptake_error = target.get("error")
                break

        adsorption_site_flags = {
            "histogram_of_molecule_positions": properties.get("Histogram of the molecule positions", False),
            "free_energy_profiles": properties.get("Free energy profiles", False),
            "three_d_density_grid_for_adsorbates": properties.get("3D density grid for adsorbates", False),
            "compute_cation_or_adsorption_sites": properties.get("Compute cation an/or adsorption sites", False),
            "movies": properties.get("Movies", False),
        }

        return {
            "conditions": conditions,
            "components": components,
            "loadings": loadings,
            "target_uptake": {
                "mode": "excess",
                "unit": self.TARGET_UNIT_SUBSTR,
                "value": target_uptake,
                "error": target_uptake_error,
            },
            "thermodynamics": {
                "enthalpy_of_adsorption": {
                    "value": enthalpy_value,
                    "error": enthalpy_error,
                    "unit": enthalpy_unit,
                },
                "qst": {
                    "value": -enthalpy_value if enthalpy_value is not None else None,
                    "error": enthalpy_error,
                    "unit": enthalpy_unit,
                    "definition": "Qst = - enthalpy_of_adsorption",
                },
                "derivative_mu_density": self._parse_scalar_average_block(
                    text, "derivative of the chemical potential with respect to density (constant T,V):"
                ),
            },
            "energies": {
                "host_adsorbate": self._parse_average_energy_block(text, "Average Host-Adsorbate energy:"),
                "adsorbate_adsorbate": self._parse_average_energy_block(text, "Average Adsorbate-Adsorbate energy:"),
                "total_energy": self._parse_scalar_average_block(text, "Total energy:"),
            },
            "widom": self._parse_widom_summary(text),
            "properties_computed": properties,
            "adsorption_site_data": {
                "available": any(adsorption_site_flags.values()),
                "flags": adsorption_site_flags,
                "note": "Preferred adsorption locations require molecule-position histograms, density grids, adsorption-site output, or snapshots; scalar uptake output alone is insufficient.",
            },
            "diagnostics": self._parse_simulation_diagnostics(text),
        }
    
    def _parse_enthalpy_adsorption_kjmol(self, text: str):
        in_block = False
        for line in text.splitlines():
            if "Enthalpy of adsorption" in line:
                in_block = True
                continue
            if in_block:
                m = self.ENTHALPY_ADS_RE.search(line)
                if m:
                    H = float(m.group(1))
                    dH = float(m.group(2))
                    return H, dH, "kJ/mol"
                
                if line.strip().startswith("derivative of the chemical potential"):
                    break
        return None, None, None

    def _parse_henry_from_data(self, text: str):
        for line in text.splitlines():
            m = self.HENRY_RE.search(line)
            if not m:
                continue
            try:
                henry = float(m.group(1))
                err = float(m.group(2))
                unit = m.group(3).strip()
                return henry, err, unit
            except Exception:
                return None, None, None

        return None, None, None

    def _parse_component_loadings(self, text: str,
                                mode: str = "excess",
                                unit_substr: str = "mol/kg framework"):
        loads = {}
        cur_name = None
        cur_idx = None

        for line in text.splitlines():
            m = self.COMPONENT_HEADER_RE.match(line)
            if m:
                cur_idx = int(m.group(1))
                cur_name = m.group(2).strip()
                continue

            if cur_name is None:
                continue

            m2 = self.LOADING_LINE_RE.match(line)
            if not m2:
                continue

            which = m2.group(1).lower().strip()   
            unit = m2.group(2).strip()
            val = float(m2.group(3))
            err = float(m2.group(4))

            if which != mode:
                continue
            if unit_substr.lower() not in unit.lower():
                continue

            
            if cur_name not in loads:
                loads[cur_name] = (val, err, unit)

        return loads


    def _pick_loading_for_selectivity(self, comp_dict: dict):
        if not comp_dict:
            return None, None, None, None

        if "absolute" in comp_dict:
            d = comp_dict["absolute"]
            return d["value"], d["error"], d["unit"], "absolute"

        if "excess" in comp_dict:
            d = comp_dict["excess"]
            return d["value"], d["error"], d["unit"], "excess"

        return None, None, None, None

    def _find_data_files(self, work_dir: Path):
        data_root = work_dir / "Output" / "System_0"
        if not data_root.is_dir():
            return []
        return sorted(data_root.glob("*.data"))

    def _save_isotherm_artifacts(
        self,
        work_dir: Path,
        mof: str,
        guest: str,
        points: List[Tuple[float, float, Optional[float]]],
        uptake_units: str
    ):
        
        csv_path = work_dir / "isotherm_uptake.csv"
        with csv_path.open("w", encoding="utf-8") as f:
            f.write("pressure_bar,uptake_excess,uptake_excess_error,uptake_units\n")
            for p, q, dq in points:
                f.write(f"{p},{q},{'' if dq is None else dq},{uptake_units}\n")

        
        import matplotlib.pyplot as plt

        pressures = [p for p, _, _ in points]
        uptakes = [q for _, q, _ in points]
        errs = [dq for _, _, dq in points]

        plt.figure()
        
        if all(e is not None for e in errs):
            plt.errorbar(pressures, uptakes, yerr=errs, fmt="o-")
        else:
            plt.plot(pressures, uptakes, "o-")

        plt.xscale("log")  
        plt.xlabel("Pressure [bar]")
        plt.ylabel(f"CO2 uptake (excess) [{uptake_units}]")
        plt.title(f"{mof} {guest} adsorption isotherm")
        plt.grid(True, which="both")
        plt.tight_layout()

        png_path = work_dir / "isotherm_uptake.png"
        plt.savefig(png_path, dpi=300)
        plt.close()

        return str(csv_path), str(png_path)

    def _fit_low_pressure_slope(self, points, p_max=0.1):
        lp = [(p, q) for p, q, _ in points if p <= p_max]
        if len(lp) < 2:
            return None

        xs = [p for p, _ in lp]
        ys = [q for _, q in lp]
        n = len(xs)
        sx = sum(xs); sy = sum(ys)
        sxx = sum(x*x for x in xs)
        sxy = sum(x*y for x, y in zip(xs, ys))
        denom = (n * sxx - sx * sx)
        if denom == 0:
            return None
        slope = (n * sxy - sx * sy) / denom
        return slope


    def _run_single(self, context: Dict[str, Any]) -> Dict[str, Any]:
        status = context.get("raspa_status")
        work_dir_str = context.get("work_dir")

        print(f"\n=== RASPAOutputAgent: parsing output in {work_dir_str} ===")

        if status in ("submitted", "running"):
            print(f"[RASPAOutputAgent] raspa_status={status} -> simulation is still running; skipping output parsing.")
            return context

        if not work_dir_str:
            raise RuntimeError("[RASPAOutputAgent] context['work_dir'] is missing.")

        work_dir = Path(work_dir_str)
        if not work_dir.is_dir():
            raise FileNotFoundError(f"[RASPAOutputAgent] work_dir does not exist: {work_dir}")

        data_files = self._find_data_files(work_dir)
        if not data_files:
            print("[RASPAOutputAgent] Could not find Output/System_0/*.data files.")
            context.setdefault("results", {})
            context["results"]["uptake_excess"] = None
            context["results"]["uptake_units"] = None
            context["results"]["uptake_error"] = None
            context["results"]["raspa_summary"] = None
            context["results"]["raspa_output_file"] = None
            context["results"]["raspa_parse_status"] = "no_data_file"
            return context

        prop = (context.get("property") or "").strip().lower().replace(" ", "_").replace("-", "_")
        is_henry = prop in ("henry", "henry_constant", "kh", "henry_const", "henry_coefficient")
        is_qst = prop in ("qst", "isosteric_heat", "heat_of_adsorption", "enthalpy_of_adsorption")
        is_selectivity = prop in ("selectivity", "binary_selectivity")

        context.setdefault("results", {})

        
        
        
        if is_henry:
            henry_val = None
            henry_err = None
            henry_unit = None
            raspa_summary = None
            used_file: Optional[Path] = None

            for df in data_files:
                try:
                    text = df.read_text()
                except Exception as e:
                    print(f"[RASPAOutputAgent] Failed to read {df}: {e}")
                    continue

                v, e_, u = self._parse_henry_from_data(text)
                if v is not None:
                    henry_val, henry_err, henry_unit = v, e_, u
                    raspa_summary = self._parse_raspa_summary_from_data(text)
                    used_file = df
                    break

            context["results"]["henry_constant"] = henry_val
            context["results"]["henry_error"] = henry_err
            context["results"]["henry_units"] = henry_unit
            context["results"]["raspa_summary"] = raspa_summary
            context["results"]["raspa_output_file"] = str(used_file) if used_file else None
            context["results"]["raspa_parse_status"] = "ok" if henry_val is not None else "parse_failed"

            if henry_val is None:
                print("[RASPAOutputAgent] Could not find the Henry coefficient line.")
            else:
                print(f"[RASPAOutputAgent] Henry = {henry_val} ± {henry_err} [{henry_unit}] from {used_file.name}")

            return context

        
        
        
        if is_qst:
            H_val = None
            H_err = None
            raspa_summary = None
            used_file: Optional[Path] = None

            for df in data_files:
                try:
                    text = df.read_text()
                except Exception as e:
                    print(f"[RASPAOutputAgent] Failed to read {df}: {e}")
                    continue

                v, e_, _unit = self._parse_enthalpy_adsorption_kjmol(text)
                if v is not None:
                    H_val, H_err = v, e_
                    raspa_summary = self._parse_raspa_summary_from_data(text)
                    used_file = df
                    break

            
            qst_val = (-H_val) if H_val is not None else None

            context["results"]["enthalpy_of_adsorption"] = H_val
            context["results"]["enthalpy_of_adsorption_error"] = H_err
            context["results"]["enthalpy_of_adsorption_units"] = "kJ/mol"

            context["results"]["qst"] = qst_val
            context["results"]["qst_error"] = H_err
            context["results"]["qst_units"] = "kJ/mol"

            context["results"]["raspa_summary"] = raspa_summary
            context["results"]["raspa_output_file"] = str(used_file) if used_file else None
            context["results"]["raspa_parse_status"] = "ok" if H_val is not None else "parse_failed"

            if H_val is None:
                print("[RASPAOutputAgent] Could not find the enthalpy of adsorption line.")
            else:
                print(f"[RASPAOutputAgent] Enthalpy_ads = {H_val} ± {H_err} [kJ/mol] -> Qst = {qst_val} [kJ/mol] from {used_file.name}")

            return context

        
        
        
        if is_selectivity:
            used_file = None
            loads = None
            raspa_summary = None

            for df in data_files:
                try:
                    text = df.read_text()
                except Exception as e:
                    print(f"[RASPAOutputAgent] Failed to read {df}: {e}")
                    continue

                loads = self._parse_component_loadings(
                    text, mode="excess", unit_substr="mol/kg framework"
                )
                if loads:
                    used_file = df
                    raspa_summary = self._parse_raspa_summary_from_data(text)
                    break

            context.setdefault("results", {})

            if not loads:
                print("[RASPAOutputAgent] Could not find component loading data for selectivity.")
                context["results"]["raspa_parse_status"] = "parse_failed"
                context["results"]["raspa_summary"] = raspa_summary
                context["results"]["raspa_output_file"] = str(used_file) if used_file else None
                return context

            
            y = context.get("gas_fractions") or {}
            guests = context.get("guests") or list(loads.keys())

            
            g0, g1 = guests[0], guests[1]

            if g0 not in loads or g1 not in loads:
                
                
                keys = list(loads.keys())
                def _find(name):
                    name_u = name.upper()
                    for k in keys:
                        if k.upper() == name_u:
                            return k
                    return None
                k0 = _find(g0) or g0
                k1 = _find(g1) or g1
            else:
                k0, k1 = g0, g1

            x0, dx0, unit0 = loads[k0]
            x1, dx1, unit1 = loads[k1]

            y0 = float(y.get(g0, 0.5))
            y1 = float(y.get(g1, 0.5))

            
            if x1 == 0 or y0 == 0:
                S = None
            else:
                S = (x0 / x1) / (y0 / y1)

            context["results"]["component_loadings_excess_molkg"] = {
                k0: {"value": x0, "error": dx0, "unit": unit0},
                k1: {"value": x1, "error": dx1, "unit": unit1},
            }
            context["results"]["selectivity"] = S
            context["results"]["selectivity_definition"] = f"(x_{g0}/x_{g1}) / (y_{g0}/y_{g1})"
            context["results"]["raspa_summary"] = raspa_summary
            context["results"]["raspa_output_file"] = str(used_file) if used_file else None
            context["results"]["raspa_parse_status"] = "ok" if S is not None else "parse_failed"

            print(f"[RASPAOutputAgent] load({k0})={x0}, load({k1})={x1}, y={y0}/{y1} => S={S}")
            return context
            
        
        
        
        uptake_value = None
        uptake_error = None
        uptake_units = None
        raspa_summary = None
        used_file: Optional[Path] = None

        for df in data_files:
            try:
                text = df.read_text()
            except Exception as e:
                print(f"[RASPAOutputAgent] Failed to read {df}: {e}")
                continue

            val, units = self._parse_uptake_from_data(text)
            if val is not None:
                uptake_value = val
                uptake_units = units
                raspa_summary = self._parse_raspa_summary_from_data(text)
                uptake_error = (raspa_summary.get("target_uptake", {}) or {}).get("error")
                used_file = df
                break

        context["results"]["uptake_excess"] = uptake_value
        context["results"]["uptake_error"] = uptake_error
        context["results"]["uptake_units"] = uptake_units
        context["results"]["raspa_summary"] = raspa_summary
        context["results"]["raspa_output_file"] = str(used_file) if used_file else None
        context["results"]["raspa_parse_status"] = "ok" if uptake_value is not None else "parse_failed"

        if uptake_value is None:
            print("[RASPAOutputAgent] Could not find the 'Average loading excess' line.")
        else:
            print(f"[RASPAOutputAgent] uptake(excess) = {uptake_value} [{uptake_units}] from {used_file.name}")

        return context


    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        
        if "batch" in context and isinstance(context["batch"], list) and len(context["batch"]) > 1:
            context.setdefault("results", {})
            batch: List[Dict[str, Any]] = context["batch"]
            top_n = context.get("top_n")  

            print(f"\n=== RASPAOutputAgent(BATCH): parsing {len(batch)} jobs ===")

            parsed_batch: List[Dict[str, Any]] = []
            for subctx in batch:
                parsed_batch.append(self._run_single(subctx))

            prop = (context.get("property") or "").strip().lower().replace(" ", "_").replace("-", "_")
            is_henry = prop in ("henry", "henry_constant", "kh", "henry_const")
            is_selectivity = prop in ("selectivity", "co2_n2_selectivity", "binary_selectivity")

            
            pressures = []
            for b in parsed_batch:
                p = b.get("pressure_bar", None)
                if p is None:
                    continue
                try:
                    pressures.append(float(p))
                except Exception:
                    pass
                
            is_isotherm_batch = (prop in ("uptake","adsorption_isotherm","isotherm")) and (len(set(pressures)) >= 2)

            if is_henry:
                ok = [b for b in parsed_batch if b.get("results", {}).get("henry_constant") is not None]
                
                ok.sort(key=lambda b: b["results"]["henry_constant"], reverse=True)

            elif is_selectivity:
                ok = [b for b in parsed_batch if b.get("results", {}).get("selectivity") is not None]
                ok.sort(key=lambda b: b["results"]["selectivity"], reverse=True)

            else:
                ok = [b for b in parsed_batch if b.get("results", {}).get("uptake_excess") is not None]

                
                if is_isotherm_batch:
                    ok.sort(key=lambda b: float(b.get("pressure_bar", 0.0)))
                else:
                    ok.sort(key=lambda b: b["results"]["uptake_excess"], reverse=True)

            
            if isinstance(top_n, int) and top_n > 0 and (not is_isotherm_batch):
                ok_top = ok[:top_n]
            else:
                ok_top = ok

            
            if is_isotherm_batch:
                work_dir = Path(context.get("work_dir", "."))
                work_dir.mkdir(parents=True, exist_ok=True)

                mof = context.get("mof", "MOF")

                guests = context.get("guests") or []
                if isinstance(guests, str):
                    guests = [guests]
                if not guests:
                    guests = ["guest"]

                guest_label = "+".join(guests)   

                points = []
                uptake_units = None

                for b in ok_top:
                    p = b.get("pressure_bar", None)
                    q = b.get("results", {}).get("uptake_excess", None)
                    u = b.get("results", {}).get("uptake_units", None)

                    if p is None or q is None:
                        continue
                    try:
                        p = float(p); q = float(q)
                    except Exception:
                        continue

                    uptake_units = uptake_units or (u or "")
                    points.append((p, q, None))  

                if len(points) >= 2:
                    csv_path, png_path = self._save_isotherm_artifacts(
                        work_dir=work_dir,
                        mof=mof,
                        guest=guest_label,            
                        points=points,
                        uptake_units=uptake_units or ""
                    )

                    slope = self._fit_low_pressure_slope(points, p_max=0.1)

                    context["results"]["isotherm_artifacts"] = {
                        "csv_path": csv_path,
                        "png_path": png_path,
                        "uptake_units": uptake_units,
                        "low_pressure_fit_max_bar": 0.1,
                        "low_pressure_slope": slope,
                        "n_points": len(points),
                        "guests": guests,             
                    }

            
            context["results"]["raspa_batch_summary"] = {
                "total": len(parsed_batch),
                "success": len(ok),
                "top_n": top_n if isinstance(top_n, int) else None,
                "is_isotherm_batch": is_isotherm_batch,
                "ranked": [
                    {
                        "mof": b.get("mof"),
                        "work_dir": b.get("work_dir"),
                        "pressure_bar": b.get("pressure_bar"),
                        "uptake_excess": b["results"].get("uptake_excess"),
                        "uptake_error": b["results"].get("uptake_error"),
                        "uptake_units": b["results"].get("uptake_units"),
                        "raspa_summary": b["results"].get("raspa_summary"),
                        "raspa_output_file": b["results"].get("raspa_output_file"),
                    }
                    for b in ok_top
                ],
            }

            
            context["batch"] = ok_top

            
            try:
                work_dir = Path(context.get("work_dir", "."))
                out = work_dir / "raspa_batch_summary.json"
                out.write_text(json.dumps(context["results"]["raspa_batch_summary"], indent=2), encoding="utf-8")
                context["results"]["raspa_batch_summary_path"] = str(out)
            except Exception as e:
                print(f"[RASPAOutputAgent] Warning: batch summary save failed: {e}")

            print(f"[RASPAOutputAgent(BATCH)] total={len(parsed_batch)} success={len(ok)} returned={len(ok_top)}")
            return context

        
        return self._run_single(context)
