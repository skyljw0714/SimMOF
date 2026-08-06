from __future__ import annotations

import json
import os
import re
import shutil

from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

from config import RASPA_DIR as RASPA_DIR_STR
from langchain.schema import HumanMessage, SystemMessage
from core.llm_logging import log_llm_decision, set_llm_context
from input.interactive_review import maybe_interactive_review_input_file
from input.raspa.prompt import (
    render_raspa_format,
    create_raspa_input_prompt,
    get_raspa_system_message,
)

RASPA_CUTOFF_VDW = 12.8


def _pick_snippet(simulation_input: dict, software: str) -> str:
    if not simulation_input:
        return ""
    for s in (simulation_input.get("snippets") or []):
        if (s.get("software") == software) and (s.get("text") or "").strip():
            return s["text"].strip()
    return ""

RASPA_REPRO_PATCH_SYSTEM = """You are a careful text editor for RASPA simulation.input files.
Return ONLY the patched simulation.input text. No markdown. No explanations."""

RASPA_REPRO_PATCH_USER = """Patch the original RASPA simulation.input by applying ONLY the required replacements below.

HARD RULES:
1) MINIMAL CHANGE: Do not alter any lines except where needed to apply REQUIRED REPLACEMENTS.
2) Preserve all other settings exactly as-is (cycles, probabilities like RegrowProbability/SwapProbability/WidomProbability,
   forcefield, charge settings, cutoffs, move probabilities, etc.) unless replacement requires it.
3) If a required field is missing in the original input, insert it in the most appropriate location:
   - FrameworkName and UnitCells belong under the "Framework 0" section.
   - UnitCells should appear after FrameworkName.
   - Component i MoleculeName / MoleculeDefinition must be within the corresponding Component i block.
4) Do NOT rename or reorder components.
5) Output MUST be a valid RASPA input file.

REQUIRED REPLACEMENTS (JSON):
{replacements_json}

ORIGINAL INPUT:
<<<{original_text}>>>
"""

RASPA_MANUAL_REVIEW_SYSTEM = """You are a careful RASPA 2.0 simulation.input reviewer.
Return ONLY the final simulation.input text. No markdown. No explanations."""

RASPA_MANUAL_REVIEW_USER = """Review the generated RASPA simulation.input using ONLY the retrieved RASPA manual evidence.

HARD RULES:
1) Preserve the generated input exactly unless the manual evidence clearly supports a minimal correction.
2) Do not change force field names, molecule names, molecule definitions, framework names, unit cells, temperature, pressure, charges, cutoffs, cycle counts, or move probabilities unless the user explicitly requested that change.
3) Prefer exact RASPA keyword spelling from the manual evidence when fixing or adding a keyword.
4) Do not add unrelated optional outputs or advanced settings.
5) If no correction is needed, return the original generated input unchanged.

USER REQUEST:
{query_text}

PROPERTY:
{property_name}

RASPA_MANUAL_RAG_HINTS:
{manual_hints}

GENERATED INPUT:
<<<{input_text}>>>
"""


RASPA_DIR = Path(RASPA_DIR_STR)
class RASPAInputAgent:
    SUPERCELL_CUTOFF = RASPA_CUTOFF_VDW

    def __init__(self, llm=None):
        from config import LLM_DEFAULT
        self.llm = llm or LLM_DEFAULT

        
        self.structures_cif_dir = RASPA_DIR / "share/raspa/structures/cif"
        self.forcefield_dir = RASPA_DIR / "share/raspa/forcefield"
        self.molecules_dir = RASPA_DIR / "share/raspa/molecules"

        
        self.available_forcefields: List[str] = self._list_forcefields()
        self.molecule_families: Dict[str, List[str]] = self._build_molecule_family_index()

    
    def _extract_cif_charges(self, cif_path: Path) -> List[float]:
        charges: List[float] = []
        try:
            lines = cif_path.read_text(errors="ignore").splitlines()
        except Exception:
            return charges

        i = 0
        n = len(lines)
        while i < n:
            line = lines[i].strip()
            if line.lower().startswith("loop_"):
                
                headers = []
                j = i + 1
                while j < n:
                    s = lines[j].strip()
                    if not s:
                        j += 1
                        continue
                    if s.startswith("_"):
                        headers.append(s.lower())
                        j += 1
                        continue
                    break  
                if not headers:
                    i = j
                    continue

                
                if "_atom_site_charge" not in headers:
                    i = j
                    continue
                charge_idx = headers.index("_atom_site_charge")

                
                k = j
                while k < n:
                    s = lines[k].strip()
                    if not s:
                        k += 1
                        continue
                    if s.lower().startswith("loop_") or s.startswith("_"):
                        break
                    parts = s.split()
                    if len(parts) > charge_idx:
                        try:
                            charges.append(float(parts[charge_idx]))
                        except Exception:
                            pass
                    k += 1

                return charges  
            i += 1

        return charges

    def _cif_charges_look_reasonable(self, charges: List[float]) -> bool:
        if not charges:
            return False

        
        if all(abs(q) < 1e-6 for q in charges):
            return False

        
        if (max(charges) - min(charges)) < 1e-4:
            return False

        
        if any(abs(q) > 5.0 for q in charges):
            return False

        
        if abs(sum(charges)) > 5.0:
            return False

        return True

    def _extract_atom_types_from_def(self, family: str, mol_name: str) -> List[str]:
        def_path = self.molecules_dir / family / f"{mol_name}.def"
        if not def_path.exists():
            raise FileNotFoundError("DEF file not found: {}".format(def_path))

        lines = def_path.read_text(errors="ignore").splitlines()

        start_idx = None
        for i, ln in enumerate(lines):
            if "atomic positions" in ln.lower():
                start_idx = i + 1
                break

        search_lines = lines[start_idx:] if start_idx is not None else lines

        atom_types = []
        for ln in search_lines:
            s = ln.strip()
            if not s:
                continue
            if s.startswith("#"):
                if start_idx is not None and atom_types:
                    break
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            if not parts[0].isdigit():
                continue
            atom_types.append(parts[1])

        
        seen = set()
        uniq = []
        for t in atom_types:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        return uniq
    
    def _parse_pseudo_atoms_charges_uff_style(self, forcefield: str) -> Dict[str, float]:
        pseudo_atoms_path = self.forcefield_dir / forcefield / "pseudo_atoms.def"
        if not pseudo_atoms_path.exists():
            raise FileNotFoundError("pseudo_atoms.def not found: {}".format(pseudo_atoms_path))

        charges: Dict[str, float] = {}
        for ln in pseudo_atoms_path.read_text(errors="ignore").splitlines():
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 7:
                continue

            atom_type = parts[0]
            charge_tok = parts[6]  

            try:
                q = float(charge_tok)
            except Exception:
                continue

            charges[atom_type] = q

        return charges


    def _guest_has_charge_from_forcefield(
        self,
        forcefield: str,
        family: str,
        mol_name: str,
        eps: float = 1e-12
    ) -> Tuple[bool, List[Tuple[str, float]]]:
        atom_types = self._extract_atom_types_from_def(family, mol_name)
        ff_charges = self._parse_pseudo_atoms_charges_uff_style(forcefield)

        hits: List[Tuple[str, float]] = []
        missing: List[str] = []
        for t in atom_types:
            if t not in ff_charges:
                missing.append(t)
                continue
            q = ff_charges[t]
            if abs(q) > eps:
                hits.append((t, q))

        
        return (len(hits) > 0), hits

    @staticmethod
    def _fix_pacman_cif_loops(cif_path: Path) -> None:
        text = cif_path.read_text(errors="replace")
        fixed = re.sub(
            r"('x, y, z'\s*\n)(\s*_atom_site_)",
            r"\1\nloop_\n\2",
            text,
        )
        if fixed != text:
            cif_path.write_text(fixed)

    def _run_pacman_on_cif(self, cif_path: Path) -> bool:
        import shutil
        try:
            from PACMANCharge.pmcharge import predict
            predict(str(cif_path), charge_type="DDEC6")
            pacman_out = Path(str(cif_path).split(".cif")[0] + "_pacman.cif")
            if not pacman_out.exists():
                print("[RASPAInputAgent] PACMAN output file not found")
                return False
            shutil.copy2(pacman_out, cif_path)
            pacman_out.unlink(missing_ok=True)
            self._fix_pacman_cif_loops(cif_path)
            cif_charges = self._extract_cif_charges(cif_path)
            ok = self._cif_charges_look_reasonable(cif_charges)
            if ok:
                print(f"[RASPAInputAgent] PACMAN DDEC6 charges written → {cif_path}")
            else:
                print("[RASPAInputAgent] PACMAN ran but charges look unreasonable")
            return ok
        except Exception as e:
            print(f"[RASPAInputAgent] PACMAN failed: {e}")
            return False

    def _decide_charge_method_with_llm(
        self,
        context: Dict[str, Any],
        cif_has_charges: bool,
    ) -> str:
        _POLAR_GUESTS = {
            "co2", "h2o", "water", "so2", "h2s", "no", "nh3", "no2", "hcn",
            "ch3oh", "methanol", "ethanol", "acetone", "dmf", "dmso",
        }
        if self.llm is None:
            if cif_has_charges:
                return "cif"
            guest_raw = (context.get("guest") or "").strip().lower()
            if any(p in guest_raw for p in _POLAR_GUESTS):
                return "eqeq"
            return "none"

        mof   = context.get("mof", "")
        guest = context.get("guest", "")
        prop  = context.get("property", "")

        query_text = context.get("query_text", "")
        charge_hints = (context.get("raspa_rag_hints") or {}).get("charge_hints", "")

        system_msg = (
            "You decide whether and how to assign partial charges for a RASPA GCMC simulation.\n"
            "Return ONLY JSON: {\"method\": \"<choice>\", \"reason\": \"<one sentence>\"}\n"
            "\n"
            "Choices:\n"
            "  none   — no charges needed (nonpolar guest, electrostatics negligible)\n"
            "  cif    — use charges already in the CIF file\n"
            "  eqeq   — compute charges with EQeq (RASPA built-in; lower accuracy than PACMAN)\n"
            "  pacman — ML-predicted charges (DDEC6-quality, no DFT, fast; better accuracy than EQeq for polar guests)\n"
            "  ddec   — true DFT-based DDEC6 via VASP + CHARGEMOL (exact, slow; use when charge accuracy is critical)\n"
            "\n"
            "Decision guide:\n"
            "- Consider whether the guest molecule has a significant charge or multipole moment.\n"
            "  Framework charges only matter when the guest itself has significant electrostatic interactions.\n"
            "- CIF already has charges: cif (always prefer this).\n"
            "- Screening / high-throughput: prefer a fast method that does not require DFT.\n"
            "- Without RAG_HINTS: weigh the cost of DFT-based methods against the accuracy needed.\n"
            "  ddec requires full DFT + CHARGEMOL calculation; reserve it for cases where the user\n"
            "  explicitly requests high accuracy or DFT charges.\n"
            "- RAG_HINTS (if provided) reflect charge methods used in similar published studies.\n"
            "  Weight them heavily: if two or more sources consistently mention a specific method (e.g., DDEC, REPEAT, EQeq),\n"
            "  prefer that method over the default even when the query does not explicitly request it.\n"
            "  Override the default only when the RAG signal is clear and consistent; a single ambiguous mention is not enough.\n"
            "  When evaluating RAG_HINTS, always consider the polarity of the guest molecule: framework charges only matter\n"
            "  when the guest itself carries a charge or significant multipole moment."
        )

        prompt = (
            f"User query: {query_text}\n"
            f"MOF: {mof}\n"
            f"Guest: {guest}\n"
            f"Property: {prop}\n"
            f"CIF has charges: {cif_has_charges}\n"
        )
        if charge_hints:
            prompt += f"\nRAG_HINTS (charge methods used in similar studies):\n{charge_hints}\n"
        prompt += "\nWhich charge method should be used?"

        try:
            set_llm_context("RASPAInputAgent", "charge_method_selection")
            resp = self.llm.invoke([
                SystemMessage(content=system_msg),
                HumanMessage(content=prompt),
            ])
            text = resp.content.strip()
            if text.startswith("```"):
                text = "\n".join(text.splitlines()[1:-1]).strip()
            obj = json.loads(text)
            method = str(obj.get("method", "none")).strip().lower()
            reason = str(obj.get("reason", "")).strip()
            if method not in ("none", "cif", "eqeq", "pacman", "ddec"):
                method = "cif" if cif_has_charges else "none"
            print(f"[RASPAInputAgent] Charge method: {method} — {reason}")
            return method
        except Exception as e:
            print(f"[RASPAInputAgent] Charge method LLM failed: {e}")
            return "cif" if cif_has_charges else "none"

    def _decide_charge_settings(
        self,
        cif_path: Path,
        forcefield: str,
        guests: List[Tuple[str, str]] = None,
        cutoff: float = 12.8,
        ewald_precision: str = "1e-6",
        context: Dict[str, Any] = None,
    ) -> Dict[str, str]:

        cif_charges = self._extract_cif_charges(cif_path)
        framework_ok = self._cif_charges_look_reasonable(cif_charges)

        guest_has_charge = False
        if guests:
            for fam, name in guests:
                has_q, hits = self._guest_has_charge_from_forcefield(forcefield, fam, name)
                if has_q:
                    guest_has_charge = True
                    break

        charge_method = "none"
        if context is not None:
            charge_method = self._decide_charge_method_with_llm(context, cif_has_charges=framework_ok)
        elif framework_ok or guest_has_charge:
            charge_method = "cif" if framework_ok else "eqeq"

        use_cif = "yes" if (framework_ok and charge_method in ("cif", "eqeq", "pacman", "ddec")) else "no"

        if charge_method == "none":
            charge_block = "ChargeMethod                  None"
        elif charge_method == "eqeq":
            charge_block = "ChargeMethod                  Ewald\nChargeEquilibration           EQeq"
            use_cif = "no"
        elif charge_method == "pacman":
            if not framework_ok:
                pacman_ok = self._run_pacman_on_cif(cif_path)
                if pacman_ok:
                    framework_ok = True
                    use_cif = "yes"
                else:
                    print("[RASPAInputAgent] PACMAN failed — falling back to EQeq")
                    charge_block = "ChargeMethod                  Ewald\nChargeEquilibration           EQeq"
                    use_cif = "no"
                    return {
                        "charge_block": charge_block,
                        "use_charges_from_cif": use_cif,
                        "charge_method": "eqeq",
                    }
            charge_block = (
                "ChargeMethod                  Ewald\n"
                "EwaldPrecision                {}".format(ewald_precision)
            )
            use_cif = "yes"
            if context is not None:
                context["charge_method_required"] = "pacman"
        elif charge_method == "ddec":
            if not framework_ok:
                from charge.ddec6 import run_ddec6_on_cif
                ddec_ok = run_ddec6_on_cif(cif_path, context or {})
                if ddec_ok:
                    framework_ok = True
                    use_cif = "yes"
                else:
                    print("[RASPAInputAgent] DFT-DDEC6 failed — falling back to EQeq")
                    charge_block = "ChargeMethod                  Ewald\nChargeEquilibration           EQeq"
                    use_cif = "no"
                    return {
                        "charge_block": charge_block,
                        "use_charges_from_cif": use_cif,
                        "charge_method": "eqeq",
                    }
            charge_block = (
                "ChargeMethod                  Ewald\n"
                "EwaldPrecision                {}".format(ewald_precision)
            )
            use_cif = "yes"
            if context is not None:
                context["charge_method_required"] = "ddec"
        else:
            charge_block = (
                "ChargeMethod                  Ewald\n"
                "EwaldPrecision                {}".format(ewald_precision)
            )

        return {
            "charge_block": charge_block,
            "use_charges_from_cif": use_cif,
            "charge_method": charge_method,
        }



    
    @staticmethod
    def _parse_cif_number(token: str) -> float:
        token = token.strip()

        
        if "(" in token:
            token = token.split("(", 1)[0]

        
        m = re.match(r"^[0-9+\-\.Ee]+", token)
        if m:
            token = m.group(0)

        return float(token)

    def _read_cell_from_cif(self, cif_path: Path) -> Tuple[float, float, float, float, float, float]:
        a = b = c = alpha = beta = gamma = None

        with open(cif_path, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue

                key = parts[0].lower()
                val = parts[1]

                if key.startswith("_cell_length_a"):
                    a = self._parse_cif_number(val)
                elif key.startswith("_cell_length_b"):
                    b = self._parse_cif_number(val)
                elif key.startswith("_cell_length_c"):
                    c = self._parse_cif_number(val)
                elif key.startswith("_cell_angle_alpha"):
                    alpha = self._parse_cif_number(val)
                elif key.startswith("_cell_angle_beta"):
                    beta = self._parse_cif_number(val)
                elif key.startswith("_cell_angle_gamma"):
                    gamma = self._parse_cif_number(val)

        if None in (a, b, c, alpha, beta, gamma):
            raise ValueError(f"Failed to read cell parameters from {cif_path}")

        return a, b, c, alpha, beta, gamma

    def _calculate_supercell_from_cif(self, cif_path: Path) -> Tuple[int, int, int]:
        from math import cos, sin, radians, sqrt, ceil

        a, b, c, alpha_deg, beta_deg, gamma_deg = self._read_cell_from_cif(cif_path)
        alpha = radians(alpha_deg)
        beta = radians(beta_deg)
        gamma = radians(gamma_deg)

        uc_volume = (
            a * b * c
            * sqrt(
                1
                - cos(alpha) ** 2
                - cos(beta) ** 2
                - cos(gamma) ** 2
                + 2 * cos(alpha) * cos(beta) * cos(gamma)
            )
        )

        cutoff = self.SUPERCELL_CUTOFF
        exp_x = ceil(cutoff * 2 / (uc_volume / (b * c * sin(alpha))))
        exp_y = ceil(cutoff * 2 / (uc_volume / (a * c * sin(beta))))
        exp_z = ceil(cutoff * 2 / (uc_volume / (a * b * sin(gamma))))

        return int(exp_x), int(exp_y), int(exp_z)
    
    def _infer_mixture_spec_with_llm(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if self.llm is None:
            
            g = (context.get("guest") or "methane").strip()
            return {"components": [{"guest": g, "mol_fraction": 1.0}]}

        guest_raw = (context.get("guest") or "").strip()
        query_text = (context.get("query_text") or context.get("user_query") or "").strip()
        job_name = (context.get("job_name") or "").strip()
        prop = (context.get("property") or "").strip()

        system_msg = (
            "You extract gas mixture components and mol fractions for a RASPA GCMC adsorption simulation.\n"
            "Return ONLY JSON in the form:\n"
            "{\"components\": [{\"guest\": \"CO2\", \"mol_fraction\": 0.15}, ...]}\n\n"
            "Rules:\n"
            "- 'guest' must be a short chemical formula string (CO2, N2, CH4, H2O, H2, O2, Ar, etc.).\n"
            "- Normalize common names: carbon dioxide->CO2, nitrogen->N2, methane->CH4, water->H2O, hydrogen->H2.\n"
            "- If composition is given (e.g., 15/84/1 or 0.15/0.84/0.01), convert to mol fractions that sum to 1.\n"
            "- If ONLY species are given with no composition, assume equal mol fractions.\n"
            "- If a single species is requested, return one component with mol_fraction=1.0.\n"
            "- Do NOT output extra keys or explanations.\n"
            "- Ensure mol_fraction are numbers and sum to 1 within 1e-6 (renormalize if needed).\n"
        )

        user_msg = (
            f"PROPERTY: {prop}\n"
            f"JOB_NAME: {job_name}\n"
            f"GUEST_FIELD: {guest_raw}\n"
            f"USER_QUERY: {query_text}\n\n"
            "Return components with mol fractions."
        )

        set_llm_context("RASPAInputAgent", "molecule_components")
        resp = self.llm.invoke([SystemMessage(content=system_msg), HumanMessage(content=user_msg)])
        text = (resp.content or "").strip()
        if text.startswith("```"):
            text = "\n".join(text.splitlines()[1:-1]).strip()

        obj = json.loads(text)
        comps = obj.get("components", [])

        
        if not isinstance(comps, list) or len(comps) == 0:
            g = (guest_raw or "methane").strip()
            return {"components": [{"guest": g, "mol_fraction": 1.0}]}

        cleaned = []
        for c in comps:
            if not isinstance(c, dict):
                continue
            g = str(c.get("guest", "")).strip()
            try:
                y = float(c.get("mol_fraction", 0.0))
            except Exception:
                y = 0.0
            if g and y > 0:
                cleaned.append({"guest": g, "mol_fraction": y})

        if not cleaned:
            g = (guest_raw or "methane").strip()
            return {"components": [{"guest": g, "mol_fraction": 1.0}]}

        s = sum(x["mol_fraction"] for x in cleaned)
        if s <= 0:
            
            n = len(cleaned)
            for x in cleaned:
                x["mol_fraction"] = 1.0 / n
        else:
            for x in cleaned:
                x["mol_fraction"] /= s

        return {"components": cleaned}
        
    def _infer_two_guests_with_llm(self, context: Dict[str, Any]) -> List[str]:
        if self.llm is None:
            raise ValueError("LLM is required for selectivity guest splitting")

        query_text = (context.get("query_text") or context.get("user_query") or "").strip()
        guest_raw = (context.get("guest") or "").strip()

        system_msg = (
            "You extract TWO gas species for a binary mixture adsorption/selectivity simulation.\n"
            "Return ONLY JSON like {\"guests\": [\"CO2\", \"N2\"]}.\n"
            "Rules:\n"
            "- Must return exactly 2 strings.\n"
            "- Normalize common names to short formula: carbon dioxide->CO2, nitrogen->N2, methane->CH4, water->H2O, hydrogen->H2.\n"
            "- If input is like 'CO2/N2' or 'CO2, N2' split it.\n"
            "- No extra text."
        )

        prompt = f"""
    User query: {query_text}
    Guest field: {guest_raw}
    Extract the two gases.
    """
        set_llm_context("RASPAInputAgent", "guest_extraction")
        resp = self.llm.invoke([SystemMessage(content=system_msg), HumanMessage(content=prompt)])
        text = resp.content.strip()
        if text.startswith("```"):
            text = "\n".join(text.splitlines()[1:-1]).strip()
        obj = json.loads(text)
        guests = obj.get("guests", [])
        if not isinstance(guests, list) or len(guests) != 2:
            raise ValueError(f"LLM failed to return 2 guests: {guests}")
        return [str(guests[0]).strip(), str(guests[1]).strip()]

    

    def _infer_TP_from_query(self, context: Dict[str, Any]) -> Tuple[float, float]:
        default_T = 298.0
        default_P_bar = 1.0

        query_text = (
            context.get("user_query")
            or context.get("query_text")
            or ""
        ).strip()

        job_name = (context.get("job_name") or "").strip()

        augmented = query_text
        if job_name:
            augmented += f"\n\nJOB_NAME: {job_name}\n"

        if not query_text or self.llm is None:
            return default_T, default_P_bar

        system_msg = (
            "You are a strict information extraction engine for RASPA simulation conditions.\n"
            "Your job is to extract a SINGLE temperature (K) and a SINGLE pressure (bar) for THIS specific job.\n\n"

            "Return ONLY valid JSON with exactly two keys:\n"
            "  {\"T_K\": <number or null>, \"P_bar\": <number or null>}\n\n"

            "CRITICAL DISAMBIGUATION RULES:\n"
            "1) The input may contain both a USER QUERY and a JOB_NAME.\n"
            "2) If JOB_NAME contains an explicit pressure (e.g., '0.1bar', '1bar', '0p1bar'),\n"
            "   you MUST use THAT pressure for P_bar, even if the user query mentions multiple pressures.\n"
            "3) Only output ONE pressure value. Do NOT output ranges or lists.\n"
            "4) If the user query contains multiple pressures and JOB_NAME does NOT specify which one,\n"
            "   set P_bar to null (do NOT guess).\n\n"

            "UNIT CONVERSION RULES:\n"
            "- Temperature: if °C is given, convert to Kelvin (K = C + 273.15).\n"
            "- Pressure: convert atm, Pa, kPa, MPa to bar.\n"
            "  * 1 bar = 100000 Pa\n"
            "  * 1 kPa = 0.01 bar\n"
            "  * 1 MPa = 10 bar\n"
            "  * 1 atm = 1.01325 bar\n\n"

            "OUTPUT RULES:\n"
            "- Use numbers only (no strings).\n"
            "- If a value is not explicitly specified under the rules above, use null.\n"
            "- No extra keys, no explanations, no markdown."
        )

        human_msg = HumanMessage(
            content=(
                f"User query:\n\"\"\"{augmented}\"\"\"\n\n"
                "Extract T_K and P_bar from this query."
            )
        )

        try:
            set_llm_context("RASPAInputAgent", "temperature_pressure")
            resp = self.llm.invoke([
                SystemMessage(content=system_msg),
                human_msg,
            ])
            text = resp.content.strip()
            if text.startswith("```"):
                text = "\n".join(text.splitlines()[1:-1]).strip()
            obj = json.loads(text)

            T = obj.get("T_K", None)
            P_bar = obj.get("P_bar", None)

            try:
                T_val = float(T) if T is not None else default_T
            except Exception:
                T_val = default_T

            try:
                P_val = float(P_bar) if P_bar is not None else default_P_bar
            except Exception:
                P_val = default_P_bar

            try:
                log_llm_decision("RASPAInputAgent", "temperature_pressure",
                                 {"T_K": T_val, "P_bar": P_val}, context)
            except Exception:
                pass
            return T_val, P_val

        except Exception as e:
            print(f"[RASPAInputAgent] _infer_TP_from_query LLM/parsing failed: {e}")
            return default_T, default_P_bar

    def _get_raspa_rag_hints(self, context: Dict[str, Any], top_files: int = 10) -> Dict[str, str]:
        if (
            os.getenv("SIMMOF_DISABLE_LITERATURE_RAG", "").strip().lower() in {"1", "true", "yes", "on"}
            or os.getenv("SIMMOF_RASPA_MODEL_RAG", "1").strip().lower() in {"0", "false", "no", "off"}
        ):
            print("[RAG] RASPA model/literature hints disabled by environment")
            return {"forcefield_hints": "", "molecule_hints": "", "charge_hints": ""}

        cached = context.get("raspa_rag_hints")
        if isinstance(cached, dict) and ("forcefield_hints" in cached or "molecule_hints" in cached):
            return {
                "forcefield_hints": (cached.get("forcefield_hints") or "").strip(),
                "molecule_hints": (cached.get("molecule_hints") or "").strip(),
                "charge_hints": (cached.get("charge_hints") or "").strip(),
            }

        out = {"forcefield_hints": "", "molecule_hints": "", "charge_hints": ""}

        try:
            from rag.agent import RagAgent

            rag_ctx = {
                "job_name": context.get("job_name") or "",
                "mof": context.get("mof") or "",
                "guest": context.get("guest") or "",
                "property": context.get("property") or "",
                "query_text": context.get("user_query") or context.get("query_text") or "",
            }

            agent = RagAgent(agent_name="RagAgent")
            r = agent.run_for_raspa_models(rag_ctx, top_files=top_files)

            out["forcefield_hints"] = (r.get("forcefield_hints") or "").strip()
            out["molecule_hints"] = (r.get("molecule_hints") or "").strip()
            out["charge_hints"] = (r.get("charge_hints") or "").strip()

            if out["forcefield_hints"] or out["molecule_hints"] or out["charge_hints"]:
                print("[RAG] RASPA model hints enabled")
            else:
                print("[RAG] no relevant RASPA model hints")


            context["raspa_rag_hints"] = out

        except Exception as e:
            print(f"[RAG] RASPA hints disabled due to error: {e}")

        return out

    def _get_raspa_manual_hints(self, context: Dict[str, Any], top_hits: int = 8) -> str:
        if os.getenv("SIMMOF_RASPA_MANUAL_RAG", "1").strip().lower() in {"0", "false", "no", "off"}:
            print("[RAG] RASPA manual hints disabled by environment")
            return ""

        cached = context.get("raspa_manual_hints")
        if isinstance(cached, str) and cached.strip():
            return cached.strip()

        query_text = " ".join(
            str(x)
            for x in [
                context.get("user_query") or context.get("query_text") or "",
                context.get("property") or "",
                context.get("guest") or "",
                "RASPA simulation.input manual keywords examples",
            ]
            if x
        )

        try:
            from input.raspa.manual_rag import retrieve_raspa_manual_hints

            out = retrieve_raspa_manual_hints(
                query_text,
                top_keywords=max(40, top_hits),
                top_sections=8,
                top_examples=8,
                max_chars_per_hit=1400,
            )
            hints = self._select_raspa_manual_evidence(query_text, out)
            if hints:
                print("[RAG] RASPA selected manual hints enabled")
            else:
                print("[RAG] no relevant RASPA manual hints")
            context["raspa_manual_hints"] = hints
            if hasattr(self, "_last_raspa_manual_selector"):
                context["raspa_manual_selector"] = getattr(self, "_last_raspa_manual_selector")
            context.setdefault("results", {})["raspa_manual_hints"] = {
                "query": out.get("query"),
                "keyword_hits": [
                    {
                        "keyword": h.get("keyword"),
                        "section_number": h.get("section_number"),
                        "score": h.get("score"),
                    }
                    for h in out.get("keyword_hits", [])[:top_hits]
                ],
                "selector": context.get("raspa_manual_selector"),
            }
            return hints
        except Exception as e:
            print(f"[RAG] RASPA manual hints disabled due to error: {e}")
            return ""

    def _select_raspa_manual_evidence(self, query_text: str, retrieval: Dict[str, Any]) -> str:
        fallback = (retrieval.get("formatted_hints") or "").strip()
        if self.llm is None:
            return fallback

        def compact(text: str, max_chars: int) -> str:
            text = re.sub(r"\s+", " ", text or "").strip()
            if len(text) <= max_chars:
                return text
            return text[: max_chars - 3].rstrip() + "..."

        candidates: List[Dict[str, Any]] = []
        for idx, item in enumerate(retrieval.get("keyword_hits", []), 1):
            candidates.append(
                {
                    "id": f"K{idx}",
                    "type": "manual_keyword",
                    "title": item.get("keyword"),
                    "score": item.get("score"),
                    "text": item.get("raw_text", ""),
                }
            )
        for idx, item in enumerate(retrieval.get("example_hits", []), 1):
            candidates.append(
                {
                    "id": f"E{idx}",
                    "type": "manual_example",
                    "title": item.get("example_title"),
                    "score": item.get("score"),
                    "text": item.get("raw_text", ""),
                }
            )
        for idx, item in enumerate(retrieval.get("section_hits", []), 1):
            candidates.append(
                {
                    "id": f"S{idx}",
                    "type": "manual_section",
                    "title": f"{item.get('section_number')} {item.get('section_title')}",
                    "score": item.get("score"),
                    "text": item.get("raw_text", ""),
                }
            )
        if not candidates:
            return fallback

        candidate_lines: List[str] = []
        for item in candidates:
            candidate_lines.extend(
                [
                    f"[{item['id']}] type={item['type']} title={item.get('title')} score={item.get('score')}",
                    compact(item.get("text", ""), 1800),
                    "",
                ]
            )

        system = (
            "You are an evidence-selection agent for RASPA 2.0 input generation.\n"
            "Read the user query and candidate manual snippets. Select only snippets that are relevant to the requested calculation objective.\n"
            "Do not select snippets merely because they share generic words such as simulation, adsorption, energy, or histogram.\n"
            "Distinguish property-specific evidence from runnable input scaffolds.\n"
            "Examples may use different molecules or frameworks; select them only for syntax patterns and never as replacements for user conditions.\n"
            "Return strict JSON only."
        )
        human = "\n".join(
            [
                f"USER_QUERY:\n{query_text}",
                "",
                "CANDIDATE_EVIDENCE:",
                "\n".join(candidate_lines)[:24000],
                "",
                "Return JSON with this schema:",
                '{',
                '  "selected_evidence_ids": ["K1", "E2"],',
                '  "rejected_evidence_ids": ["S1"],',
                '  "rationale": "Briefly explain why selected snippets support the requested RASPA input."',
                '}',
                "",
                "Selection rules:",
                "- Prefer manual keyword/example evidence that directly supports the requested output property or simulation type.",
                "- If a candidate is about a different histogram/property, reject it.",
                "- If an example uses a different molecule or framework, say it is syntax-only in the rationale.",
                "- Keep the selected set compact enough to guide generation.",
            ]
        )

        try:
            set_llm_context("RASPAInputAgent", "manual_rag_evidence_selector")
            resp = self.llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
            text = (resp.content or "").strip()
            if text.startswith("```"):
                lines = text.splitlines()
                if lines and lines[0].lstrip().startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                text = "\n".join(lines).strip()
            try:
                selection = json.loads(text)
            except Exception:
                match = re.search(r"\{.*\}", text, flags=re.S)
                selection = json.loads(match.group(0)) if match else {"selected_evidence_ids": [], "rationale": text[:1000]}
        except Exception as exc:
            self._last_raspa_manual_selector = {"error": repr(exc)}
            return fallback

        candidate_by_id = {item["id"]: item for item in candidates}
        selected_ids = [
            str(x)
            for x in selection.get("selected_evidence_ids", [])
            if str(x) in candidate_by_id
        ]
        selected = [candidate_by_id[x] for x in selected_ids]
        if not selected:
            self._last_raspa_manual_selector = {"selection": selection, "selected_ids": [], "fallback": True}
            return fallback

        lines = ["[LLM-selected RASPA manual evidence]"]
        for item in selected:
            lines.extend(
                [
                    f"- [{item['id']}] {item['type']}: {item.get('title')}",
                    compact(item.get("text", ""), 1800),
                ]
            )
        if selection.get("rationale"):
            lines.extend(["", "[Evidence selector rationale]", compact(str(selection.get("rationale")), 1200)])
        self._last_raspa_manual_selector = {
            "selection": selection,
            "selected_ids": selected_ids,
            "fallback": False,
        }
        return "\n".join(lines).strip()

    def _apply_raspa_manual_review(
        self,
        input_text: str,
        context: Dict[str, Any],
        manual_hints: str,
    ) -> str:
        if not manual_hints.strip() or self.llm is None:
            return input_text

        try:
            set_llm_context("RASPAInputAgent", "manual_rag_input_review")
            resp = self.llm.invoke([
                SystemMessage(content=RASPA_MANUAL_REVIEW_SYSTEM),
                HumanMessage(content=RASPA_MANUAL_REVIEW_USER.format(
                    query_text=context.get("user_query") or context.get("query_text") or "",
                    property_name=context.get("property") or "",
                    manual_hints=manual_hints[:8000],
                    input_text=input_text,
                )),
            ])
            reviewed = (resp.content or "").strip()
            if reviewed.startswith("```"):
                lines = reviewed.splitlines()
                if lines and lines[0].lstrip().startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                reviewed = "\n".join(lines).strip()
            if not reviewed:
                return input_text
            return reviewed
        except Exception as e:
            print(f"[RAG] RASPA manual review skipped due to error: {e}")
            return input_text
        
    

    GENERAL_FORCEFIELDS = ("UFF", "UFF4MOF", "DREIDING")
    FORCEFIELD_DESCRIPTIONS = """Implemented force-field descriptions:
- UFF: Universal Force Field, a broad-coverage generic force field for molecular and crystalline systems across much of the periodic table.
- UFF4MOF: MOF-oriented extension of UFF with atom typing and parameters adapted for metal-organic framework environments.
- DREIDING: generic force field based on simple atom typing and hybridization rules, commonly used as a transferable option for organic and framework atoms. NOTE: DREIDING lacks parameters for metal centers; when the MOF contains metal nodes, use DREIDING as the base with UFF or UFF4MOF overrides for the metal elements (mixed type).
"""

    def _list_forcefields(self) -> List[str]:
        if not self.forcefield_dir.exists():
            return list(self.GENERAL_FORCEFIELDS)

        available = {p.name for p in self.forcefield_dir.iterdir() if p.is_dir()}
        return sorted(ff for ff in self.GENERAL_FORCEFIELDS if ff in available)

    def _choose_forcefield_with_llm(
        self,
        context: Dict[str, Any],
        rag_hints: str = "",
        si_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        available = self.available_forcefields
        default_ff = "UFF" if "UFF" in available else (available[0] if available else "UFF")

        if self.llm is None:
            return {"type": "single", "forcefield": default_ff}

        has_si = bool(si_chunks)
        si_option_text = ""
        if has_si:
            si_preview = "\n".join(
                f"  [{i+1}] {c['filename']} (score={c['score']:.3f}):\n    {c['text'][:200]}"
                for i, c in enumerate(si_chunks)
            )
            si_option_text = (
                "\n\nLIT_FF_CHUNKS (excerpts from literature — papers or supporting information):\n"
                f"{si_preview}"
            )

        custom_ff_format = (
            "  Custom FF: {\"type\": \"custom_ff\"}\n" if has_si else ""
        )
        custom_ff_rule = (
            "  custom_ff — if LIT_FF_CHUNKS contain LJ parameter tables OR descriptions of "
            "fitted/optimized ε and σ values for atoms relevant to this MOF/guest system, "
            "prefer this to extract and use those parameters instead of a generic force field.\n"
            if has_si else ""
        )

        system_msg = (
            "You are selecting a RASPA forcefield for a MOF adsorption simulation.\n"
            "\n"
            f"Choose from these options:\n"
            f"  single    — one FF for all framework atoms (based on RAG_HINTS)\n"
            f"  mixed     — different FF for metal vs organic atoms (based on RAG_HINTS)\n"
            f"{custom_ff_rule}"
            "\n"
            "Mixed scheme: use one base FF for organic framework atoms (C, H, N, O, S, F, Cl, P) "
            "and a different FF for metal centers (e.g. Zr, Al, Fe, Cu, Zn, Co, Ni, Cr, Mn, Ti, V).\n"
            "Use mixed ONLY when the RAG hints or MOF chemistry clearly support it.\n"
            "First use the user query and RAG hints. If these are not actionable, use the "
            "force-field descriptions as conservative internal-library guidance rather than "
            "literature evidence.\n"
            "\n"
            "Return ONLY JSON in one of these formats:\n"
            "  Single:    {\"type\": \"single\",    \"forcefield\": \"<name from allowed list>\"}\n"
            "  Mixed:     {\"type\": \"mixed\",     \"forcefield\": \"<base FF name>\", "
            "\"overrides\": {\"<element>\": \"<FF name>\", ...}}\n"
            f"{custom_ff_format}"
            "Both 'forcefield' and all override values must come from the allowed list.\n"
            "'overrides' maps element symbols to their FF — include only elements that differ from the base."
        )

        prompt = f"""
Allowed forcefields: {available}

{self.FORCEFIELD_DESCRIPTIONS}

MOF: {context.get('mof')}
Guest: {context.get('guest')}
Property: {context.get('property')}
User query: {context.get('user_query') or context.get('query_text', '')}

RAG_HINTS (for single/mixed decision; may be irrelevant):
{rag_hints}
{si_option_text}"""

        try:
            set_llm_context("RASPAInputAgent", "forcefield_selection")
            resp = self.llm.invoke([
                SystemMessage(content=system_msg),
                HumanMessage(content=prompt),
            ])
            text = resp.content.strip()
            if text.startswith("```"):
                text = "\n".join(text.splitlines()[1:-1]).strip()
            obj = json.loads(text)

            ff_type = str(obj.get("type", "single")).strip()

            if ff_type == "custom_ff":
                result = {"type": "custom_ff", "forcefield": None}
                print("[RASPAInputAgent] Custom FF selected")
            else:
                base_ff = str(obj.get("forcefield", "")).strip()
                if base_ff not in available:
                    raise ValueError(f"base FF '{base_ff}' not in available list")

                if ff_type == "mixed":
                    overrides = obj.get("overrides", {})
                    if not isinstance(overrides, dict):
                        overrides = {}
                    validated = {
                        elem: ff for elem, ff in overrides.items()
                        if isinstance(ff, str) and ff in available
                    }
                    result = {"type": "mixed", "forcefield": base_ff, "overrides": validated}
                    print(f"[RASPAInputAgent] Mixed FF selected: base={base_ff}, overrides={validated}")
                else:
                    result = {"type": "single", "forcefield": base_ff}
                    print(f"[RASPAInputAgent] Single FF selected: {base_ff}")

            from config import ask_user_confirmation

            def _reinvoke_ff(instruction: str) -> str:
                revised = prompt + f"\n\nUser instruction: {instruction}\nRevise your forcefield selection accordingly."
                set_llm_context("RASPAInputAgent", "forcefield_selection_revision")
                r = self.llm.invoke([SystemMessage(content=system_msg), HumanMessage(content=revised)])
                return r.content.strip()

            action, revised_text = ask_user_confirmation(
                "RASPAInputAgent",
                f"Proposed forcefield: {json.dumps(result)}",
                reinvoke_fn=_reinvoke_ff,
                required=True,
            )
            if action == "apply" and revised_text != f"Proposed forcefield: {json.dumps(result)}":
                try:
                    t = revised_text.strip()
                    if t.startswith("```"):
                        t = "\n".join(t.splitlines()[1:-1]).strip()
                    obj2 = json.loads(t)
                    ff_type2 = str(obj2.get("type", "single")).strip()
                    if ff_type2 == "custom_ff":
                        result = {"type": "custom_ff", "forcefield": None}
                    else:
                        base2 = str(obj2.get("forcefield", base_ff)).strip()
                        if base2 in available:
                            if ff_type2 == "mixed":
                                ov2 = {e: f for e, f in obj2.get("overrides", {}).items() if f in available}
                                result = {"type": "mixed", "forcefield": base2, "overrides": ov2}
                            else:
                                result = {"type": "single", "forcefield": base2}
                    print(f"[RASPAInputAgent] FF updated per user instruction: {result}")
                except Exception:
                    pass
            try:
                log_llm_decision("RASPAInputAgent", "forcefield_selection", result, context)
            except Exception:
                pass
            return result

        except Exception as e:
            print(f"[RASPAInputAgent] forcefield JSON parse failed: {e}")

        return {"type": "single", "forcefield": default_ff}

    def _parse_pseudo_atoms_file(self, ff_name: str) -> Dict[str, Dict]:
        path = self.forcefield_dir / ff_name / "pseudo_atoms.def"
        if not path.exists():
            return {}
        result = {}
        in_data = False
        with open(path) as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                if not in_data:
                    try:
                        int(parts[0])
                        in_data = True
                    except ValueError:
                        pass
                    continue
                if len(parts) >= 3:
                    result[parts[0]] = {"line": line.rstrip("\n"), "element": parts[2]}
        return result

    _MIXING_RULE_KEYWORDS = {"Lorentz-Berthelot", "Jorgensen", "WaldmanHagler"}

    def _parse_mixing_rules_file(self, ff_name: str):
        path = self.forcefield_dir / ff_name / "force_field_mixing_rules.def"
        if not path.exists():
            return [], {}, ""
        header_lines = []
        rules = {}
        mixing_rule = ""
        in_data = False
        with open(path) as f:
            for line in f:
                stripped = line.strip()
                if not in_data:
                    if stripped.startswith("#") or not stripped:
                        header_lines.append(line.rstrip("\n"))
                        continue
                    parts = stripped.split()
                    try:
                        int(parts[0])
                        in_data = True
                        continue
                    except ValueError:
                        header_lines.append(line.rstrip("\n"))
                        continue
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                if parts:
                    if parts[0] in self._MIXING_RULE_KEYWORDS:
                        mixing_rule = parts[0]
                    else:
                        rules[parts[0]] = line.rstrip("\n")
        return header_lines, rules, mixing_rule


    _ATOMIC_MASSES: Dict[str, float] = {
        "H": 1.008, "He": 4.003, "Li": 6.941, "Be": 9.012, "B": 10.811,
        "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.18,
        "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.086, "P": 30.974,
        "S": 32.065, "Cl": 35.453, "Ar": 39.948, "K": 39.098, "Ca": 40.078,
        "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938,
        "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38,
        "Ga": 69.723, "Ge": 72.63, "As": 74.922, "Se": 78.96, "Br": 79.904,
        "Kr": 83.798, "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224,
        "Mo": 95.96, "Ru": 101.07, "Rh": 102.906, "Pd": 106.42, "Ag": 107.868,
        "Cd": 112.411, "In": 114.818, "Sn": 118.71, "Sb": 121.76, "Te": 127.60,
        "I": 126.904, "Xe": 131.293, "Cs": 132.905, "Ba": 137.327, "La": 138.905,
        "Ce": 140.116, "Pr": 140.908, "Nd": 144.242, "Sm": 150.36, "Eu": 151.964,
        "Gd": 157.25, "Tb": 158.925, "Dy": 162.50, "Ho": 164.930, "Er": 167.259,
        "Tm": 168.934, "Yb": 173.054, "Lu": 174.967, "Hf": 178.49, "Ta": 180.948,
        "W": 183.84, "Re": 186.207, "Os": 190.23, "Ir": 192.217, "Pt": 195.084,
        "Au": 196.967, "Hg": 200.59, "Tl": 204.383, "Pb": 207.2, "Bi": 208.980,
        "U": 238.029,
    }

    def _search_si_ff(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        mof   = context.get("mof", "")
        guest = context.get("guest", "")
        prop  = context.get("property", "")
        query = context.get("query_text") or f"LJ parameters {mof} {guest} force field sigma epsilon"
        try:
            from rag.agent import RagAgent
            agent = RagAgent()
            chunks = agent.get_si_ff_chunks(query, top_k=5)
            print(f"[RASPAInputAgent] SI FF search: {len(chunks)} chunks found")
            return chunks
        except Exception as e:
            print(f"[RASPAInputAgent] SI FF search failed: {e}")
            return []

    def _extract_ff_params_from_chunks(self, chunks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not chunks:
            return None

        combined = "\n\n---\n\n".join(
            f"[Source: {c['filename']}]\n{c['text']}" for c in chunks
        )

        system_msg = (
            "You are a force-field parameter extractor for RASPA MOF simulations.\n"
            "Extract ALL Lennard-Jones parameters from the text below.\n"
            "Return ONLY valid JSON with this structure:\n"
            "{\n"
            "  \"ff_name\": \"<short identifier, e.g. MgMOF74_custom>\",\n"
            "  \"source\": \"<source filename>\",\n"
            "  \"mixing_rule\": \"Lorentz-Berthelot\",\n"
            "  \"atom_types\": [\n"
            "    {\"name\": \"<atom_type>\", \"element\": \"<element_symbol>\",\n"
            "     \"sigma\": <float_Angstrom>, \"epsilon\": <float_K>,\n"
            "     \"charge\": <float_e>, \"mass\": <float_amu>},\n"
            "    ...\n"
            "  ]\n"
            "}\n"
            "Rules:\n"
            "- sigma in Angstrom, epsilon in K (ε/kB), charge in e, mass in amu.\n"
            "- element: infer from atom_type name (e.g. OA→O, CA→C, Mg→Mg, Zn→Zn).\n"
            "- mass: use standard atomic mass for the element.\n"
            "- If epsilon is in kcal/mol, convert to K (multiply by 503.0).\n"
            "- If sigma is in nm, convert to Angstrom (multiply by 10).\n"
            "- Include ALL atom types found. If no clear LJ table found, return null."
        )

        try:
            set_llm_context("RASPAInputAgent", "si_ff_params")
            resp = self.llm.invoke([
                SystemMessage(content=system_msg),
                HumanMessage(content=combined),
            ])
            text = resp.content.strip()
            if text.lower() == "null":
                return None
            if text.startswith("```"):
                text = "\n".join(text.splitlines()[1:-1]).strip()
            obj = json.loads(text)
            if isinstance(obj, list):
                obj = {"ff_name": "custom_ff",
                       "source": chunks[0]["filename"] if chunks else "",
                       "mixing_rule": "Lorentz-Berthelot",
                       "atom_types": obj}
            if not obj or not isinstance(obj, dict) or not obj.get("atom_types"):
                return None

            for at in obj["atom_types"]:
                elem = at.get("element", "")
                if not at.get("mass") and elem in self._ATOMIC_MASSES:
                    at["mass"] = self._ATOMIC_MASSES[elem]

            print(f"[RASPAInputAgent] Extracted {len(obj['atom_types'])} atom types "
                  f"from SI (source: {obj.get('source', '?')})")
            return obj

        except Exception as e:
            print(f"[RASPAInputAgent] FF param extraction failed: {e}")
            return None

    def _build_custom_raspa_ff(self, ff_params: Dict[str, Any]) -> str:
        ff_name = ff_params.get("ff_name", "custom_ff").replace(" ", "_")
        ff_dir  = self.forcefield_dir / ff_name
        ff_dir.mkdir(parents=True, exist_ok=True)

        atom_types = ff_params["atom_types"]

        pa_lines = [
            "#number of pseudo atoms",
            str(len(atom_types)),
            "#type      print   as  scatt oxidation\tmass       charge  polarization\tB-factor radii  connectivity\tanisotropic\tanisotropic-type\ttinker-type",
        ]
        for at in atom_types:
            name    = at["name"]
            element = at.get("element", name[:2].rstrip("0123456789"))
            mass    = at.get("mass") or self._ATOMIC_MASSES.get(element, 1.0)
            charge  = at.get("charge", 0.0)
            pa_lines.append(
                f"{name}\tyes\t{element}\t{element}\t0\t{mass}\t{charge}\t"
                f"0.0\t1.0\t1.0\t0.0\t0.0\tabsolute\t0.0"
            )
        (ff_dir / "pseudo_atoms.def").write_text("\n".join(pa_lines) + "\n")

        mr_lines = [
            "# general rule for shifted vs truncated",
            "truncated",
            "# general rule tailcorrections",
            "no",
            "# number of defined interactions",
            str(len(atom_types)),
            "# type interaction",
        ]
        for at in atom_types:
            eps   = at.get("epsilon", 0.0)
            sigma = at.get("sigma",   0.0)
            mr_lines.append(f"{at['name']}\tlennard-jones\t{eps}\t{sigma}")
        (ff_dir / "force_field_mixing_rules.def").write_text("\n".join(mr_lines) + "\n")

        (ff_dir / "force_field.def").write_text(
            "# rules to overwrite\n0\n"
            "# number of defined interactions\n0\n"
            "# mixing rules to overwrite\n0\n"
        )

        print(f"[RASPAInputAgent] Custom FF '{ff_name}' written to {ff_dir}")
        return ff_name

    def _create_mixed_ff(self, base_ff: str, overrides: Dict[str, str]) -> str:
        all_override_ffs = set(overrides.values())

        base_atoms = self._parse_pseudo_atoms_file(base_ff)
        base_header, base_rules, base_mixing_rule = self._parse_mixing_rules_file(base_ff)

        override_atoms: Dict[str, Dict[str, Dict]] = {}
        override_rules: Dict[str, Dict[str, str]] = {}
        for ff in all_override_ffs:
            override_atoms[ff] = self._parse_pseudo_atoms_file(ff)
            _, override_rules[ff], _ = self._parse_mixing_rules_file(ff)

        override_elem_to_types: Dict[str, Dict[str, str]] = {}
        for elem, ff in overrides.items():
            override_elem_to_types[elem] = {
                tname: info["line"]
                for tname, info in override_atoms[ff].items()
                if info["element"] == elem
            }

        override_elements = set(overrides.keys())

        merged_pa: Dict[str, str] = {}
        for tname, info in base_atoms.items():
            if info["element"] not in override_elements:
                merged_pa[tname] = info["line"]
        for elem, types in override_elem_to_types.items():
            merged_pa.update(types)

        merged_mr: Dict[str, str] = {}
        for tname, line in base_rules.items():
            elem = base_atoms.get(tname, {}).get("element")
            if elem not in override_elements:
                merged_mr[tname] = line
        for elem, ff in overrides.items():
            ff_rules = override_rules.get(ff, {})
            for tname in override_elem_to_types.get(elem, {}):
                if tname in ff_rules:
                    merged_mr[tname] = ff_rules[tname]

        override_tag = "_".join(f"{e}{v}" for e, v in sorted(overrides.items()))
        new_ff_name = f"{base_ff}_{override_tag}_mixed"
        new_ff_dir = self.forcefield_dir / new_ff_name
        new_ff_dir.mkdir(parents=True, exist_ok=True)

        with open(new_ff_dir / "pseudo_atoms.def", "w") as f:
            f.write("#number of pseudo atoms\n")
            f.write(f"{len(merged_pa)}\n")
            f.write(
                "#type      print   as  scatt oxidation\t"
                "mass       charge  polarization\t"
                "B-factor radii  connectivity\tanisotropic\tanisotropic-type\ttinker-type\n"
            )
            for line in merged_pa.values():
                f.write(line + "\n")

        with open(new_ff_dir / "force_field_mixing_rules.def", "w") as f:
            for line in base_header:
                f.write(line + "\n")
            f.write(f"{len(merged_mr)}\n# type interaction\n")
            for line in merged_mr.values():
                f.write(line + "\n")
            if base_mixing_rule:
                f.write(f"# general mixing rule for Lennard-Jones\n{base_mixing_rule}\n")

        with open(new_ff_dir / "force_field.def", "w") as f:
            f.write(
                "# rules to overwrite\n0\n"
                "# pair\ttruncated/shifted tailcorrections\n"
                "# number of defined interactions\n0\n"
                "# type      type2       interaction\n"
                "# mixing rules to overwrite\n0\n"
            )

        print(
            f"[RASPAInputAgent] Mixed FF '{new_ff_name}' created "
            f"(base={base_ff}, overrides={overrides}, "
            f"types={len(merged_pa)}, rules={len(merged_mr)})"
        )
        return new_ff_name


    def _build_molecule_family_index(self) -> Dict[str, List[str]]:
        index: Dict[str, List[str]] = {}
        if not self.molecules_dir.exists():
            return index

        for family_dir in self.molecules_dir.iterdir():
            if not family_dir.is_dir():
                continue
            family = family_dir.name
            names = sorted([f.stem for f in family_dir.glob("*.def")])
            if names:
                index[family] = names
        return index
    
    def _guest_aliases(self, guest_raw: str) -> list[str]:
        g = (guest_raw or "").strip()
        aliases = {g}

        
        if g.upper() == "H2" or g.lower() == "hydrogen":
            aliases |= {"hydrogen", "H2", "h2"}

        
        if g.upper() == "CO2" or g.lower() in ("carbon dioxide", "co2"):
            aliases |= {"CO2", "co2", "carbon_dioxide"}

        
        return list(aliases)

    def _choose_molecule_definition_with_llm(self, context: Dict[str, Any], rag_hints: str = "") -> str:
        guest = context.get("guest") or ""
        query_text = (context.get("query_text") or context.get("user_query") or "").strip()
        job_name = (context.get("job_name") or "").strip()
        aliases = set(self._guest_aliases(guest))

        matching_families = []
        for fam, names in self.molecule_families.items():
            if any(a in names for a in aliases):
                matching_families.append(fam)

        families = sorted(matching_families) if matching_families else sorted(self.molecule_families.keys())

        if not families:
            return "TraPPE"

        default_def = "TraPPE" if "TraPPE" in families else families[0]

        if self.llm is None:
            return default_def

        family_descriptions = {
            "TraPPE": "Transferable Potentials for Phase Equilibria molecule definitions for transferable adsorption and phase-equilibrium models.",
            "TraPPE-UA": "united-atom TraPPE molecule definitions, commonly used for transferable adsorbate models.",
            "TraPPE-EH": "extended TraPPE-style molecule definitions available in the local RASPA library.",
            "EPM2": "three-site carbon dioxide model family used for CO2 adsorption simulations.",
            "Generic": "generic local molecule definitions used when no more specific implemented family is supported by the query or RAG evidence.",
            "CastilloVlugtCalero2009": "literature-derived RASPA molecule definitions for aromatic hydrocarbon adsorbates.",
        }
        described_families = "\n".join(
            f"- {fam}: {family_descriptions[fam]}"
            for fam in families
            if fam in family_descriptions
        )
        if not described_families:
            described_families = "- No curated description is available for the listed families; rely on RAG_HINTS and explicit user requests."

        system_msg = (
            "You choose a RASPA MoleculeDefinition family (guest model directory) for the guest molecule.\n"
            "Return ONLY JSON like {\"definition\": \"EPM2\"}.\n\n"

            "Decision procedure:\n"
            "1) If the user query explicitly requests a guest model/family (e.g., EPM2, TraPPE, TraPPE-UA, SPC/E), "
            "choose that exact family if it exists in the allowed list.\n"
            "2) Normalize spelling variants: 'Trappe' or 'trappe' means 'TraPPE'.\n"
            "3) If multiple models are mentioned, choose the one that matches THIS job's name if it contains a model token "
            "(e.g., job name contains 'EPM2' or 'TraPPE'). If job name doesn't specify, choose the first model mentioned.\n"
            "4) Otherwise, inspect RAG_HINTS and choose the family best supported by retrieved literature for the guest and task.\n"
            "5) If RAG_HINTS are absent or not actionable, use the family descriptions as conservative internal-library guidance.\n"
            "6) You MUST return exactly one allowed family name.\n"
            "No extra text."
        )

        prompt = f"""
Allowed molecule definition families: {families}

Implemented molecule-definition family descriptions:
{described_families}

Guest (target molecule): {context.get('guest')}
JOB_NAME: {job_name}
USER_QUERY: {query_text}

Normalize common names: carbon dioxide -> CO2, nitrogen -> N2, methane -> CH4, hydrogen -> H2, water -> H2O.

RAG_HINTS (optional; may be irrelevant. Use only if clearly applicable):
{rag_hints}

Task:
- Choose exactly one MoleculeDefinition family from the allowed list.
- If the USER_QUERY explicitly requests a model/family (e.g., EPM2, TraPPE, TraPPE-UA), pick it if available.
- Normalize spelling: "Trappe"/"trappe" -> "TraPPE".
- If multiple models are mentioned, prefer the one indicated by JOB_NAME if present; otherwise choose the first mentioned.
- Output ONLY JSON: {{"definition": "<one of allowed families>"}}.

Return ONLY JSON.
"""

        try:
            set_llm_context("RASPAInputAgent", "molecule_definition")
            resp = self.llm.invoke([
                SystemMessage(content=system_msg),
                HumanMessage(content=prompt),
            ])
            text = resp.content.strip()
            if text.startswith("```"):
                text = "\n".join(text.splitlines()[1:-1]).strip()
            obj = json.loads(text)
            cand = str(obj.get("definition", "")).strip()
            result = cand if cand in families else default_def
            print(f"[RASPAInputAgent] Molecule definition selected: {result}")

            from config import ask_user_confirmation

            def _reinvoke_mol(instruction: str) -> str:
                revised = prompt + f"\n\nUser instruction: {instruction}\nRevise your molecule definition selection accordingly."
                set_llm_context("RASPAInputAgent", "molecule_definition_revision")
                r = self.llm.invoke([SystemMessage(content=system_msg), HumanMessage(content=revised)])
                t = r.content.strip()
                if t.startswith("```"):
                    t = "\n".join(t.splitlines()[1:-1]).strip()
                try:
                    c = str(json.loads(t).get("definition", "")).strip()
                    return c if c in families else result
                except Exception:
                    return result

            action, revised = ask_user_confirmation(
                "RASPAInputAgent",
                f"Proposed molecule definition: {result}",
                reinvoke_fn=_reinvoke_mol,
                required=True,
            )
            if action == "apply" and revised != f"Proposed molecule definition: {result}":
                if revised in families:
                    print(f"[RASPAInputAgent] Molecule definition updated: {revised}")
                    result = revised
            try:
                log_llm_decision("RASPAInputAgent", "molecule_definition",
                                 {"definition": result}, context)
            except Exception:
                pass
            return result
        except Exception as e:
            print("[RASPAInputAgent] molecule_definition JSON parse failed:", e)

        return default_def

    def _select_molecule_name(self, guest_raw: str, family: str) -> str:
        names = self.molecule_families.get(family, [])
        if not names:
            raise ValueError(f"No .def files found under molecules/{family}")

        system_msg = (
            "You are helping to choose a RASPA molecule name for a simulation.\n"
            "You are given a user-specified guest name and a list of valid molecule names\n"
            "(corresponding to existing '.def' files under a given family directory).\n\n"
            "Your task:\n"
            "- Choose the SINGLE best matching molecule name from the candidate list.\n"
            "- If none of the candidates is a reasonable match, return null.\n\n"
            "Output format:\n"
            "- Return ONLY a JSON object like: {\"name\": \"CO2\"}\n"
            "- Or, if there is no good match: {\"name\": null}\n"
            "Do NOT include any extra text or explanation."
        )

        prompt = f"""
User guest name: {guest_raw}

Candidate molecule names (from molecules/{family}/*.def):
{names}

Choose the best matching candidate name, or null if nothing matches.
"""

        try:
            set_llm_context("RASPAInputAgent", "molecule_name_match")
            resp = self.llm.invoke([
                SystemMessage(content=system_msg),
                HumanMessage(content=prompt),
            ])
            text = resp.content.strip()
            if text.startswith("```"):
                text = "\n".join(text.splitlines()[1:-1]).strip()

            obj = json.loads(text)
            cand = obj.get("name", None)

            if cand is None:
                raise ValueError(
                    f"[RASPAInputAgent] LLM returned null for guest '{guest_raw}' "
                    f"in family '{family}'. Candidates: {names}"
                )

            cand = str(cand).strip()
            if cand not in names:
                raise ValueError(
                    f"[RASPAInputAgent] LLM chose '{cand}' which is not in candidate names "
                    f"for family '{family}'. Candidates: {names}"
                )

            return cand

        except Exception as e:
            
            raise ValueError(
                f"[RASPAInputAgent] Failed to select molecule name for guest '{guest_raw}' "
                f"in family '{family}'. Error: {e}. Candidates: {names}"
            )


    
    def _build_component_blocks(self, components: List[Dict[str, Any]]) -> str:
        blocks: List[str] = []
        for i, comp in enumerate(components):
            blocks.append(
                f"""Component {i} MoleculeName      {comp['molecule_name']}
    MoleculeDefinition        {comp['molecule_definition']}
    IdealGasRosenbluthWeight  1.0
    TranslationProbability    1.0
    ReinsertionProbability    1.0
    RotationProbability       1.0
    RegrowProbability         3.0
    SwapProbability           4.0
    WidomProbability          1.0
    MolFraction               {comp['mol_fraction']}"""
            )
        return "\n\n".join(blocks)

    def _cleanup_raspa_input_text(self, text: str) -> str:
        text = (text or "").strip()
        if text.startswith("<<<") and text.endswith(">>>"):
            text = text[3:-3].strip()
        elif text.startswith("<<<"):
            text = text[3:].strip()
        if text.endswith(">>>"):
            text = text[:-3].strip()
        text = re.sub(r"^(\s*)MoleculeDefinitions\b", r"\1MoleculeDefinition", text, flags=re.MULTILINE)
        return text.rstrip() + "\n"

    def _build_params(self, context: Dict[str, Any], include_guest: bool = True) -> Dict[str, Any]:
        
        cif_path = Path(context["mof_path"])
        framework_name = context.get("mof") or cif_path.stem
        context["mof"] = framework_name

        ux, uy, uz = self._calculate_supercell_from_cif(cif_path)

        
        T_K, P_bar = self._infer_TP_from_query(context)
        if "pressure_pa" in context and context["pressure_pa"] is not None:
            P_pa = float(context["pressure_pa"])
            P_bar = P_pa / 1e5
        elif "pressure_bar" in context and context["pressure_bar"] is not None:
            P_bar = float(context["pressure_bar"])
            P_pa = P_bar * 1e5
        else:
            P_pa = P_bar * 1e5

        rag = self._get_raspa_rag_hints(context, top_files=10)
        ff_hints = rag.get("forcefield_hints", "")
        mol_hints = rag.get("molecule_hints", "")

        si_chunks = self._search_si_ff(context)

        ff_choice = self._choose_forcefield_with_llm(
            context, rag_hints=ff_hints, si_chunks=si_chunks if si_chunks else None
        )
        if ff_choice["type"] == "custom_ff":
            ff_params = self._extract_ff_params_from_chunks(si_chunks)
            if ff_params:
                ff_name = self._build_custom_raspa_ff(ff_params)
                ff_choice["forcefield"] = ff_name
                print(f"[RASPAInputAgent] Custom SI FF built: {ff_name}")
            else:
                print("[RASPAInputAgent] Custom SI FF extraction failed — falling back to UFF")
                default_ff = "UFF" if "UFF" in self.available_forcefields else self.available_forcefields[0]
                ff_choice = {"type": "single", "forcefield": default_ff}
                ff_name = default_ff
        elif ff_choice["type"] == "mixed":
            ff_name = self._create_mixed_ff(
                base_ff=ff_choice["forcefield"],
                overrides=ff_choice["overrides"],
            )
            ff_choice["forcefield"] = ff_name
        else:
            ff_name = ff_choice["forcefield"]
        context["ff_choice"] = ff_choice

        params = {
            "forcefield": ff_name,
            "framework_name": framework_name,
            "unitcell_x": ux,
            "unitcell_y": uy,
            "unitcell_z": uz,
            "temperature": T_K,
            "pressure_pa": P_pa,
            "pressure_bar": P_bar,
        }

        
        if include_guest:
            guest = context.get("guest") or "methane"
            ctx_guest = {**context, "guest": guest}
            molecule_def = self._choose_molecule_definition_with_llm(ctx_guest, rag_hints=mol_hints)
            molecule_name = self._select_molecule_name(guest, molecule_def)
            params.update({
                "molecule_definition": molecule_def,
                "molecule_name": molecule_name,
            })

            params.update(self._decide_charge_settings(
                cif_path=cif_path,
                forcefield=ff_name,
                guests=[(molecule_def, molecule_name)],
                cutoff=self.SUPERCELL_CUTOFF,
                context=context,
            ))

        else:
            params.update(self._decide_charge_settings(
                cif_path=cif_path,
                forcefield=ff_name,
                guests=None,
                cutoff=self.SUPERCELL_CUTOFF,
                context=context,
            ))


        return params

    def _generate_simulation_input_with_llm(
        self,
        params: dict,
        context: dict,
        rag_hints: str = "",
        manual_hints: str = "",
    ) -> str:
        filled_template = render_raspa_format(params)

        query = {
            "property": context.get("property"),
            "mof": params.get("framework_name"),
            "guest": context.get("guest"),
            "temperature_K": params.get("temperature"),
            "pressure_bar": params.get("pressure_bar"),
            "forcefield": params.get("forcefield"),
            "unitcells": f"{params.get('unitcell_x')} {params.get('unitcell_y')} {params.get('unitcell_z')}",
            "use_charges_from_cif": params.get("use_charges_from_cif"),
        }
        for key in (
            "molecule_name", "molecule_definition",
            "molecule_name_0", "molecule_definition_0", "mol_fraction_0",
            "molecule_name_1", "molecule_definition_1", "mol_fraction_1",
            "components",
        ):
            if params.get(key) is not None:
                query[key] = params[key]
        query = {k: v for k, v in query.items() if v is not None}

        prompt = create_raspa_input_prompt(
            query=query,
            filled_template=filled_template,
            params=params,
            rag_hints=rag_hints,
            manual_hints=manual_hints,
        )

        set_llm_context("RASPAInputAgent", "input_generation")
        resp = self.llm.invoke([
            SystemMessage(content=get_raspa_system_message()),
            HumanMessage(content=prompt),
        ])
        text = (resp.content or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1]).strip()
        return text

    def _llm_patch_raspa_input(self, original_text: str, replacements: Dict[str, Any]) -> str:
        if self.llm is None:
            raise ValueError("LLM is required for RASPA reproduce patching (self.llm is None).")

        rep_json = json.dumps(replacements, ensure_ascii=False, indent=2)

        set_llm_context("RASPAInputAgent", "input_patch")
        resp = self.llm.invoke([
            SystemMessage(content=RASPA_REPRO_PATCH_SYSTEM),
            HumanMessage(content=RASPA_REPRO_PATCH_USER.format(
                replacements_json=rep_json,
                original_text=original_text
            )),
        ])

        out = (resp.content or "").strip()

        
        if out.startswith("```"):
            lines = out.splitlines()
            if lines and lines[0].lstrip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            out = "\n".join(lines).strip()

        if not out:
            raise ValueError("LLM returned empty patched input.")

        return out

    def _compute_replacements_for_reproduce(self, context: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        mof_path = Path(context["mof_path"])
        fw_name = context.get("mof") or mof_path.stem
        context["mof"] = fw_name

        
        ux, uy, uz = self._calculate_supercell_from_cif(mof_path)

        prop = (context.get("property") or "").strip().lower().replace(" ", "_").replace("-", "_")

        replacements: Dict[str, Any] = {
            "FrameworkName": fw_name,
            "UnitCells": f"{ux} {uy} {uz}",
        }

        meta: Dict[str, Any] = {
            "framework_name": fw_name,
            "unitcell_x": ux, "unitcell_y": uy, "unitcell_z": uz,
        }

        if prop == "selectivity":
            
            g0, g1 = self._infer_two_guests_with_llm(context)

            ctx0 = {**context, "guest": g0}
            rag0 = self._get_raspa_rag_hints(ctx0, top_files=10)
            def0 = self._choose_molecule_definition_with_llm(ctx0, rag_hints=(rag0.get("molecule_hints") or ""))
            name0 = self._select_molecule_name(g0, def0)

            ctx1 = {**context, "guest": g1}
            rag1 = self._get_raspa_rag_hints(ctx1, top_files=10)
            def1 = self._choose_molecule_definition_with_llm(ctx1, rag_hints=(rag1.get("molecule_hints") or ""))
            name1 = self._select_molecule_name(g1, def1)

            replacements["Component0"] = {"MoleculeName": name0, "MoleculeDefinition": def0}
            replacements["Component1"] = {"MoleculeName": name1, "MoleculeDefinition": def1}

            meta.update({
                "molecule_name_0": name0, "molecule_definition_0": def0,
                "molecule_name_1": name1, "molecule_definition_1": def1,
                "guest_labels": [g0, g1],
            })

        else:
            guest = context.get("guest") or "methane"
            ctxg = {**context, "guest": guest}
            rag = self._get_raspa_rag_hints(ctxg, top_files=10)
            mol_def = self._choose_molecule_definition_with_llm(ctxg, rag_hints=(rag.get("molecule_hints") or ""))
            mol_name = self._select_molecule_name(guest, mol_def)

            replacements["Component0"] = {"MoleculeName": mol_name, "MoleculeDefinition": mol_def}

            meta.update({
                "molecule_name": mol_name,
                "molecule_definition": mol_def,
            })

        return replacements, meta


    

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        mof_path = Path(context["mof_path"])

        
        fw_name = context.get("mof") or mof_path.stem
        context["mof"] = fw_name
        raspa_cif_target = self.structures_cif_dir / f"{fw_name}.cif"
        shutil.copy2(mof_path, raspa_cif_target)

        prop = (context.get("property") or "").strip().lower().replace(" ", "_").replace("-", "_")

        
        simulation_input = context.get("simulation_input") or {}
        example_text = _pick_snippet(simulation_input, "RASPA")

        if example_text:
            try:
                
                replacements, meta = self._compute_replacements_for_reproduce(context)

                
                patched_text = self._llm_patch_raspa_input(example_text, replacements)
                work_dir = Path(context["work_dir"])
                input_path = work_dir / "simulation.input"
                input_path.write_text(patched_text)
                maybe_interactive_review_input_file(
                    software="RASPA",
                    path=str(input_path),
                    context=context,
                    llm=self.llm,
                    label="RASPAInputAgent",
                )

                
                context["work_dir"] = str(work_dir)
                context["input_file"] = str(input_path)
                context["mof"] = meta.get("framework_name", fw_name)

                if prop == "selectivity":
                    context["molecule_name_0"] = meta.get("molecule_name_0")
                    context["molecule_name_1"] = meta.get("molecule_name_1")
                    context["molecule_definition_0"] = meta.get("molecule_definition_0")
                    context["molecule_definition_1"] = meta.get("molecule_definition_1")
                    context["guest_labels"] = meta.get("guest_labels", [])
                else:
                    context["molecule_name"] = meta.get("molecule_name")
                    context["molecule_definition"] = meta.get("molecule_definition")

                return context

            except Exception as e:
                print(f"[RASPAInputAgent] reproduce (LLM patch) failed -> fallback to templates: {e}")
                

        manual_hints = self._get_raspa_manual_hints(context)

        if prop in ("henry", "henry_constant", "kh", "henry_const", "henry_coefficient"):
            params = self._build_params(context, include_guest=True)
            input_text = self._generate_simulation_input_with_llm(
                params, context, manual_hints=manual_hints
            )

        elif prop == "selectivity":
            params = self._build_params(context, include_guest=False)

            mix = self._infer_mixture_spec_with_llm(context)
            components = mix["components"]
            comp_map = {c["guest"]: float(c["mol_fraction"]) for c in components}

            g0, g1 = self._infer_two_guests_with_llm(context)

            if g0 not in comp_map or g1 not in comp_map:
                raise ValueError(
                    f"Mixture fractions for selectivity guests not found. "
                    f"guests=({g0}, {g1}), components={components}"
                )

            y0 = comp_map[g0]
            y1 = comp_map[g1]

            ctx0 = {**context, "guest": g0}
            rag0 = self._get_raspa_rag_hints(ctx0, top_files=10)
            def0 = self._choose_molecule_definition_with_llm(ctx0, rag_hints=(rag0.get("molecule_hints") or ""))
            name0 = self._select_molecule_name(g0, def0)

            ctx1 = {**context, "guest": g1}
            rag1 = self._get_raspa_rag_hints(ctx1, top_files=10)
            def1 = self._choose_molecule_definition_with_llm(ctx1, rag_hints=(rag1.get("molecule_hints") or ""))
            name1 = self._select_molecule_name(g1, def1)

            params.update({
                "molecule_name_0": name0,
                "molecule_definition_0": def0,
                "mol_fraction_0": y0,
                "molecule_name_1": name1,
                "molecule_definition_1": def1,
                "mol_fraction_1": y1,
            })

            params.update(self._decide_charge_settings(
                cif_path=Path(context["mof_path"]),
                forcefield=params["forcefield"],
                guests=[(def0, name0), (def1, name1)],
                cutoff=self.SUPERCELL_CUTOFF,
                context=context,
            ))

            input_text = self._generate_simulation_input_with_llm(
                params, context, manual_hints=manual_hints
            )
            context["guests"] = [name0, name1]
            context["gas_fractions"] = {name0: y0, name1: y1}
            context["guest_labels"] = [g0, g1]
            context["molecule_name_0"] = name0
            context["molecule_name_1"] = name1
            context["molecule_definition_0"] = def0
            context["molecule_definition_1"] = def1

        else:
            params = self._build_params(context, include_guest=False)

            mix = self._infer_mixture_spec_with_llm(context)
            comps = mix["components"]

            components_for_blocks = []
            guest_pairs_for_charge = []

            for c in comps:
                g = c["guest"]
                y = float(c["mol_fraction"])

                ctxg = {**context, "guest": g}

                rag = self._get_raspa_rag_hints(ctxg, top_files=10)
                mol_def = self._choose_molecule_definition_with_llm(
                    ctxg, rag_hints=(rag.get("molecule_hints") or "")
                )
                mol_name = self._select_molecule_name(g, mol_def)

                components_for_blocks.append({
                    "molecule_definition": mol_def,
                    "molecule_name": mol_name,
                    "mol_fraction": y,
                })
                guest_pairs_for_charge.append((mol_def, mol_name))

            params.update(self._decide_charge_settings(
                cif_path=Path(context["mof_path"]),
                forcefield=params["forcefield"],
                guests=guest_pairs_for_charge,
                cutoff=self.SUPERCELL_CUTOFF,
                context=context,
            ))

            params["components"] = components_for_blocks
            if len(components_for_blocks) == 1:
                params["molecule_name"] = components_for_blocks[0]["molecule_name"]
                params["molecule_definition"] = components_for_blocks[0]["molecule_definition"]

            context["guests"] = [x["molecule_name"] for x in components_for_blocks]
            context["gas_fractions"] = {x["molecule_name"]: x["mol_fraction"] for x in components_for_blocks}

            input_text = self._generate_simulation_input_with_llm(
                params, context, manual_hints=manual_hints
            )
        input_text = self._cleanup_raspa_input_text(input_text)

        work_dir = Path(context["work_dir"])
        input_path = work_dir / "simulation.input"
        input_path.write_text(input_text)
        maybe_interactive_review_input_file(
            software="RASPA",
            path=str(input_path),
            context=context,
            llm=self.llm,
            label="RASPAInputAgent",
        )
        try:
            log_llm_decision("RASPAInputAgent", "input_generated",
                             {"input_file": str(input_path),
                              "forcefield": params.get("forcefield"),
                              "molecule_definition": params.get("molecule_definition"),
                              "temperature": params.get("temperature"),
                              "pressure_bar": params.get("pressure_bar",
                                                         params.get("pressure_pa", 0) / 1e5)},
                             context)
        except Exception:
            pass

        context["work_dir"] = str(work_dir)
        context["input_file"] = str(input_path)
        context["temperature"] = params["temperature"]
        context["pressure_bar"] = params.get("pressure_bar", params["pressure_pa"] / 1e5)
        context["forcefield"] = params["forcefield"]
        if "molecule_definition" in params:
            context["molecule_definition"] = params["molecule_definition"]

        if "molecule_definition_0" in params:
            context["molecule_definition_0"] = params.get("molecule_definition_0")
        if "molecule_definition_1" in params:
            context["molecule_definition_1"] = params.get("molecule_definition_1")

        return context
