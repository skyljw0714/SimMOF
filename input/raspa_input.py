import json
import re
import shutil

from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

from config import RASPA_DIR as RASPA_DIR_STR
from langchain.schema import HumanMessage, SystemMessage

RASPA_CUTOFF_VDW = 12.8
RASPA_GCMC_NUMBER_OF_CYCLES = 20000
RASPA_GCMC_INIT_CYCLES = 10000
RASPA_HENRY_NUMBER_OF_CYCLES = 5000
RASPA_HENRY_INIT_CYCLES = 0
RASPA_HENRY_WIDOM_INSERTIONS = 10000


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


RASPA_DIR = Path(RASPA_DIR_STR)
class RASPAInputAgent:
    GCMC_TEMPLATE = f"""SimulationType                MonteCarlo
NumberOfCycles                {RASPA_GCMC_NUMBER_OF_CYCLES}
NumberOfInitializationCycles  {RASPA_GCMC_INIT_CYCLES}
PrintEvery                    1000
PrintForcefieldToOutput       no

Forcefield                    {{forcefield}}
CutOffVDW                     {RASPA_CUTOFF_VDW}

{{charge_block}}

Framework 0
FrameworkName                 {{framework_name}}
UseChargesFromCIFFile         {{use_charges_from_cif}}

UnitCells                     {{unitcell_x}} {{unitcell_y}} {{unitcell_z}}
ExternalTemperature           {{temperature}}
ExternalPressure              {{pressure_pa}}

{{component_blocks}}
"""
    SUPERCELL_CUTOFF = RASPA_CUTOFF_VDW
    
    HENRY_TEMPLATE = f"""
SimulationType                MonteCarlo
NumberOfCycles                {RASPA_HENRY_NUMBER_OF_CYCLES}
NumberOfInitializationCycles  {RASPA_HENRY_INIT_CYCLES}
PrintEvery                    1000
PrintForcefieldToOutput       no

Forcefield                    {{forcefield}}
CutOffVDW                     {RASPA_CUTOFF_VDW}
{{charge_block}}

Framework 0
FrameworkName                 {{framework_name}}
UseChargesFromCIFFile         {{use_charges_from_cif}}

UnitCells                     {{unitcell_x}} {{unitcell_y}} {{unitcell_z}}
ExternalTemperature           {{temperature}}

Component 0 MoleculeName      {{molecule_name}}
    MoleculeDefinition        {{molecule_definition}}
    CreateNumberOfMolecules   0
    WidomProbability          1.0

    TranslationProbability    0.0
    ReinsertionProbability    0.0
    RotationProbability       0.0
    RegrowProbability         0.0
    SwapProbability           0.0
    MolFraction               1.0
"""

    SELECTIVITY_TEMPLATE = f"""SimulationType                MonteCarlo
NumberOfCycles                {RASPA_GCMC_NUMBER_OF_CYCLES}
NumberOfInitializationCycles  {RASPA_GCMC_INIT_CYCLES}
PrintEvery                    1000
PrintForcefieldToOutput       no

Forcefield                    {{forcefield}}
CutOffVDW                     {RASPA_CUTOFF_VDW}
{{charge_block}}

Framework 0
FrameworkName                 {{framework_name}}
UseChargesFromCIFFile         {{use_charges_from_cif}}

UnitCells                     {{unitcell_x}} {{unitcell_y}} {{unitcell_z}}
ExternalTemperature           {{temperature}}
ExternalPressure              {{pressure_pa}}

Component 0 MoleculeName      {{molecule_name_0}}
    MoleculeDefinition        {{molecule_definition_0}}
    IdealGasRosenbluthWeight  1.0
    TranslationProbability    1.0
    ReinsertionProbability    1.0
    RotationProbability       1.0
    RegrowProbability         3.0
    SwapProbability           4.0
    WidomProbability          1.0
    MolFraction               {{mol_fraction_0}}

Component 1 MoleculeName      {{molecule_name_1}}
    MoleculeDefinition        {{molecule_definition_1}}
    IdealGasRosenbluthWeight  1.0
    TranslationProbability    1.0
    ReinsertionProbability    1.0
    RotationProbability       1.0
    RegrowProbability         3.0
    SwapProbability           4.0
    WidomProbability          1.0
    MolFraction               {{mol_fraction_1}}
"""

    def __init__(self, llm=None):
        self.llm = llm

        
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

    
    def _decide_charge_settings(
        self,
        cif_path: Path,
        forcefield: str,
        guests: List[Tuple[str, str]] = None,  
        cutoff: float = 12.8,
        ewald_precision: str = "1e-6",
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

        use_ewald = framework_ok or guest_has_charge
        use_cif = "yes" if framework_ok else "no"

        if use_ewald:
            charge_block = (
                "ChargeMethod                  Ewald\n"
                "EwaldPrecision               {}".format(ewald_precision)
            )
        else:
            charge_block = "ChargeMethod                  None"

        return {
            "charge_block": charge_block,
            "use_charges_from_cif": use_cif,
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

            return T_val, P_val

        except Exception as e:
            print(f"[RASPAInputAgent] _infer_TP_from_query LLM/parsing failed: {e}")
            return default_T, default_P_bar

    def _get_raspa_rag_hints(self, context: Dict[str, Any], top_files: int = 5) -> Dict[str, str]:
        cached = context.get("raspa_rag_hints")
        if isinstance(cached, dict) and ("forcefield_hints" in cached or "molecule_hints" in cached):
            return {
                "forcefield_hints": (cached.get("forcefield_hints") or "").strip(),
                "molecule_hints": (cached.get("molecule_hints") or "").strip(),
            }

        sim_in = context.get("simulation_input") or {}
        if _pick_snippet(sim_in, "RASPA"):
            return {"forcefield_hints": "", "molecule_hints": ""}

        out = {"forcefield_hints": "", "molecule_hints": ""}

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

            if out["forcefield_hints"] or out["molecule_hints"]:
                print("[RAG] RASPA model hints enabled")
            else:
                print("[RAG] no relevant RASPA model hints")

            
            context["raspa_rag_hints"] = out

        except Exception as e:
            print(f"[RAG] RASPA hints disabled due to error: {e}")

        return out
        
    

    def _list_forcefields(self) -> List[str]:
        if not self.forcefield_dir.exists():
            return []

        
        preferred = {"UFF", "DREIDING", "GenericMOFs", "GenericZeolites"}

        available = {p.name for p in self.forcefield_dir.iterdir() if p.is_dir()}
        chosen = sorted(preferred & available)

        return chosen

    def _choose_forcefield_with_llm(self, context: Dict[str, Any], rag_hints: str = "") -> str:
        available = self.available_forcefields
        if not available:
            return "UFF"

        default_ff = "UFF" if "UFF" in available else available[0]

        if self.llm is None:
            return default_ff

        system_msg = (
            "You are choosing a RASPA forcefield directory for adsorption in a MOF framework.\n"
            "IMPORTANT RULES:\n"
            "- The forcefield MUST provide Lennard-Jones (vdW) parameters for the FRAMEWORK atoms.\n"
            "- For MOFs with metals (e.g., Cu, Zn, Zr, Cr, Fe), prefer UFF or Dreiding (framework-capable).\n"
            "- Do NOT choose guest-only forcefields (e.g., TraPPE) as the main forcefield for a MOF framework.\n"
            "- If unsure, choose UFF.\n"
            "Return ONLY JSON like {\"forcefield\": \"UFF\"} choosing exactly one from the allowed list."
        )

        prompt = f"""
Allowed forcefields: {available}

MOF: {context.get('mof')}
Guest: {context.get('guest')}
Property: {context.get('property')}
User query: {context.get('user_query') or context.get('query_text', '')}

RAG_HINTS (optional; may be irrelevant. Use only if clearly applicable):
{rag_hints}
"""

        try:
            resp = self.llm.invoke([
                SystemMessage(content=system_msg),
                HumanMessage(content=prompt),
            ])
            text = resp.content.strip()
            if text.startswith("```"):
                text = "\n".join(text.splitlines()[1:-1]).strip()
            obj = json.loads(text)
            cand = str(obj.get("forcefield", "")).strip()
            if cand in available:
                return cand
        except Exception as e:
            print("[RASPAInputAgent] forcefield JSON parse failed:", e)

        return default_ff

    

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

        system_msg = (
            "You choose a RASPA MoleculeDefinition family (guest model directory) for the guest molecule.\n"
            "Return ONLY JSON like {\"definition\": \"EPM2\"}.\n\n"

            "HARD RULES (priority order):\n"
            "1) If the user query explicitly requests a guest model/family (e.g., EPM2, TraPPE, TraPPE-UA, SPC/E), "
            "   you MUST choose that exact family IF it exists in the allowed list.\n"
            "2) Normalize spelling variants: 'Trappe' or 'trappe' means 'TraPPE'.\n"
            "3) If multiple models are mentioned, choose the one that matches THIS job's name if it contains a model token "
            "(e.g., job name contains 'EPM2' or 'TraPPE'). If job name doesn't specify, choose the first model mentioned.\n"
            "4) If no model is explicitly requested, then use defaults: CO2 -> EPM2 if available; otherwise prefer TraPPE/TraPPE-UA.\n"
            "5) You MUST return exactly one allowed family name.\n"
            "No extra text."
        )

        prompt = f"""
Allowed molecule definition families: {families}

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
            resp = self.llm.invoke([
                SystemMessage(content=system_msg),
                HumanMessage(content=prompt),
            ])
            text = resp.content.strip()
            if text.startswith("```"):
                text = "\n".join(text.splitlines()[1:-1]).strip()
            obj = json.loads(text)
            cand = str(obj.get("definition", "")).strip()
            if cand in families:
                return cand
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

        rag = self._get_raspa_rag_hints(context, top_files=5)
        ff_hints = rag.get("forcefield_hints", "")
        mol_hints = rag.get("molecule_hints", "")

        
        ff_name = self._choose_forcefield_with_llm(context, rag_hints=ff_hints)

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
                cutoff=self.SUPERCELL_CUTOFF
            ))

        else:
            params.update(self._decide_charge_settings(
                cif_path=cif_path,
                forcefield=ff_name,
                guests=None,
                cutoff=self.SUPERCELL_CUTOFF
            ))


        return params

    def _llm_patch_raspa_input(self, original_text: str, replacements: Dict[str, Any]) -> str:
        if self.llm is None:
            raise ValueError("LLM is required for RASPA reproduce patching (self.llm is None).")

        rep_json = json.dumps(replacements, ensure_ascii=False, indent=2)

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
            rag0 = self._get_raspa_rag_hints(ctx0, top_files=5)
            def0 = self._choose_molecule_definition_with_llm(ctx0, rag_hints=(rag0.get("molecule_hints") or ""))
            name0 = self._select_molecule_name(g0, def0)

            ctx1 = {**context, "guest": g1}
            rag1 = self._get_raspa_rag_hints(ctx1, top_files=5)
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
            rag = self._get_raspa_rag_hints(ctxg, top_files=5)
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
                

        
        if prop in ("henry", "henry_constant", "kh", "henry_const", "henry_coefficient"):
            params = self._build_params(context, include_guest=True)
            input_text = self.HENRY_TEMPLATE.format(**params)

        elif prop == "selectivity":
            params = self._build_params(context, include_guest=False)

            g0, g1 = self._infer_two_guests_with_llm(context)
            if {g0, g1} == {"CO2", "N2"}:
                if g0 != "CO2":
                    g0, g1 = g1, g0
                y0, y1 = 0.15, 0.85
            elif {g0, g1} == {"Xe", "Kr"}:
                if g0 != "Xe":
                    g0, g1 = g1, g0
                y0, y1 = 0.2, 0.8
            else:
                y0, y1 = 0.5, 0.5

            ctx0 = {**context, "guest": g0}
            rag0 = self._get_raspa_rag_hints(ctx0, top_files=5)
            def0 = self._choose_molecule_definition_with_llm(ctx0, rag_hints=(rag0.get("molecule_hints") or ""))
            name0 = self._select_molecule_name(g0, def0)

            ctx1 = {**context, "guest": g1}
            rag1 = self._get_raspa_rag_hints(ctx1, top_files=5)
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
                cutoff=self.SUPERCELL_CUTOFF
            ))

            input_text = self.SELECTIVITY_TEMPLATE.format(**params)

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

                rag = self._get_raspa_rag_hints(ctxg, top_files=5)
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
                cutoff=self.SUPERCELL_CUTOFF
            ))

            
            params["component_blocks"] = self._build_component_blocks(components_for_blocks)

            
            context["guests"] = [x["molecule_name"] for x in components_for_blocks]
            context["gas_fractions"] = {x["molecule_name"]: x["mol_fraction"] for x in components_for_blocks}

            input_text = self.GCMC_TEMPLATE.format(**params)

        work_dir = Path(context["work_dir"])
        input_path = work_dir / "simulation.input"
        input_path.write_text(input_text)

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
