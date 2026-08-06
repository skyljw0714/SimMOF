from __future__ import annotations

import json
import itertools
import math
import os
import re
import shutil
import time
import subprocess
from pathlib import Path
import numpy as np
from ase.io import read
from ase.io import write
from typing import Any, Dict, List, Optional, Union, Sequence, Tuple

from pydantic import BaseModel, Field, ValidationError
from langchain.schema import SystemMessage, HumanMessage
from config import LLM_DEFAULT, AGENT_LLM_MAP





class ExplanationGoalModel(BaseModel):
    goal: str = Field(...)


class HypothesisModel(BaseModel):
    hypothesis: str = Field(...)


class PlanStepModel(BaseModel):
    name: str
    method: str
    reason: str = ""

class SimulationPlanModel(BaseModel):
    steps: List[PlanStepModel]


class CalculationRequestModel(BaseModel):
    method: str
    engine: str
    agent: str
    requested_by: List[str] = Field(default_factory=list)


class AnalysisRecommendationModel(BaseModel):
    analysis_plan: SimulationPlanModel
    calculation_requests: List[CalculationRequestModel] = Field(default_factory=list)


class ReportSectionModel(BaseModel):
    heading: str
    body: str


class InterpretationModel(BaseModel):
    summary: str
    key_findings: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)
    next_best_step: str = ""
    report_sections: List[ReportSectionModel] = Field(default_factory=list)


def _pydantic_dump(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _pydantic_validate(model_cls, obj: Dict[str, Any]):
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(obj)
    return model_cls.parse_obj(obj)





def _tool_spec(
    engine: str,
    description: str,
    data_needs: str,
    produces: str,
    *,
    category: str = "",
    implementation: str = "",
    cost: str = "low",
) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "engine": engine,
        "description": description,
        "data_needs": data_needs,
        "produces": produces,
        "cost": cost,
    }
    if category:
        spec["category"] = category
    if implementation:
        spec["tool"] = implementation
    return spec


ANALYSIS_METHODS: Dict[str, Dict[str, Any]] = {
    "bader_charge": _tool_spec(
        "VASP",
        "Runs reference-density Bader partitioning to quantify atom-resolved electron populations and host-guest charge redistribution.",
        "Converged VASP charge densities for the relevant isolated and combined structures.",
        "Per-atom Bader populations, species summaries, and guest charge transfer.",
        cost="high",
    ),
    "binding_energy": _tool_spec(
        "VASP",
        "Computes relaxed adsorption energy from separately optimized MOF, guest, and complex energies, with a frozen-geometry interaction-energy follow-up when deformation is large.",
        "MOF, isolated guest, and optimized MOF-guest complex structures.",
        "Relaxed adsorption energy including deformation effects and, when triggered, direct frozen interaction energy; more negative values are stronger.",
        cost="high",
    ),
    "projected_dos": _tool_spec(
        "VASP",
        "Computes atom- and orbital-projected electronic densities of states for an optimized complex.",
        "A converged optimized complex and a compatible static VASP calculation.",
        "Energy-resolved orbital projections and guest-site spectral-overlap descriptors.",
        cost="high",
    ),
    "henry_coefficient": _tool_spec(
        "RASPA",
        "Calculates infinite-dilution adsorption affinity using Widom insertion or equivalent low-loading sampling.",
        "A force-field-ready framework, guest model, and temperature.",
        "Henry coefficient with uncertainty.",
        cost="medium",
    ),
    "heat_of_adsorption": _tool_spec(
        "RASPA",
        "Calculates the isosteric heat or adsorption enthalpy associated with host-guest interactions.",
        "A completed adsorption calculation with energy statistics at the requested condition.",
        "Qst or adsorption enthalpy with uncertainty.",
        cost="medium",
    ),
    "uptake": _tool_spec(
        "RASPA",
        "Runs adsorption Monte Carlo sampling to determine guest loading at specified thermodynamic conditions.",
        "A force-field-ready framework, guest model, temperature, and pressure or fugacity.",
        "Gravimetric and volumetric uptake with uncertainty and loading statistics.",
        cost="high",
    ),
    "selectivity": _tool_spec(
        "RASPA",
        "Calculates equilibrium mixture selectivity from adsorbed and gas-phase compositions.",
        "A multicomponent adsorption setup with gas fractions, temperature, and pressure.",
        "Component loadings and equilibrium selectivity.",
        cost="high",
    ),
    "pore_size_distribution": _tool_spec(
        "Zeo++",
        "Samples the void space to describe the distribution of pore sizes.",
        "A periodic framework structure and probe definition.",
        "Pore-size distribution.",
    ),
    "pore_limiting_diameter": _tool_spec(
        "Zeo++",
        "Finds the narrowest diameter along a percolating pore path.",
        "A periodic framework structure and probe definition.",
        "Pore-limiting diameter (PLD).",
    ),
    "largest_cavity_diameter": _tool_spec(
        "Zeo++",
        "Finds the diameter of the largest included cavity in the pore network.",
        "A periodic framework structure and probe definition.",
        "Largest-cavity diameter (LCD).",
    ),
    "pore_volume": _tool_spec(
        "Zeo++",
        "Measures probe-accessible void volume on mass and cell-volume bases.",
        "A periodic framework structure and probe definition.",
        "Accessible pore volume and accessible-volume fraction.",
    ),
    "surface_area": _tool_spec(
        "Zeo++",
        "Measures probe-accessible internal surface area.",
        "A periodic framework structure and probe definition.",
        "Accessible surface area on gravimetric and volumetric bases.",
    ),
    "msd": _tool_spec(
        "LAMMPS",
        "Computes the guest mean-squared displacement as a function of time.",
        "An MD trajectory with unwrapped guest coordinates and time information.",
        "MSD curve and fit-ready time series.",
        cost="high",
    ),
    "diffusivity": _tool_spec(
        "LAMMPS",
        "Estimates self-diffusivity from the long-time slope of guest displacement statistics.",
        "A sufficiently long equilibrated MD trajectory with guest coordinates.",
        "Total diffusion coefficient and fit diagnostics.",
        cost="high",
    ),
    "thermal_expansion": _tool_spec(
        "LAMMPS",
        "Measures lattice or volume changes across temperature under an appropriate ensemble.",
        "Equilibrated cell trajectories at one or more temperatures.",
        "Lattice/volume response and thermal-expansion coefficient when enough temperatures exist.",
        cost="high",
    ),
    "youngs_modulus": _tool_spec(
        "LAMMPS",
        "Applies uniaxial strain and derives Young's modulus from the stress-strain response.",
        "A mechanically equilibrated framework and a controlled deformation protocol.",
        "Stress-strain curve and Young's modulus.",
        cost="high",
    ),
    "charge_transfer_analysis": _tool_spec(
        "Analysis",
        "Interprets where electrons are gained or lost when a guest binds and connects redistribution to the local environment.",
        "Atom-resolved charge-partitioning results; structural or electronic evidence may be added when relevant.",
        "Charge-transfer mechanism, dominant atoms/species, uncertainty, and limitations.",
        category="Interpretation",
    ),
    "electronic_structure_analysis": _tool_spec(
        "Analysis",
        "Interprets guest and binding-site orbital contributions and their energy alignment.",
        "Projected electronic-structure data for the systems being compared.",
        "Orbital-resolved explanation and qualitative interaction evidence.",
        category="Interpretation",
    ),
    "henry_analysis": _tool_spec(
        "Analysis",
        "Interprets differences in infinite-dilution adsorption affinity.",
        "Henry coefficients for comparable guests, structures, and temperatures.",
        "Low-loading affinity comparison with uncertainty-aware conclusions.",
        category="Interpretation",
    ),
    "heat_of_adsorption_analysis": _tool_spec(
        "Analysis",
        "Interprets adsorption-heat differences and whether interaction strength explains observed adsorption.",
        "Qst or adsorption-enthalpy results under comparable conditions.",
        "Thermodynamic interaction-strength comparison and limitations.",
        category="Interpretation",
    ),
    "selectivity_analysis": _tool_spec(
        "Analysis",
        "Interprets equilibrium mixture selectivity and separates composition, affinity, size, and optional kinetic evidence.",
        "Mixture selectivity and component-loading results under comparable conditions.",
        "Selectivity mechanism and evidence-weighted comparison.",
        category="Interpretation",
    ),
    "pore_structure_analysis": _tool_spec(
        "Analysis",
        "Combines geometric pore descriptors to compare accessibility, bottlenecks, cavities, capacity, and surface exposure.",
        "The pore descriptors needed for the geometric question being asked.",
        "Integrated pore-structure comparison.",
        category="Interpretation",
    ),
    "diffusion_analysis": _tool_spec(
        "Analysis",
        "Interprets molecular transport using diffusion coefficients and any selected trajectory-level evidence.",
        "Comparable diffusion results; trajectory descriptors may be added for mechanism questions.",
        "Transport comparison, mechanism, uncertainty, and next calculation.",
        category="Interpretation",
    ),
    "thermal_expansion_analysis": _tool_spec(
        "Analysis",
        "Interprets framework lattice or volume response to temperature.",
        "Thermal cell/volume results over the temperature range of interest.",
        "Thermal-expansion trend and structural interpretation.",
        category="Interpretation",
    ),
    "mechanical_response_analysis": _tool_spec(
        "Analysis",
        "Interprets stiffness, anisotropy, and structural origins of stress-strain behavior.",
        "Young's-modulus or stress-strain results for comparable structures and directions.",
        "Mechanical-response comparison and mechanism.",
        category="Interpretation",
    ),
    "anisotropic_diffusion": _tool_spec(
        "Analysis",
        "Resolves diffusion along Cartesian or cell directions to identify channel-like transport.",
        "An equilibrated trajectory with unwrapped guest coordinates.",
        "Directional diffusion coefficients and anisotropy ratios.",
    ),
    "rdf_guest_host_contact": _tool_spec(
        "Analysis",
        "Computes guest-host radial distribution functions and nearest-contact statistics.",
        "A guest-containing MD trajectory with atom identities and periodic cell information.",
        "Pair RDFs, peak positions, coordination statistics, and dominant contacts.",
    ),
    "residence_hopping": _tool_spec(
        "Analysis",
        "Measures how long guests remain near sites and how frequently they hop between them.",
        "A time-resolved guest trajectory and a site representation.",
        "Residence-time distributions, hopping rates, and transition counts.",
    ),
    "van_hove_non_gaussian": _tool_spec(
        "Analysis",
        "Tests whether motion is homogeneous Fickian diffusion or contains cages and intermittent jumps.",
        "A time-resolved guest trajectory with adequate temporal sampling.",
        "Van Hove displacement distributions and non-Gaussian parameters.",
    ),
    "velocity_autocorrelation": _tool_spec(
        "Analysis",
        "Measures short-time velocity memory, collision behavior, and vibrational dynamics.",
        "Guest velocities sampled at sufficiently short time intervals.",
        "Velocity-autocorrelation function and characteristic decay times.",
    ),
    "energy_autocorrelation": _tool_spec(
        "Analysis",
        "Measures persistence and relaxation times of guest or system energy fluctuations.",
        "Time-resolved energy columns sampled during MD.",
        "Energy-autocorrelation function and correlation time.",
    ),
    "node_linker_contact": _tool_spec(
        "Analysis",
        "Aggregates guest contacts by MOF node and linker rather than by atom label alone.",
        "A guest trajectory plus node/linker assignments for the framework atoms.",
        "Node/linker contact fractions and contact-weight summaries.",
    ),
    "diffusion_activation_barrier": _tool_spec(
        "Analysis",
        "Fits an Arrhenius relationship to temperature-dependent diffusion.",
        "Comparable diffusion coefficients at multiple temperatures.",
        "Diffusion activation energy, prefactor, and fit quality.",
    ),
    "diffusion_replicate_consistency": _tool_spec(
        "Analysis",
        "Checks whether independent diffusion runs agree and quantifies between-run uncertainty.",
        "Multiple independent diffusivity estimates under identical conditions.",
        "Replicate statistics, outlier diagnostics, and consistency assessment.",
    ),
    "pore_network_hopping_graph": _tool_spec(
        "Analysis",
        "Represents adsorption regions as nodes and observed guest transitions as weighted edges.",
        "A sufficiently sampled guest trajectory and spatial site assignments.",
        "Site-to-site hopping graph, transition probabilities, and dominant pathways.",
    ),
    "uptake_analysis": _tool_spec(
        "Analysis",
        "Interprets adsorption-uptake differences using only the independently selected thermodynamic, structural, and site evidence.",
        "Comparable uptake results and any additional evidence explicitly selected for the question.",
        "Uncertainty-aware uptake explanation and unresolved alternatives.",
        category="Interpretation",
    ),
    "energy_histogram_analysis": _tool_spec(
        "Analysis",
        "Analyzes sampled adsorption-energy distributions, strong-site fractions, and multimodality.",
        "RASPA energy-histogram output generated during adsorption sampling.",
        "Energy quantiles, strong-interaction fractions, and distribution-shape descriptors.",
    ),
    "adsorption_site_density_analysis": _tool_spec(
        "Analysis",
        "Converts guest positions from adsorption sampling into spatial density and preferred-site statistics.",
        "RASPA movie or trajectory output with framework and guest coordinates.",
        "Density maps, hotspot coordinates, and site-occupancy summaries.",
    ),
    "uptake_basis_comparison": _tool_spec(
        "Analysis",
        "Compares gravimetric and volumetric uptake to expose density and normalization effects.",
        "Uptake reported on both mass and framework-volume bases with uncertainties.",
        "Basis-dependent rankings and uncertainty-aware comparison.",
    ),
    "adsorption_regime_analysis": _tool_spec(
        "Analysis",
        "Determines whether a condition is best interpreted as low-loading affinity, mixed behavior, or capacity-dominated adsorption.",
        "Pressure, temperature, uptake, and preferably neighboring isotherm points or Henry data.",
        "Adsorption-regime classification with confidence and caveats.",
    ),
    "isotherm_shape_analysis": _tool_spec(
        "Analysis",
        "Analyzes pressure-dependent curvature, knee pressure, plateau behavior, and working capacity.",
        "A consistent multi-pressure adsorption isotherm.",
        "Shape descriptors, pressure-region comparisons, and working-capacity metrics.",
    ),
    "binding_analysis": _tool_spec(
        "Analysis",
        "Interprets binding-energy differences using only the independently selected geometric, chemical, charge, and electronic evidence.",
        "Comparable binding energies and any additional evidence explicitly selected for the mechanism question.",
        "Evidence-weighted binding mechanism, uncertainties, and next test.",
        category="Interpretation",
    ),
    "binding_configuration_analysis": _tool_spec(
        "Analysis",
        "Determines where and how a guest binds from the optimized periodic host-guest geometry.",
        "Optimized complex geometry, framework atom identities, and guest atom mapping.",
        "Guest pose, contact distances, orientation descriptors, and local binding environment.",
    ),
    "linker_chemistry_analysis": _tool_spec(
        "Analysis",
        "Decomposes a periodic MOF into node and linker building units and exports chemically meaningful linker records.",
        "A periodic MOF structure, preferably CIF.",
        "Node/linker assignments, linker structures, formulas, SMILES, and atom mappings.",
        category="MOF Structure/Chemistry",
        implementation="mofstructure",
        cost="medium",
    ),
    "open_metal_site_analysis": _tool_spec(
        "Analysis",
        "Identifies exposed or under-coordinated metal sites from periodic coordination environments.",
        "A periodic MOF structure with element identities and bonding geometry.",
        "Per-metal coordination numbers, neighbor species, and open-site classifications.",
        category="MOF Structure/Chemistry",
        implementation="mofstructure/omsdetector-forked",
        cost="medium",
    ),
    "pore_surface_chemistry_analysis": _tool_spec(
        "Analysis",
        "Measures which metal, heteroatom, aromatic, node, and linker environments face probe-accessible pore space.",
        "A periodic MOF structure; node/linker atom assignments are needed for unit-resolved output.",
        "Pore-facing chemical fractions and spatial distributions.",
        category="MOF Structure/Chemistry",
        implementation="periodic probe-accessible surface + mofstructure mapping",
        cost="medium",
    ),
    "linker_functional_group_analysis": _tool_spec(
        "Analysis",
        "Detects named linker functional groups using structure-based RDKit SMARTS fingerprints.",
        "Disconnected linker structures or SMILES with source-atom mappings.",
        "Per-linker carboxylate, amine, hydroxyl, halogen, azole, and related fingerprints.",
        category="MOF Structure/Chemistry",
        implementation="RDKit SMARTS fingerprint",
        cost="medium",
    ),
}

ALLOWED_METHODS = list(ANALYSIS_METHODS.keys())

ENGINE_AGENT_MAP = {
    "VASP": "VASPAgent",
    "RASPA": "RASPAAgent",
    "Zeo++": "ZeoppAgent",
    "LAMMPS": "LAMMPSAgent",
}

DEFAULT_SYSTEM = """You are an expert computational chemistry assistant.
You must respond in STRICT JSON only (no markdown, no commentary).
Do NOT invent results.
If you need numeric values, only use those explicitly provided in the input evidence/results.
If insufficient evidence exists, clearly state uncertainty and propose the single best next step.

Hard rules:
- Output must be valid JSON.
- Do NOT include keys not requested by the schema.

Scope rule:
- Use only the exact quantities explicitly requested in the user query.
- Do not introduce additional variants, subtypes, or related metrics unless the user explicitly asks for them.
- When the user asks to explain or compare a result, select the smallest set of independent
  evidence providers needed to test the stated hypothesis; those providers are within scope.
- If the requested quantity is ambiguous, keep the goal/hypothesis/plans minimal and do not expand the scope.
"""


class AnalysisAgent:
    METAL_SPECIES = {
        "Li", "Be", "Na", "Mg", "K", "Ca", "Rb", "Sr", "Cs", "Ba",
        
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        
        "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        
        "Al", "Ga", "In", "Sn", "Pb", "Bi",
    }

    def __init__(self, llm=None, agent_name: str = "AnalysisAgent"):
        self.agent_name = agent_name
        self.llm = llm if llm is not None else AGENT_LLM_MAP.get(agent_name, LLM_DEFAULT)


    @staticmethod
    def _match_atoms_by_distance(
        poscar_mof: Path,
        poscar_complex: Path,
        cutoff: float = 0.5,
        excluded_complex_indices1: Optional[Sequence[int]] = None,
    ):
        from ase.io import read

        atoms_mof = read(poscar_mof)
        atoms_complex = read(poscar_complex)
        excluded_complex_indices = {
            int(index)
            for index in (excluded_complex_indices1 or [])
        }

        N_mof = len(atoms_mof)
        N_complex = len(atoms_complex)

        if excluded_complex_indices:
            remaining_complex_indices1 = [
                index
                for index in range(1, N_complex + 1)
                if index not in excluded_complex_indices
            ]
            if (
                len(remaining_complex_indices1) == N_mof
                and atoms_mof.get_chemical_symbols()
                == [
                    atoms_complex[index1 - 1].symbol
                    for index1 in remaining_complex_indices1
                ]
            ):
                return (
                    {
                        mof_index1: complex_index1
                        for mof_index1, complex_index1 in enumerate(
                            remaining_complex_indices1,
                            start=1,
                        )
                    },
                    sorted(excluded_complex_indices),
                )

        def poscar_species_counts(path: Path) -> Optional[Tuple[List[str], List[int]]]:
            try:
                lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
                species = lines[5].split()
                counts = [int(x) for x in lines[6].split()]
            except Exception:
                return None
            if not species or len(species) != len(counts):
                return None
            return species, counts

        mof_sc = poscar_species_counts(poscar_mof)
        complex_sc = poscar_species_counts(poscar_complex)
        if (
            not excluded_complex_indices
            and mof_sc
            and complex_sc
            and mof_sc[0] == complex_sc[0]
        ):
            species, mof_counts = mof_sc
            _, complex_counts = complex_sc
            if len(mof_counts) == len(complex_counts) and all(c >= m for m, c in zip(mof_counts, complex_counts)):
                mapping: Dict[int, int] = {}
                guest_indices: List[int] = []
                mof_start = 1
                complex_start = 1
                for mof_count, complex_count in zip(mof_counts, complex_counts):
                    for offset in range(mof_count):
                        mapping[int(mof_start + offset)] = int(complex_start + offset)
                    for offset in range(mof_count, complex_count):
                        guest_indices.append(int(complex_start + offset))
                    mof_start += mof_count
                    complex_start += complex_count
                if len(mapping) == N_mof and len(mapping) + len(guest_indices) == N_complex:
                    return mapping, guest_indices

        combined = atoms_mof + atoms_complex

        mapping: Dict[int, int] = {}
        matched_complex: set[int] = set()

        for i_mof in range(N_mof):
            target_indices = [
                N_mof + j_complex0
                for j_complex0 in range(N_complex)
                if (j_complex0 + 1) not in matched_complex
                and (j_complex0 + 1) not in excluded_complex_indices
                and atoms_complex[j_complex0].symbol == atoms_mof[i_mof].symbol
            ]
            if not target_indices:
                target_indices = [
                    N_mof + j_complex0
                    for j_complex0 in range(N_complex)
                    if (j_complex0 + 1) not in matched_complex
                    and (j_complex0 + 1) not in excluded_complex_indices
                ]
            if not target_indices:
                continue
            dists = combined.get_distances(i_mof, target_indices, mic=True)

            
            j_rel = int(np.argmin(dists))          
            d_min = float(dists[j_rel])

            j_complex0 = int(target_indices[j_rel] - N_mof)
            j_complex1 = int(j_complex0 + 1)       
            mof_idx1 = int(i_mof + 1)              

            if d_min > cutoff:
                print(
                    f"[Warning] Distance from MOF atom {mof_idx1} to the nearest complex atom = "
                    f"{d_min:.2f} Å (exceeds cutoff {cutoff} Å)"
                )

            mapping[mof_idx1] = j_complex1
            matched_complex.add(j_complex1)

        
        guest_indices: List[int] = [
            int(j) for j in range(1, N_complex + 1) if j not in matched_complex
        ]

        return mapping, guest_indices


    def _parse_poscar_idx_to_species(self, poscar_path: Path) -> Dict[int, str]:
        lines = poscar_path.read_text(encoding="utf-8", errors="ignore").splitlines()

        if len(lines) < 7:
            raise ValueError(f"POSCAR too short: {poscar_path}")

        species_line = lines[5].split()
        counts_line = lines[6].split()

        if not species_line:
            raise ValueError(f"POSCAR has empty species line: {poscar_path}")
        if not counts_line:
            raise ValueError(f"POSCAR has empty counts line: {poscar_path}")

        species = species_line
        counts = list(map(int, counts_line))

        if len(species) != len(counts):
            raise ValueError(
                f"POSCAR species/count mismatch in {poscar_path}:\n"
                f"  species={species}\n  counts={counts}"
            )

        idx_to_species: Dict[int, str] = {}
        idx = 1  

        for sp, cnt in zip(species, counts):
            for _ in range(cnt):
                idx_to_species[idx] = sp
                idx += 1

        return idx_to_species

    def _parse_acf_dat(self, acf_path: Path) -> Dict[int, float]:
        if not acf_path.exists():
            raise FileNotFoundError(f"ACF.dat not found at {acf_path}")

        idx_to_e: Dict[int, float] = {}

        lines = acf_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line in lines:
            if re.match(r"^\s*#", line):
                continue
            if re.match(r"^\s*-{3,}", line):
                continue
            if not line.strip():
                continue

            parts = line.split()
            
            if len(parts) >= 5 and parts[0].isdigit():
                idx = int(parts[0])
                e = float(parts[4])
                idx_to_e[idx] = e

        if not idx_to_e:
            raise ValueError(f"ACF.dat has no valid atom lines: {acf_path}")

        return idx_to_e

    def _summarize_delta_q(self, delta: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
        framework = delta.get("framework", {}) or {}
        metal_sites = delta.get("metal_sites", {}) or {}
        guest = delta.get("guest", {}) or {}
        summ = delta.get("summary", {}) or {}
        aggregates = delta.get("aggregates", {}) or {}

        framework_items = []
        framework_total = 0.0
        framework_abs_total = 0.0
        species_delta: Dict[str, Dict[str, Any]] = {}
        for idx_str, rec in framework.items():
            try:
                idx = int(idx_str)
            except Exception:
                idx = idx_str
            dq = float(rec.get("delta_q", 0.0))
            sp = rec.get("species", "?")
            framework_total += dq
            framework_abs_total += abs(dq)
            bucket = species_delta.setdefault(
                sp,
                {"count": 0, "delta_q_total": 0.0, "abs_delta_q_total": 0.0},
            )
            bucket["count"] += 1
            bucket["delta_q_total"] += dq
            bucket["abs_delta_q_total"] += abs(dq)
            framework_items.append(
                {
                    "mof_index": idx,
                    "complex_index": rec.get("complex_index"),
                    "species": sp,
                    "delta_q": dq,
                    "q_mof": rec.get("q_mof"),
                    "q_complex": rec.get("q_complex"),
                }
            )

        framework_items_by_index = sorted(framework_items, key=lambda x: x["mof_index"])
        framework_items_by_abs = sorted(
            framework_items,
            key=lambda x: abs(float(x["delta_q"])),
            reverse=True,
        )
        framework_top = framework_items_by_abs[: max(1, int(top_k))]

        if aggregates:
            framework_total = float(
                aggregates.get("framework_delta_e_total", framework_total)
            )
            aggregate_by_species = aggregates.get("framework_by_species", {}) or {}
            species_delta = {
                species: {
                    **record,
                    "abs_delta_q_total": (
                        species_delta.get(species, {}).get("abs_delta_q_total")
                        if framework_items
                        else None
                    ),
                }
                for species, record in aggregate_by_species.items()
            }
            if not framework_items:
                framework_abs_total = None

        metal_items = []
        metal_total = 0.0
        for idx_str, rec in metal_sites.items():
            try:
                idx = int(idx_str)
            except Exception:
                idx = idx_str
            dq = float(rec.get("delta_q", 0.0))
            metal_total += dq
            metal_items.append(
                {
                    "mof_index": idx,
                    "species": rec.get("species", "?"),
                    "delta_q": dq,
                    "q_mof": rec.get("q_mof"),
                    "q_complex": rec.get("q_complex"),
                }
            )

        metal_items_sorted = sorted(metal_items, key=lambda x: abs(float(x["delta_q"])), reverse=True)
        metal_top = metal_items_sorted[: max(1, int(top_k))]
        if aggregates:
            metal_total = float(
                aggregates.get("metal_delta_e_total", metal_total)
            )

        
        guest_items = []
        guest_charge_sum = 0.0
        guest_species_count = {}
        for idx_str, rec in guest.items():
            try:
                idx = int(idx_str)
            except Exception:
                idx = idx_str
            q = float(rec.get("q_complex", 0.0))
            sp = rec.get("species", "?")
            guest_charge_sum += q
            guest_items.append({"complex_index": idx, "species": sp, "q_complex": q})
            guest_species_count[sp] = guest_species_count.get(sp, 0) + 1

        guest_items_sorted = sorted(guest_items, key=lambda x: x["complex_index"])

        return {
            "definition": {
                "acf_column": "CHARGE (ACF parts[4])",
                "acf_charge_semantics": "integrated electron population inside each Bader basin; not net atomic charge or formal oxidation state",
                "framework_delta_q": "legacy field name for each matched atom's electron-count change: CHARGE_complex - CHARGE_mof",
                "framework_delta_q_total": "framework electron-count change: sum(CHARGE_complex) - sum(CHARGE_mof)",
                "delta_sign": "positive means electron gain; negative means electron loss",
                "metal_delta_q": "legacy field name for the metal-subset electron-count change",
                "co2_charge": "sum of Bader electron populations over guest atoms in the complex; not the guest net charge",
                "guest_net_bader_charge": "from charge conservation, equal to framework electron-count change; negative means the guest gained electrons",
            },
            "counts": {
                "n_framework_atoms": summ.get("n_framework_atoms"),
                "n_framework_atoms_with_delta_q": len(framework_items),
                "n_guest_atoms": summ.get("n_guest_atoms"),
                "n_metal_sites": summ.get("n_metal_sites"),
                "metal_species_found": summ.get("metal_species_found"),
                "guest_species_count": guest_species_count,
                "atom_mapping_status": summ.get("atom_mapping_status"),
            },
            "framework": {
                "delta_q_total": framework_total,
                "abs_delta_q_total": framework_abs_total,
                "by_species": species_delta,
                "top_atoms_by_abs_delta_q": framework_top,
                "all_atoms": framework_items_by_index,
                "atom_mapping": delta.get("atom_mapping", {}),
            },
            "metal": {
                "delta_q_total": metal_total,
                "top_sites_by_abs_delta_q": metal_top,
            },
            "co2": {
                "guest_charge_sum_in_complex": guest_charge_sum,
                "guest_net_bader_charge_from_conservation_e": framework_total,
                "guest_electron_gain_from_conservation_e": -framework_total,
                "guest_atoms": guest_items_sorted,
            },
        }

    @staticmethod
    def _match_framework_atoms_with_structure_matcher(
        mof_poscar: Path,
        complex_poscar: Path,
        guest_indices1: Sequence[int],
    ) -> Dict[str, Any]:
        try:
            from pymatgen.analysis.structure_matcher import StructureMatcher
            from pymatgen.core import Structure
        except Exception as exc:
            return {
                "status": "unavailable",
                "reason": f"pymatgen import failed: {type(exc).__name__}: {exc}",
            }

        try:
            mof_structure = Structure.from_file(str(mof_poscar))
            complex_structure = Structure.from_file(str(complex_poscar))
            remaining_complex_indices1 = [
                index
                for index in range(1, len(complex_structure) + 1)
                if index not in {int(value) for value in guest_indices1}
            ]
            framework_structure = complex_structure.copy()
            framework_structure.remove_sites(
                sorted(
                    {int(value) - 1 for value in guest_indices1},
                    reverse=True,
                )
            )
            matcher = StructureMatcher(
                ltol=0.3,
                stol=0.5,
                angle_tol=10,
                primitive_cell=False,
                scale=False,
                attempt_supercell=False,
            )
            mapping_array = matcher.get_mapping(
                mof_structure,
                framework_structure,
            )
            if mapping_array is None:
                return {
                    "status": "unavailable",
                    "reason": "framework structures do not match within StructureMatcher tolerances",
                    "ltol": 0.3,
                    "stol": 0.5,
                    "angle_tol_deg": 10,
                }
            rms = matcher.get_rms_dist(
                mof_structure,
                framework_structure,
            )
            mapping = {
                int(mof_index0 + 1): int(
                    remaining_complex_indices1[framework_index0]
                )
                for framework_index0, mof_index0 in enumerate(mapping_array)
            }
            return {
                "status": "ok",
                "method": "pymatgen StructureMatcher",
                "mapping": mapping,
                "normalized_rms_displacement": (
                    float(rms[0]) if rms is not None else None
                ),
                "normalized_max_displacement": (
                    float(rms[1]) if rms is not None else None
                ),
                "ltol": 0.3,
                "stol": 0.5,
                "angle_tol_deg": 10,
            }
        except Exception as exc:
            return {
                "status": "unavailable",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        
    def _build_bader_delta_q_for_mof_complex(
        self,
        mof_dir: Path,
        complex_dir: Path,
        guest_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:

        mof_poscar = mof_dir / "POSCAR"
        complex_poscar = complex_dir / "POSCAR"
        acf_mof = mof_dir / "ACF.dat"
        acf_complex = complex_dir / "ACF.dat"

        guest_identification: Dict[str, Any] = {}
        guest_structure = (
            self._preferred_vasp_structure_path(guest_dir)
            if guest_dir is not None
            else None
        )
        if guest_structure is not None:
            guest_identification = self._match_guest_by_internal_geometry(
                guest_structure,
                complex_poscar,
            )

        if guest_identification.get("status") == "ok":
            guest_indices = [
                int(index)
                for index in guest_identification["complex_indices"]
            ]
            atom_mapping = self._match_framework_atoms_with_structure_matcher(
                mof_poscar,
                complex_poscar,
                guest_indices,
            )
            mapping = atom_mapping.get("mapping", {})
        else:
            mapping, guest_indices = self._match_atoms_by_distance(
                mof_poscar,
                complex_poscar,
                cutoff=0.5,
            )
            atom_mapping = {
                "status": "fallback_distance_mapping",
                "mapping": mapping,
            }


        
        idx_to_e_mof = self._parse_acf_dat(acf_mof)
        idx_to_e_complex = self._parse_acf_dat(acf_complex)

        
        idx_to_sp_mof = self._parse_poscar_idx_to_species(mof_poscar)
        idx_to_sp_complex = self._parse_poscar_idx_to_species(complex_poscar)

        framework_complex_indices = [
            index
            for index in sorted(idx_to_e_complex)
            if index not in {int(value) for value in guest_indices}
        ]
        framework_by_species: Dict[str, Dict[str, Any]] = {}
        for idx, electron_count in idx_to_e_mof.items():
            species = idx_to_sp_mof.get(idx, "?")
            bucket = framework_by_species.setdefault(
                species,
                {
                    "count": 0,
                    "electron_count_mof": 0.0,
                    "electron_count_complex": 0.0,
                    "delta_q_total": 0.0,
                },
            )
            bucket["count"] += 1
            bucket["electron_count_mof"] += float(electron_count)
        for idx in framework_complex_indices:
            species = idx_to_sp_complex.get(idx, "?")
            bucket = framework_by_species.setdefault(
                species,
                {
                    "count": 0,
                    "electron_count_mof": 0.0,
                    "electron_count_complex": 0.0,
                    "delta_q_total": 0.0,
                },
            )
            bucket["electron_count_complex"] += float(idx_to_e_complex[idx])
        for bucket in framework_by_species.values():
            bucket["delta_q_total"] = (
                bucket["electron_count_complex"]
                - bucket["electron_count_mof"]
            )

        framework_delta_e_total = sum(
            float(idx_to_e_complex[idx])
            for idx in framework_complex_indices
        ) - sum(float(value) for value in idx_to_e_mof.values())
        metal_delta_e_total = sum(
            record["delta_q_total"]
            for species, record in framework_by_species.items()
            if species in self.METAL_SPECIES
        )

        
        framework = {}
        for mof_idx, complex_idx in mapping.items():
            if mof_idx not in idx_to_e_mof or complex_idx not in idx_to_e_complex:
                continue

            sp = idx_to_sp_mof.get(mof_idx, "?")
            q_mof = idx_to_e_mof[mof_idx]
            q_complex = idx_to_e_complex[complex_idx]
            dq = q_complex - q_mof

            framework[mof_idx] = {
                "species": sp,
                "q_mof": q_mof,
                "q_complex": q_complex,
                "delta_q": dq,
                "complex_index": complex_idx,
            }

        
        guest = {}
        for idx in guest_indices:
            if idx in idx_to_e_complex:
                sp = idx_to_sp_complex.get(idx, "?")
                q = idx_to_e_complex[idx]
                guest[idx] = {
                    "species": sp,
                    "q_complex": q,
                }

        
        metal_sites = {
            int(idx): rec
            for idx, rec in framework.items()
            if rec["species"] in self.METAL_SPECIES
        }

        summary = {
            "n_framework_atoms": int(len(framework)),
            "n_guest_atoms": int(len(guest)),
            "n_metal_sites": int(
                sum(
                    record["count"]
                    for species, record in framework_by_species.items()
                    if species in self.METAL_SPECIES
                )
            ),
            "guest_indices": [int(i) for i in guest_indices],
            "atom_mapping_status": atom_mapping.get("status"),
            "metal_species_found": sorted(
                {
                    species
                    for species in framework_by_species
                    if species in self.METAL_SPECIES
                }
            ),
        }

        return {
            "framework": {int(k): v for k, v in framework.items()},
            "guest": {int(k): v for k, v in guest.items()},
            "metal_sites": metal_sites,
            "summary": summary,
            "guest_identification": guest_identification,
            "atom_mapping": {
                key: value
                for key, value in atom_mapping.items()
                if key != "mapping"
            },
            "aggregates": {
                "framework_delta_e_total": framework_delta_e_total,
                "framework_by_species": framework_by_species,
                "metal_delta_e_total": metal_delta_e_total,
            },
        }


    def _extract_bader_dirs_any(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Tuple[Path, Path, Optional[Path]]]:
        upstream = context.get("upstream_plans", {}) or {}
        out: Dict[str, Tuple[Path, Path, Optional[Path]]] = {}
        binding_pairs = self._extract_binding_structure_pairs_any(context)

        for plan_name, plan_blob in upstream.items():
            if not isinstance(plan_blob, dict):
                continue
            if not plan_name.endswith("_bader_charge"):
                continue

            mof_job = plan_blob.get(f"{plan_name}_mof", {})
            complex_job = plan_blob.get(f"{plan_name}_complex", {})

            mof_dir = mof_job.get("results", {}).get("bader_charge", {}).get("bader_dir")
            complex_dir = complex_job.get("results", {}).get("bader_charge", {}).get("bader_dir")

            if mof_dir and complex_dir:
                target_mof = mof_job.get("mof") or complex_job.get("mof")
                target_guest = mof_job.get("guest") or complex_job.get("guest")
                matching_pairs = [
                    info
                    for info in binding_pairs.values()
                    if info.get("mof") == target_mof
                    and info.get("guest") == target_guest
                ]
                guest_dir = (
                    matching_pairs[0].get("guest_dir")
                    if len(matching_pairs) == 1
                    else None
                )
                out[plan_name] = (
                    Path(mof_dir),
                    Path(complex_dir),
                    guest_dir,
                )

        return out

    @staticmethod
    def _preferred_vasp_structure_path(vasp_dir: Path) -> Optional[Path]:
        for name in ("CONTCAR", "POSCAR"):
            path = vasp_dir / name
            if path.exists() and path.stat().st_size > 0:
                return path
        return None

    @staticmethod
    def _match_guest_by_internal_geometry(
        guest_structure: Path,
        complex_structure: Path,
        max_combinations: int = 200000,
    ) -> Dict[str, Any]:
        guest_atoms = read(guest_structure)
        complex_atoms = read(complex_structure)
        guest_symbols = guest_atoms.get_chemical_symbols()
        species = sorted(set(guest_symbols))

        guest_counts = {
            symbol: guest_symbols.count(symbol)
            for symbol in species
        }
        complex_indices = {
            symbol: [
                index
                for index, atom in enumerate(complex_atoms)
                if atom.symbol == symbol
            ]
            for symbol in species
        }
        if any(
            len(complex_indices[symbol]) < count
            for symbol, count in guest_counts.items()
        ):
            return {
                "status": "species_count_mismatch",
                "guest_formula": guest_atoms.get_chemical_formula(),
            }

        combination_count = 1
        for symbol, count in guest_counts.items():
            combination_count *= math.comb(len(complex_indices[symbol]), count)
        if combination_count > max_combinations:
            return {
                "status": "too_many_combinations",
                "combination_count": combination_count,
                "max_combinations": max_combinations,
            }

        def distance_fingerprint(atoms, indices0: Sequence[int]) -> Dict[str, List[float]]:
            fingerprint: Dict[str, List[float]] = {}
            for left_pos, left_index in enumerate(indices0):
                for right_index in indices0[left_pos + 1 :]:
                    key = "-".join(
                        sorted(
                            [
                                atoms[left_index].symbol,
                                atoms[right_index].symbol,
                            ]
                        )
                    )
                    fingerprint.setdefault(key, []).append(
                        float(
                            atoms.get_distance(
                                left_index,
                                right_index,
                                mic=True,
                            )
                        )
                    )
            for values in fingerprint.values():
                values.sort()
            return fingerprint

        reference = distance_fingerprint(
            guest_atoms,
            list(range(len(guest_atoms))),
        )
        choice_lists = [
            list(
                itertools.combinations(
                    complex_indices[symbol],
                    guest_counts[symbol],
                )
            )
            for symbol in species
        ]

        best_indices0: Optional[List[int]] = None
        best_rms = float("inf")
        for grouped_choice in itertools.product(*choice_lists):
            candidate_indices0 = sorted(
                index
                for group in grouped_choice
                for index in group
            )
            candidate = distance_fingerprint(
                complex_atoms,
                candidate_indices0,
            )
            if set(candidate) != set(reference):
                continue
            differences = []
            valid = True
            for key, reference_distances in reference.items():
                candidate_distances = candidate.get(key, [])
                if len(reference_distances) != len(candidate_distances):
                    valid = False
                    break
                differences.extend(
                    candidate_distance - reference_distance
                    for candidate_distance, reference_distance in zip(
                        candidate_distances,
                        reference_distances,
                    )
                )
            if not valid:
                continue
            rms = float(
                np.sqrt(np.mean(np.square(differences)))
                if differences
                else 0.0
            )
            if rms < best_rms:
                best_rms = rms
                best_indices0 = candidate_indices0

        if best_indices0 is None:
            return {
                "status": "no_geometry_match",
                "combination_count": combination_count,
            }
        return {
            "status": "ok",
            "method": "minimum pair-distance fingerprint RMSD to isolated guest",
            "guest_formula": guest_atoms.get_chemical_formula(),
            "complex_indices": [int(index + 1) for index in best_indices0],
            "symbols": [
                complex_atoms[index].symbol
                for index in best_indices0
            ],
            "pair_distance_rmsd_A": best_rms,
            "combination_count": combination_count,
        }

    @staticmethod
    def _job_mof_cif_path(job: Optional[Dict[str, Any]]) -> Optional[Path]:
        if not isinstance(job, dict):
            return None

        def as_existing_cif(value: Any) -> Optional[Path]:
            if not value:
                return None
            path = Path(str(value))
            if path.suffix.lower() == ".cif" and path.exists():
                return path
            return None

        for key in ("optimized_mof_path", "mof_path", "cif_path"):
            path = as_existing_cif(job.get(key))
            if path:
                return path

        results = job.get("results", {}) or {}
        if isinstance(results, dict):
            for key in ("optimized_mof_path", "mof_path", "cif_path"):
                path = as_existing_cif(results.get(key))
                if path:
                    return path

        return None

    def _extract_binding_structure_pairs_any(self, context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        upstream = context.get("upstream_plans", {}) or {}
        out: Dict[str, Dict[str, Any]] = {}

        for plan_name, plan_blob in upstream.items():
            if not isinstance(plan_blob, dict):
                continue
            if not plan_name.endswith("_binding_energy"):
                continue

            mof_job = plan_blob.get(f"{plan_name}_mof")
            guest_job = plan_blob.get(f"{plan_name}_guest")
            complex_job = plan_blob.get(f"{plan_name}_complex")

            if not isinstance(mof_job, dict) or not isinstance(complex_job, dict):
                for job_id, job in plan_blob.items():
                    if not isinstance(job, dict):
                        continue
                    role = job.get("vasp_role")
                    if role == "mof" or str(job_id).endswith("_mof"):
                        mof_job = job
                    elif role == "guest" or str(job_id).endswith("_guest"):
                        guest_job = job
                    elif role == "complex" or str(job_id).endswith("_complex"):
                        complex_job = job

            if not isinstance(mof_job, dict) or not isinstance(complex_job, dict):
                continue

            mof_dir = mof_job.get("vasp_dir") or (mof_job.get("vasp_system") or {}).get("dir")
            guest_dir = None
            if isinstance(guest_job, dict):
                guest_dir = guest_job.get("vasp_dir") or (
                    guest_job.get("vasp_system") or {}
                ).get("dir")
            complex_dir = complex_job.get("vasp_dir") or (complex_job.get("vasp_system") or {}).get("dir")
            if not mof_dir or not complex_dir:
                continue

            mof_name = complex_job.get("mof") or mof_job.get("mof") or plan_name
            guest_name = complex_job.get("guest") or mof_job.get("guest")
            out[plan_name] = {
                "mof": mof_name,
                "guest": guest_name,
                "mof_dir": Path(mof_dir),
                "guest_dir": Path(guest_dir) if guest_dir else None,
                "complex_dir": Path(complex_dir),
                "mof_cif_path": self._job_mof_cif_path(mof_job),
            }

        return out

    @staticmethod
    def _site_type_for_species(species: str) -> str:
        if species in AnalysisAgent.METAL_SPECIES:
            return "metal_site"
        if species in {"O", "N", "S", "P", "F", "Cl", "Br", "I"}:
            return "polar_linker_or_heteroatom"
        if species == "H":
            return "hydrogen_contact"
        if species == "C":
            return "organic_linker_carbon"
        return "framework_atom"

    @staticmethod
    def _formula_from_symbols(symbols: List[str]) -> str:
        counts: Dict[str, int] = {}
        for sp in symbols:
            counts[sp] = counts.get(sp, 0) + 1
        parts = []
        for sp in sorted(counts.keys()):
            n = counts[sp]
            parts.append(sp if n == 1 else f"{sp}{n}")
        return "".join(parts)

    @staticmethod
    def _vector_angle_deg(v1: np.ndarray, v2: np.ndarray) -> Optional[float]:
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 == 0.0 or n2 == 0.0:
            return None
        cosang = float(np.dot(v1, v2) / (n1 * n2))
        cosang = max(-1.0, min(1.0, cosang))
        return float(np.degrees(np.arccos(abs(cosang))))

    @staticmethod
    def _map_structure_atom_indices_by_distance(
        source_structure: Path,
        target_structure: Path,
        cutoff_A: float = 0.6,
    ) -> Dict[int, int]:
        source_atoms = read(source_structure)
        target_atoms = read(target_structure)
        combined = source_atoms + target_atoms
        target_offset = len(source_atoms)
        target_indices = list(range(target_offset, target_offset + len(target_atoms)))
        used_targets: set[int] = set()
        mapping: Dict[int, int] = {}

        for src_i0, src_atom in enumerate(source_atoms):
            dists = combined.get_distances(src_i0, target_indices, mic=True)
            best: Optional[Tuple[float, int]] = None
            for rel_j, d in enumerate(dists):
                tgt_i0 = rel_j
                if tgt_i0 in used_targets:
                    continue
                if target_atoms[tgt_i0].symbol != src_atom.symbol:
                    continue
                d_float = float(d)
                if best is None or d_float < best[0]:
                    best = (d_float, tgt_i0)
            if best is None:
                continue
            d_best, tgt_i0 = best
            if d_best <= cutoff_A:
                mapping[int(src_i0 + 1)] = int(tgt_i0 + 1)
                used_targets.add(tgt_i0)
        return mapping

    def _chemistry_units_by_mof_index(
        self,
        chemistry_summary: Dict[str, Any],
        chemistry_source_path: Path,
        mof_structure: Path,
    ) -> Dict[int, Dict[str, Any]]:
        structures = chemistry_summary.get("structures", []) if isinstance(chemistry_summary, dict) else []
        if not structures:
            return {}

        selected = None
        source_resolved = str(Path(chemistry_source_path).resolve())
        for item in structures:
            if not isinstance(item, dict):
                continue
            try:
                if str(Path(item.get("cif_path", "")).resolve()) == source_resolved:
                    selected = item
                    break
            except Exception:
                continue
        if selected is None:
            selected = structures[0]
        if selected.get("status") != "ok":
            return {}

        source_to_mof = self._map_structure_atom_indices_by_distance(
            chemistry_source_path,
            mof_structure,
            cutoff_A=0.8,
        )

        by_mof_index: Dict[int, Dict[str, Any]] = {}
        for unit_key, unit_type in (("nodes", "node"), ("linkers", "linker")):
            for unit in selected.get(unit_key, []) or []:
                source_indices = unit.get("source_atom_indices", []) or []
                mof_indices = []
                for src_idx1 in source_indices:
                    mof_idx1 = source_to_mof.get(int(src_idx1))
                    if mof_idx1 is not None:
                        mof_indices.append(int(mof_idx1))
                unit_record = {
                    "unit_type": unit_type,
                    "unit_index": unit.get("index"),
                    "formula": unit.get("formula"),
                    "sbu_type": unit.get("sbu_type"),
                    "smiles": unit.get("smiles"),
                    "inchikey": unit.get("inchikey"),
                    "functional_tags": unit.get("functional_tags", []),
                    "point_count": unit.get("point_count"),
                    "exports": unit.get("exports", {}),
                    "mof_atom_indices": sorted(set(mof_indices)),
                }
                for mof_idx1 in unit_record["mof_atom_indices"]:
                    by_mof_index[int(mof_idx1)] = unit_record
        return by_mof_index

    @staticmethod
    def _compact_chemistry_unit(unit: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not unit:
            return None
        return {
            "unit_type": unit.get("unit_type"),
            "unit_index": unit.get("unit_index"),
            "formula": unit.get("formula"),
            "sbu_type": unit.get("sbu_type"),
            "functional_tags": unit.get("functional_tags", []),
            "smiles": unit.get("smiles"),
            "inchikey": unit.get("inchikey"),
        }

    def _summarize_unit_contacts(self, contacts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        grouped: Dict[str, Dict[str, Any]] = {}
        for contact in contacts:
            unit = contact.get("chemistry_unit")
            if not unit:
                continue
            key = f"{unit.get('unit_type')}:{unit.get('unit_index')}:{unit.get('formula')}"
            rec = grouped.setdefault(
                key,
                {
                    "unit": unit,
                    "nearest_distance_A": None,
                    "contact_atoms": set(),
                    "guest_atoms": set(),
                    "n_contacts_within_cutoff": 0,
                },
            )
            d = float(contact["distance_A"])
            if rec["nearest_distance_A"] is None or d < rec["nearest_distance_A"]:
                rec["nearest_distance_A"] = d
                rec["nearest_contact"] = contact
            rec["contact_atoms"].add(contact.get("framework_atom"))
            rec["guest_atoms"].add(contact.get("guest_atom"))
            rec["n_contacts_within_cutoff"] += 1

        out = []
        for rec in grouped.values():
            nearest = dict(rec.get("nearest_contact", {}))
            nearest.pop("chemistry_unit", None)
            out.append(
                {
                    "unit": rec["unit"],
                    "nearest_distance_A": rec["nearest_distance_A"],
                    "contact_atoms": sorted(a for a in rec["contact_atoms"] if a),
                    "guest_atoms": sorted(a for a in rec["guest_atoms"] if a),
                    "n_contacts_within_cutoff": rec["n_contacts_within_cutoff"],
                    "nearest_contact": nearest,
                }
            )
        return sorted(out, key=lambda x: x["nearest_distance_A"])

    @staticmethod
    def _normalize_weights(weights: Dict[str, float], total: float) -> Dict[str, float]:
        if total <= 0.0:
            return {}
        return {k: float(v / total) for k, v in sorted(weights.items()) if v > 0.0}

    def _build_local_binding_environment(
        self,
        contacts: List[Dict[str, Any]],
        cutoff_A: float,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        weighted_contacts: List[Dict[str, Any]] = []
        unit_type_weights: Dict[str, float] = {}
        site_type_weights: Dict[str, float] = {}
        guest_profiles: Dict[str, Dict[str, Any]] = {}
        unit_weights: Dict[str, Dict[str, Any]] = {}
        total_weight = 0.0
        n_with_unit = 0

        for contact in contacts:
            d = self._clean_numeric(contact.get("distance_A"))
            if d is None or d > cutoff_A:
                continue
            weight = max(0.0, 1.0 - float(d) / float(cutoff_A))
            if weight <= 0.0:
                continue

            unit = contact.get("chemistry_unit") or {}
            unit_type = unit.get("unit_type") or "unassigned"
            site_type = contact.get("site_type") or "unknown"
            guest_atom = contact.get("guest_atom") or "guest"

            total_weight += weight
            if contact.get("chemistry_unit"):
                n_with_unit += 1
            unit_type_weights[unit_type] = unit_type_weights.get(unit_type, 0.0) + weight
            site_type_weights[site_type] = site_type_weights.get(site_type, 0.0) + weight

            gp = guest_profiles.setdefault(
                guest_atom,
                {
                    "total_weight": 0.0,
                    "unit_type_weights": {},
                    "site_type_weights": {},
                },
            )
            gp["total_weight"] += weight
            gp["unit_type_weights"][unit_type] = gp["unit_type_weights"].get(unit_type, 0.0) + weight
            gp["site_type_weights"][site_type] = gp["site_type_weights"].get(site_type, 0.0) + weight

            unit_key = f"{unit_type}:{unit.get('unit_index')}:{unit.get('formula')}"
            urec = unit_weights.setdefault(
                unit_key,
                {
                    "unit": unit if unit else None,
                    "total_weight": 0.0,
                    "nearest_distance_A": None,
                    "n_weighted_contacts": 0,
                    "contact_atoms": set(),
                    "guest_atoms": set(),
                },
            )
            urec["total_weight"] += weight
            urec["n_weighted_contacts"] += 1
            urec["contact_atoms"].add(contact.get("framework_atom"))
            urec["guest_atoms"].add(guest_atom)
            if urec["nearest_distance_A"] is None or d < urec["nearest_distance_A"]:
                urec["nearest_distance_A"] = d

            wc = {
                "guest_atom": guest_atom,
                "framework_atom": contact.get("framework_atom"),
                "distance_A": d,
                "weight": weight,
                "site_type": site_type,
                "unit_type": unit_type,
                "unit_formula": unit.get("formula"),
                "sbu_type": unit.get("sbu_type"),
                "functional_tags": unit.get("functional_tags", []),
                "framework_mof_index": contact.get("framework_mof_index"),
                "framework_complex_index": contact.get("framework_complex_index"),
                "guest_complex_index": contact.get("guest_complex_index"),
            }
            weighted_contacts.append(wc)

        guest_atom_profiles: Dict[str, Any] = {}
        for guest_atom, profile in sorted(guest_profiles.items()):
            total = float(profile["total_weight"])
            guest_atom_profiles[guest_atom] = {
                "total_weight": total,
                "unit_type_weight_fraction": self._normalize_weights(profile["unit_type_weights"], total),
                "site_type_weight_fraction": self._normalize_weights(profile["site_type_weights"], total),
            }

        unit_contact_weight_summary = []
        for rec in unit_weights.values():
            unit_contact_weight_summary.append(
                {
                    "unit": rec["unit"],
                    "total_weight": float(rec["total_weight"]),
                    "weight_fraction": float(rec["total_weight"] / total_weight) if total_weight > 0.0 else None,
                    "nearest_distance_A": rec["nearest_distance_A"],
                    "n_weighted_contacts": rec["n_weighted_contacts"],
                    "contact_atoms": sorted(a for a in rec["contact_atoms"] if a),
                    "guest_atoms": sorted(a for a in rec["guest_atoms"] if a),
                }
            )
        unit_contact_weight_summary.sort(key=lambda x: x["total_weight"], reverse=True)
        weighted_contacts.sort(key=lambda x: x["weight"], reverse=True)

        return {
            "definition": "Distance-weighted chemistry fingerprint around the guest in this optimized pose.",
            "weighting": {
                "type": "linear_cutoff",
                "cutoff_A": cutoff_A,
                "formula": "w = max(0, 1 - distance_A / cutoff_A)",
            },
            "fingerprint": {
                "unit_type_weight_fraction": self._normalize_weights(unit_type_weights, total_weight),
                "site_type_weight_fraction": self._normalize_weights(site_type_weights, total_weight),
                "guest_atom_profiles": guest_atom_profiles,
                "unit_contact_weight_summary": unit_contact_weight_summary,
                "top_weighted_contacts": weighted_contacts[: max(1, int(top_k))],
            },
            "diagnostics": {
                "total_contact_weight": total_weight,
                "n_weighted_contacts": len(weighted_contacts),
                "n_weighted_contacts_with_chemistry_unit": n_with_unit,
                "chemistry_unit_coverage_fraction": (
                    float(n_with_unit / len(weighted_contacts)) if weighted_contacts else None
                ),
                "cutoff_A": cutoff_A,
            },
        }

    @staticmethod
    def _mean_position_from_indices(atoms, indices0: List[int]) -> Optional[np.ndarray]:
        if not indices0:
            return None
        return np.array([atoms[i].position for i in indices0], dtype=float).mean(axis=0)

    def _guest_internal_geometry(self, atoms, guest_indices0: List[int]) -> Dict[str, Any]:
        symbols = [atoms[i].symbol for i in guest_indices0]
        out: Dict[str, Any] = {
            "formula": self._formula_from_symbols(symbols),
            "n_atoms": len(guest_indices0),
            "symbols": symbols,
        }

        if symbols.count("C") == 1 and symbols.count("O") == 2 and len(guest_indices0) == 3:
            c_idx = next(i for i in guest_indices0 if atoms[i].symbol == "C")
            o_indices = [i for i in guest_indices0 if atoms[i].symbol == "O"]
            o1, o2 = o_indices
            v1 = atoms.get_distance(c_idx, o1, mic=True, vector=True)
            v2 = atoms.get_distance(c_idx, o2, mic=True, vector=True)
            axis = atoms.get_distance(o1, o2, mic=True, vector=True)
            angle = self._vector_angle_deg(v1, v2)
            if angle is not None:
                angle = 180.0 - angle
            out.update(
                {
                    "guest_type": "CO2",
                    "primary_axis": "O-O axis",
                    "C_O_bond_lengths_A": [
                        float(atoms.get_distance(c_idx, o1, mic=True)),
                        float(atoms.get_distance(c_idx, o2, mic=True)),
                    ],
                    "O_C_O_angle_deg": angle,
                    "axis_vector": [float(x) for x in axis],
                }
            )
        elif len(guest_indices0) >= 2:
            max_pair = None
            for i, a in enumerate(guest_indices0):
                for b in guest_indices0[i + 1:]:
                    d = float(atoms.get_distance(a, b, mic=True))
                    if max_pair is None or d > max_pair[0]:
                        max_pair = (d, a, b)
            if max_pair:
                _, a, b = max_pair
                axis = atoms.get_distance(a, b, mic=True, vector=True)
                out.update(
                    {
                        "guest_type": "molecular_guest",
                        "primary_axis": "longest intramolecular atom-pair axis",
                        "axis_atom_indices_complex": [int(a + 1), int(b + 1)],
                        "axis_vector": [float(x) for x in axis],
                    }
                )
        else:
            out.update({"guest_type": "single_atom_guest", "primary_axis": None})
        return out

    @staticmethod
    def _plane_from_positions(positions: np.ndarray) -> Optional[Dict[str, Any]]:
        if len(positions) < 3:
            return None
        centroid = positions.mean(axis=0)
        centered = positions - centroid
        try:
            _, s, vh = np.linalg.svd(centered, full_matrices=False)
        except Exception:
            return None
        if len(s) < 2 or float(s[1]) < 1e-6:
            return None
        normal = vh[-1]
        n = float(np.linalg.norm(normal))
        if n == 0.0:
            return None
        normal = normal / n
        return {
            "centroid": centroid,
            "normal": normal,
            "planarity_rms_A": float(np.sqrt(np.mean(np.dot(centered, normal) ** 2))),
        }

    def _unit_pose_descriptor(
        self,
        atoms,
        guest_indices0: List[int],
        unit: Dict[str, Any],
        mof_to_complex1: Dict[int, int],
        guest_geometry: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        mof_indices1 = unit.get("pose_mof_atom_indices") or unit.get("mof_atom_indices", []) or []
        complex_indices0 = [
            int(mof_to_complex1[int(i)] - 1)
            for i in mof_indices1
            if int(i) in mof_to_complex1
        ]
        if not complex_indices0:
            return None

        heavy_indices0 = [i for i in complex_indices0 if atoms[i].symbol != "H"]
        geom_indices0 = heavy_indices0 if len(heavy_indices0) >= 3 else complex_indices0
        guest_center = self._mean_position_from_indices(atoms, guest_indices0)
        if guest_center is None:
            return None

        local_positions = []
        try:
            from ase.geometry import find_mic

            for i in geom_indices0:
                delta = np.array(atoms[i].position, dtype=float) - guest_center
                mic_delta, _ = find_mic(delta, atoms.cell, pbc=atoms.pbc)
                local_positions.append(guest_center + mic_delta)
        except Exception:
            local_positions = [np.array(atoms[i].position, dtype=float) for i in geom_indices0]

        unit_positions = np.array(local_positions, dtype=float)
        unit_centroid = unit_positions.mean(axis=0)

        axis = np.array(guest_geometry.get("axis_vector") or [], dtype=float)
        axis_to_centroid = None
        guest_to_unit = unit_centroid - guest_center
        if axis.size == 3:
            axis_to_centroid = self._vector_angle_deg(axis, guest_to_unit)

        descriptor: Dict[str, Any] = {
            "unit": self._compact_chemistry_unit(unit),
            "unit_centroid_distance_from_guest_center_A": float(np.linalg.norm(guest_to_unit)),
            "guest_axis_to_unit_centroid_angle_deg": axis_to_centroid,
            "n_unit_atoms_used_for_geometry": len(geom_indices0),
        }

        if unit.get("unit_type") == "linker":
            plane = self._plane_from_positions(unit_positions)
            if plane is not None:
                rel = guest_center - plane["centroid"]
                offset = float(abs(np.dot(rel, plane["normal"])))
                in_plane = rel - np.dot(rel, plane["normal"]) * plane["normal"]
                axis_to_normal = self._vector_angle_deg(axis, plane["normal"]) if axis.size == 3 else None
                axis_to_plane = (90.0 - axis_to_normal) if axis_to_normal is not None else None
                descriptor["linker_relative_pose_descriptors"] = {
                    "guest_center_plane_offset_A": offset,
                    "guest_center_in_plane_distance_from_linker_centroid_A": float(np.linalg.norm(in_plane)),
                    "guest_axis_to_linker_normal_angle_deg": axis_to_normal,
                    "guest_axis_to_linker_plane_angle_deg": axis_to_plane,
                    "linker_planarity_rms_A": plane["planarity_rms_A"],
                }
        return descriptor

    def _build_guest_pose_degrees_of_freedom(
        self,
        atoms,
        guest_indices0: List[int],
        unit_by_mof_index: Dict[int, Dict[str, Any]],
        mof_to_complex1: Dict[int, int],
        local_binding_environment: Dict[str, Any],
        contacts_within_cutoff: List[Dict[str, Any]],
        top_k: int = 4,
    ) -> Dict[str, Any]:
        guest_geometry = self._guest_internal_geometry(atoms, guest_indices0)
        seen_units: Dict[str, Dict[str, Any]] = {}
        for contact in contacts_within_cutoff:
            mof_idx1 = contact.get("framework_mof_index")
            if mof_idx1 is None or int(mof_idx1) not in unit_by_mof_index:
                continue
            unit = dict(unit_by_mof_index[int(mof_idx1)])
            key = f"{unit.get('unit_type')}:{unit.get('unit_index')}:{unit.get('formula')}"
            rec = seen_units.setdefault(key, unit)
            rec.setdefault("pose_mof_atom_indices", [])
            rec["pose_mof_atom_indices"].append(int(mof_idx1))
        for unit in seen_units.values():
            unit["pose_mof_atom_indices"] = sorted(set(unit.get("pose_mof_atom_indices", [])))

        ranked_keys = []
        unit_summaries = (
            local_binding_environment.get("fingerprint", {}).get("unit_contact_weight_summary", [])
            if isinstance(local_binding_environment, dict)
            else []
        )
        for rec in unit_summaries:
            unit = rec.get("unit") or {}
            key = f"{unit.get('unit_type')}:{unit.get('unit_index')}:{unit.get('formula')}"
            if key in seen_units and key not in ranked_keys:
                ranked_keys.append(key)
        for key in seen_units:
            if key not in ranked_keys:
                ranked_keys.append(key)

        descriptors = []
        for key in ranked_keys[: max(1, int(top_k))]:
            desc = self._unit_pose_descriptor(
                atoms,
                guest_indices0,
                seen_units[key],
                mof_to_complex1,
                guest_geometry,
            )
            if desc:
                descriptors.append(desc)

        return {
            "definition": (
                "Geometry descriptors for guest translational/orientational/internal degrees of freedom "
                "relative to nearby MOF node/linker chemistry units. Unit geometry is computed from framework atoms "
                "that are actually within the contact cutoff, so these are local pose descriptors rather than global unit descriptors. "
                "These are evidence for LLM interpretation, not rule labels."
            ),
            "llm_interpretation_policy": [
                "Use numeric descriptors to describe the guest pose relative to node/linker units.",
                "Do not infer a global preferred binding site from a single optimized pose.",
                "For linker poses, use plane offset, in-plane distance, and axis-plane angles when available.",
                "For linear guests such as CO2, use internal angle/bond lengths and axis orientation together with contact fingerprint.",
            ],
            "guest_internal_geometry": guest_geometry,
            "unit_relative_pose_descriptors": descriptors,
        }

    def _co2_orientation_summary(
        self,
        atoms,
        guest_indices0: List[int],
        nearest_contact: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        guest_symbols = [atoms[i].symbol for i in guest_indices0]
        if guest_symbols.count("C") != 1 or guest_symbols.count("O") != 2 or len(guest_indices0) != 3:
            return {}

        c_idx = next(i for i in guest_indices0 if atoms[i].symbol == "C")
        o_indices = [i for i in guest_indices0 if atoms[i].symbol == "O"]
        o1, o2 = o_indices
        axis = atoms.get_distance(o1, o2, mic=True, vector=True)

        summary: Dict[str, Any] = {
            "guest_type": "CO2",
            "axis_definition": "O-O vector using minimum-image convention",
        }

        if nearest_contact:
            guest_atom_complex_index = int(nearest_contact["guest_complex_index"]) - 1
            fw_atom_complex_index = int(nearest_contact["framework_complex_index"]) - 1
            contact_vec = atoms.get_distance(
                c_idx,
                fw_atom_complex_index,
                mic=True,
                vector=True,
            )
            summary["axis_to_site_angle_deg"] = self._vector_angle_deg(axis, contact_vec)
            summary["nearest_contact_guest_atom"] = nearest_contact.get("guest_atom")
            summary["nearest_contact_framework_atom"] = nearest_contact.get("framework_atom")

            d_c = atoms.get_distance(c_idx, fw_atom_complex_index, mic=True)
            d_o_min = min(atoms.get_distance(o, fw_atom_complex_index, mic=True) for o in o_indices)
            if guest_atom_complex_index in o_indices and d_o_min + 0.15 < d_c:
                summary["configuration_label"] = "CO2 O-end contact near framework site"
            elif guest_atom_complex_index == c_idx and d_c + 0.15 < d_o_min:
                summary["configuration_label"] = "CO2 C-end contact near framework site"
            else:
                summary["configuration_label"] = "CO2 side-on or mixed contact near framework site"

        return summary

    @staticmethod
    def _atom_color_table() -> Dict[str, str]:
        return {
            "H": "#f2f2f2",
            "C": "#2f2f2f",
            "N": "#3056d3",
            "O": "#d62728",
            "F": "#2ca02c",
            "Mg": "#7fd13b",
            "Zn": "#7f7fbf",
            "Cu": "#b87333",
            "Zr": "#39a7a5",
            "Fe": "#b7410e",
            "default": "#9e9e9e",
        }

    def _build_binding_site_visualization(
        self,
        atoms,
        guest_indices0: List[int],
        contacts_sorted: List[Dict[str, Any]],
        output_dir: Path,
        mof_name: Optional[str],
        guest_name: Optional[str],
        radius_A: Optional[float] = None,
        top_contact_lines: int = 8,
    ) -> Dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        base = self._safe_path_token(f"{mof_name or 'mof'}_{guest_name or 'guest'}_binding_site")
        png_path = output_dir / f"{base}.png"
        xyz_path = output_dir / f"{base}_cluster.xyz"
        metadata_path = output_dir / f"{base}_visualization_metadata.json"

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.lines as mlines
            import matplotlib.pyplot as plt
            from ase import Atoms
            from ase.data import atomic_numbers, covalent_radii
            from ase.geometry import find_mic
        except Exception as exc:
            return {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "output_dir": str(output_dir),
            }

        if not guest_indices0:
            return {"status": "failed", "error": "no_guest_atoms", "output_dir": str(output_dir)}

        guest_positions = np.array([atoms[i].position for i in guest_indices0], dtype=float)
        guest_center = guest_positions.mean(axis=0)

        local_positions: Dict[int, np.ndarray] = {}
        for i in range(len(atoms)):
            delta = np.array(atoms[i].position, dtype=float) - guest_center
            try:
                mic_delta, _ = find_mic(delta, atoms.cell, pbc=atoms.pbc)
            except Exception:
                mic_delta = delta
            local_positions[i] = guest_center + np.array(mic_delta, dtype=float)

        render_scope = "full_complex" if radius_A is None else "local_cluster"
        if radius_A is None:
            selected = set(range(len(atoms)))
        else:
            selected = set(guest_indices0)
            for i, pos in local_positions.items():
                if i in selected:
                    continue
                if float(np.linalg.norm(pos - guest_center)) <= float(radius_A):
                    selected.add(i)

        contact_lines = []
        for contact in contacts_sorted[: max(0, int(top_contact_lines))]:
            try:
                gi0 = int(contact["guest_complex_index"]) - 1
                fi0 = int(contact["framework_complex_index"]) - 1
            except Exception:
                continue
            if gi0 < 0 or fi0 < 0 or gi0 >= len(atoms) or fi0 >= len(atoms):
                continue
            selected.add(gi0)
            selected.add(fi0)
            contact_lines.append(
                {
                    "guest_index": int(gi0 + 1),
                    "guest_atom": atoms[gi0].symbol,
                    "framework_index": int(fi0 + 1),
                    "framework_atom": atoms[fi0].symbol,
                    "distance_A": self._clean_numeric(contact.get("distance_A")),
                    "site_type": contact.get("site_type"),
                    "chemistry_unit": contact.get("chemistry_unit"),
                }
            )

        selected_indices = sorted(selected)
        if render_scope == "full_complex":
            structure_center = np.array(atoms.get_positions(), dtype=float).mean(axis=0)
            shifted = {i: np.array(atoms[i].position, dtype=float) - structure_center for i in selected_indices}
            try:
                cell = np.array(atoms.cell.array, dtype=float)
                origin = -structure_center
                cell_vertices = np.array(
                    [
                        origin,
                        origin + cell[0],
                        origin + cell[1],
                        origin + cell[2],
                        origin + cell[0] + cell[1],
                        origin + cell[0] + cell[2],
                        origin + cell[1] + cell[2],
                        origin + cell[0] + cell[1] + cell[2],
                    ],
                    dtype=float,
                )
            except Exception:
                cell_vertices = None
        else:
            shifted = {i: local_positions[i] - guest_center for i in selected_indices}
            cell_vertices = None
        cluster = Atoms(
            symbols=[atoms[i].symbol for i in selected_indices],
            positions=[shifted[i] for i in selected_indices],
            pbc=False,
        )

        try:
            write(str(xyz_path), cluster, format="xyz")
        except Exception:
            xyz_path = None

        colors = self._atom_color_table()
        used_elements = sorted({atoms[i].symbol for i in selected_indices})
        guest_set = set(guest_indices0)
        contact_framework_set = {line["framework_index"] - 1 for line in contact_lines}
        atom_colors = [colors.get(atoms[i].symbol, colors["default"]) for i in selected_indices]
        if render_scope == "full_complex":
            radius_values = {
                "guest": 0.46,
                "top_contact_framework": 0.34,
                "other_framework": 0.22,
            }
        else:
            radius_values = {
                "guest": 0.72,
                "top_contact_framework": 0.56,
                "other_framework": 0.34,
            }
        atom_radii = [
            radius_values["guest"] if i in guest_set else (
                radius_values["top_contact_framework"] if i in contact_framework_set else radius_values["other_framework"]
            )
            for i in selected_indices
        ]

        local_index = {complex_i0: cluster_i for cluster_i, complex_i0 in enumerate(selected_indices)}
        bonds = []
        cluster_positions = np.array(cluster.positions, dtype=float)
        cluster_symbols = cluster.get_chemical_symbols()
        for a in range(len(cluster)):
            for b in range(a + 1, len(cluster)):
                complex_a = selected_indices[a]
                complex_b = selected_indices[b]
                sym_a = cluster_symbols[a]
                sym_b = cluster_symbols[b]
                if complex_a in guest_set and complex_b in guest_set:
                    cutoff_scale = 1.35
                elif complex_a in guest_set or complex_b in guest_set:
                    continue
                else:
                    cutoff_scale = 1.25
                za = atomic_numbers.get(sym_a, 0)
                zb = atomic_numbers.get(sym_b, 0)
                if za <= 0 or zb <= 0:
                    continue
                cutoff = cutoff_scale * float(covalent_radii[za] + covalent_radii[zb])
                d = float(np.linalg.norm(cluster_positions[a] - cluster_positions[b]))
                if 0.25 < d <= max(cutoff, 0.9):
                    bonds.append(
                        {
                            "cluster_indices": [int(a), int(b)],
                            "complex_indices": [int(complex_a + 1), int(complex_b + 1)],
                            "symbols": [sym_a, sym_b],
                            "distance_A": d,
                            "is_guest_internal": bool(complex_a in guest_set and complex_b in guest_set),
                        }
                    )

        def rotation_matrix(rotation: str) -> np.ndarray:
            mat = np.eye(3)
            for token in rotation.split(","):
                token = token.strip()
                if not token:
                    continue
                axis = token[-1].lower()
                try:
                    angle = math.radians(float(token[:-1]))
                except Exception:
                    continue
                c = math.cos(angle)
                s = math.sin(angle)
                if axis == "x":
                    r = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)
                elif axis == "y":
                    r = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)
                elif axis == "z":
                    r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
                else:
                    continue
                mat = mat @ r
            return mat

        def hex_to_rgb(hex_color: str) -> np.ndarray:
            hex_color = hex_color.lstrip("#")
            return np.array([int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=float)

        def rgb_to_hex(rgb: np.ndarray) -> str:
            vals = np.clip(np.asarray(rgb, dtype=float), 0.0, 1.0)
            return "#" + "".join(f"{int(v * 255):02x}" for v in vals)

        def shade_color(hex_color: str, depth_value: float, z_min: float, z_max: float) -> str:
            t = 0.65 if z_max <= z_min else (float(depth_value) - float(z_min)) / (float(z_max) - float(z_min))
            base_rgb = hex_to_rgb(hex_color)
            shaded = base_rgb * (0.54 + 0.42 * t) + np.ones(3) * (0.10 + 0.08 * t)
            return rgb_to_hex(shaded)

        def project(points: np.ndarray, rot: np.ndarray) -> np.ndarray:
            return np.asarray(points, dtype=float) @ rot.T

        def draw_cell(ax, projected_cell: Optional[np.ndarray]) -> None:
            if projected_cell is None or len(projected_cell) != 8:
                return
            edges = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4), (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7)]
            for a, b in edges:
                ax.plot(
                    [projected_cell[a, 0], projected_cell[b, 0]],
                    [projected_cell[a, 1], projected_cell[b, 1]],
                    color="#8f8f8f",
                    linewidth=0.8,
                    alpha=0.55,
                    zorder=0,
                )

        fig, axes = plt.subplots(2, 2, figsize=(11.8, 10.0), constrained_layout=False)
        axes_flat = list(axes.ravel())
        views = [
            ("a-b view", "0x,0y,0z"),
            ("a-c view", "90x,0y,0z"),
            ("VESTA-like oblique", "18x,-34y,24z"),
            ("binding angle", "58x,-24y,-28z"),
        ]
        contact_pair_cluster_indices = []
        for line in contact_lines[: max(0, int(top_contact_lines))]:
            gi0 = int(line["guest_index"]) - 1
            fi0 = int(line["framework_index"]) - 1
            if gi0 in local_index and fi0 in local_index:
                contact_pair_cluster_indices.append((local_index[gi0], local_index[fi0]))

        for ax, (view_label, rotation) in zip(axes_flat, views):
            ax.set_facecolor("#fbfbf8")
            rot = rotation_matrix(rotation)
            projected = project(cluster_positions, rot)
            projected_cell = project(cell_vertices, rot) if cell_vertices is not None else None
            x = projected[:, 0]
            y = projected[:, 1]
            z = projected[:, 2]
            z_min = float(np.min(z)) if len(z) else 0.0
            z_max = float(np.max(z)) if len(z) else 1.0

            draw_cell(ax, projected_cell)

            sorted_bonds = sorted(
                bonds,
                key=lambda bond: float((z[bond["cluster_indices"][0]] + z[bond["cluster_indices"][1]]) / 2.0),
            )
            for bond in sorted_bonds:
                ia, ib = bond["cluster_indices"]
                is_guest_internal = bond["is_guest_internal"]
                z_mid = float((z[ia] + z[ib]) / 2.0)
                depth = 0.5 if z_max <= z_min else (z_mid - z_min) / (z_max - z_min)
                ax.plot(
                    [x[ia], x[ib]],
                    [y[ia], y[ib]],
                    color="#222222" if is_guest_internal else "#9a9a9a",
                    linewidth=(3.8 if is_guest_internal else 1.7) * (0.72 + 0.48 * depth),
                    alpha=0.96 if is_guest_internal else 0.58 + 0.30 * depth,
                    zorder=1,
                    solid_capstyle="round",
                )

            for ia, ib in contact_pair_cluster_indices:
                ax.plot(
                    [x[ia], x[ib]],
                    [y[ia], y[ib]],
                    color="#f0a000",
                    linewidth=1.0,
                    alpha=0.48,
                    linestyle=(0, (2.0, 2.5)),
                    zorder=1.8,
                )

            order = np.argsort(z)
            for cluster_i in order:
                complex_i0 = selected_indices[int(cluster_i)]
                is_guest = complex_i0 in guest_set
                is_contact_fw = complex_i0 in contact_framework_set
                depth = 0.5 if z_max <= z_min else (z[cluster_i] - z_min) / (z_max - z_min)
                radius_boost = 0.78 + 0.42 * depth
                size = (atom_radii[cluster_i] * 60.0 * radius_boost) ** 2
                face_color = shade_color(atom_colors[cluster_i], z[cluster_i], z_min, z_max)
                ax.scatter(
                    x[cluster_i] + 0.08,
                    y[cluster_i] - 0.08,
                    s=size * 1.06,
                    c="#000000",
                    edgecolors="none",
                    alpha=0.10 if is_guest or is_contact_fw else 0.07,
                    zorder=2.7 + depth,
                )
                ax.scatter(
                    x[cluster_i],
                    y[cluster_i],
                    s=size,
                    c=face_color,
                    edgecolors="#111111" if is_guest else ("#e39a00" if is_contact_fw else "#666666"),
                    linewidths=1.25 if is_guest or is_contact_fw else 0.35,
                    alpha=1.0 if is_guest or is_contact_fw else 0.88,
                    zorder=3.0 + depth,
                )
                if is_guest or is_contact_fw:
                    ax.scatter(
                        x[cluster_i] - 0.10 * radius_boost,
                        y[cluster_i] + 0.12 * radius_boost,
                        s=size * 0.16,
                        c="#ffffff",
                        edgecolors="none",
                        alpha=0.42,
                        zorder=3.4 + depth,
                    )

            ax.text(
                0.02,
                0.96,
                view_label,
                transform=ax.transAxes,
                fontsize=10,
                color="#333333",
                ha="left",
                va="top",
                bbox={"boxstyle": "round,pad=0.22", "facecolor": "#ffffff", "edgecolor": "#dddddd", "alpha": 0.82},
            )
            all_x = list(x)
            all_y = list(y)
            if projected_cell is not None:
                all_x.extend(projected_cell[:, 0].tolist())
                all_y.extend(projected_cell[:, 1].tolist())
            if all_x:
                span = max(float(np.max(all_x) - np.min(all_x)), float(np.max(all_y) - np.min(all_y)), 1.0)
                pad = max(1.4, 0.05 * span)
                ax.set_xlim(float(np.min(all_x) - pad), float(np.max(all_x) + pad))
                ax.set_ylim(float(np.min(all_y) - pad), float(np.max(all_y) + pad))
            ax.set_aspect("equal", adjustable="box")
            ax.set_axis_off()

        fig.suptitle(f"{mof_name or 'MOF'} / {guest_name or 'guest'} full complex snapshots", fontsize=15)

        handles = [
            mlines.Line2D([], [], color=colors.get(el, colors["default"]), marker="o", linestyle="None", markersize=8, label=f"{el}")
            for el in used_elements
        ]
        handles.extend(
            [
                mlines.Line2D([], [], color="#111111", marker="o", linestyle="None", markerfacecolor="#ffffff", markersize=10, label="guest atoms"),
                mlines.Line2D([], [], color="#e39a00", marker="o", linestyle="None", markerfacecolor="#ffffff", markersize=8, label="nearest framework atoms"),
                mlines.Line2D([], [], color="#9a9a9a", linestyle="-", linewidth=1.7, label="framework bonds"),
                mlines.Line2D([], [], color="#f0a000", linestyle=(0, (2.0, 2.5)), linewidth=1.0, label="guest-host contacts"),
                mlines.Line2D([], [], color="#8f8f8f", linestyle="-", linewidth=0.8, label="unit cell"),
            ]
        )
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            fontsize=8,
            frameon=False,
            ncol=min(7, max(3, len(handles))),
        )
        fig.text(
            0.02,
            0.015,
            "Snapshot-style render from ASE coordinates; JSON remains the source of truth for distances and atom indices.",
            fontsize=8,
            color="#555555",
        )
        plt.tight_layout(rect=(0.0, 0.07, 1.0, 0.94))
        fig.savefig(png_path, dpi=260, bbox_inches="tight", facecolor="#fbfbf8")
        plt.close(fig)

        metadata = {
            "status": "ok",
            "png": str(png_path),
            "cluster_xyz": str(xyz_path) if xyz_path else None,
            "metadata_json": str(metadata_path),
            "view_definition": {
                "center": "whole-structure centroid" if render_scope == "full_complex" else "guest atom centroid",
                "render_scope": render_scope,
                "radius_A": float(radius_A) if radius_A is not None else None,
                "coordinates": (
                    "original complex coordinates centered by the whole-structure centroid"
                    if render_scope == "full_complex"
                    else "minimum-image local coordinates shifted so guest center is near the origin"
                ),
            },
            "atom_color_legend": {el: colors.get(el, colors["default"]) for el in used_elements},
            "highlight_legend": {
                "guest_atoms": "large ASE-rendered atoms",
                "contact_framework_atoms": "medium ASE-rendered atoms selected from top contact distances",
                "framework_bonds": "gray lines inferred from covalent radii inside the local cluster",
                "guest_internal_bonds": "black lines inferred from covalent radii among guest atoms",
                "contact_distances": "orange dashed guide lines for top contacts; exact values are stored in contact_lines",
            },
            "rendering": {
                "renderer": "ASE-coordinate snapshot render with depth shading, unit cell, and covalent-radius bond overlay",
                "views": [{"label": label, "rotation": rotation} for label, rotation in views],
                "atom_radii": {
                    "guest": radius_values["guest"],
                    "top_contact_framework": radius_values["top_contact_framework"],
                    "other_framework": radius_values["other_framework"],
                },
                "bond_detection": {
                    "framework": "distance <= 1.25 * (covalent_radius_i + covalent_radius_j)",
                    "guest_internal": "distance <= 1.35 * (covalent_radius_i + covalent_radius_j)",
                    "guest_host": "not drawn as covalent bonds; use contact_lines metadata",
                    "n_bonds_drawn": int(len(bonds)),
                },
            },
            "drawn_bonds": bonds,
            "guest_complex_indices": [int(i + 1) for i in guest_indices0],
            "selected_complex_indices": [int(i + 1) for i in selected_indices],
            "n_selected_atoms": int(len(selected_indices)),
            "contact_lines": contact_lines,
            "vlm_prompt": (
                "Use this binding-site image together with the quantitative contact JSON. "
                "Describe the visible binding environment, but treat distances, atom indices, "
                "and node/linker assignments from JSON as the source of truth."
            ),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        return metadata

    def _analyze_binding_configuration_for_pair(
        self,
        mof_dir: Path,
        complex_dir: Path,
        guest_dir: Optional[Path] = None,
        mof_name: Optional[str] = None,
        guest_name: Optional[str] = None,
        mof_cif_path: Optional[Path] = None,
        chemistry_output_root: Optional[Path] = None,
        contact_cutoff_A: float = 3.5,
        top_k: int = 12,
    ) -> Dict[str, Any]:
        mof_structure = self._preferred_vasp_structure_path(mof_dir)
        complex_structure = self._preferred_vasp_structure_path(complex_dir)

        if mof_structure is None or complex_structure is None:
            return {
                "status": "missing_structure",
                "mof": mof_name,
                "guest": guest_name,
                "mof_dir": str(mof_dir),
                "complex_dir": str(complex_dir),
            }

        guest_identification: Dict[str, Any] = {}
        guest_structure = (
            self._preferred_vasp_structure_path(guest_dir)
            if guest_dir is not None
            else None
        )
        if guest_structure is not None:
            guest_identification = self._match_guest_by_internal_geometry(
                guest_structure,
                complex_structure,
            )

        if guest_identification.get("status") == "ok":
            identified_guest_indices1 = [
                int(index)
                for index in guest_identification["complex_indices"]
            ]
            mapping, guest_indices1 = self._match_atoms_by_distance(
                mof_structure,
                complex_structure,
                cutoff=0.5,
                excluded_complex_indices1=identified_guest_indices1,
            )
            guest_indices1 = identified_guest_indices1
        else:
            mapping, guest_indices1 = self._match_atoms_by_distance(
                mof_structure,
                complex_structure,
                cutoff=0.5,
            )
        complex_to_mof1 = {int(complex_idx): int(mof_idx) for mof_idx, complex_idx in mapping.items()}
        atoms_complex = read(complex_structure)

        framework_indices0 = sorted({int(v) - 1 for v in mapping.values()})
        guest_indices0 = sorted({int(v) - 1 for v in guest_indices1})
        guest_symbols = [atoms_complex[i].symbol for i in guest_indices0]

        chemistry_summary: Dict[str, Any] = {}
        unit_by_mof_index: Dict[int, Dict[str, Any]] = {}
        chemistry_source_path = mof_cif_path if mof_cif_path and mof_cif_path.exists() else None
        if chemistry_output_root is None:
            chemistry_output_root = mof_dir.parent / "binding_configuration_chemistry"
        chemistry_output_root.mkdir(parents=True, exist_ok=True)

        if chemistry_source_path is None:
            chemistry_source_path = chemistry_output_root / f"{self._safe_path_token(mof_name or mof_dir.name)}_mof_for_chemistry.cif"
            try:
                write(str(chemistry_source_path), read(mof_structure), format="cif")
            except Exception:
                chemistry_source_path = None

        if chemistry_source_path is not None:
            chemistry_summary = self._analyze_linker_chemistry_any(
                {
                    "query_text": "linker chemistry for binding configuration",
                    "mof": mof_name,
                    "cif_path": str(chemistry_source_path),
                    "linker_chemistry_output_dir": str(chemistry_output_root),
                }
            )
            try:
                unit_by_mof_index = self._chemistry_units_by_mof_index(
                    chemistry_summary,
                    chemistry_source_path,
                    mof_structure,
                )
            except Exception as exc:
                chemistry_summary.setdefault("warnings", []).append(
                    f"Failed to map chemistry units to VASP MOF atoms: {type(exc).__name__}: {exc}"
                )

        contacts: List[Dict[str, Any]] = []
        for gi0 in guest_indices0:
            for fi0 in framework_indices0:
                d = float(atoms_complex.get_distance(gi0, fi0, mic=True))
                mof_idx1 = complex_to_mof1.get(int(fi0 + 1))
                chemistry_unit = self._compact_chemistry_unit(
                    unit_by_mof_index.get(int(mof_idx1)) if mof_idx1 is not None else None
                )
                contacts.append(
                    {
                        "guest_complex_index": int(gi0 + 1),
                        "guest_atom": atoms_complex[gi0].symbol,
                        "framework_complex_index": int(fi0 + 1),
                        "framework_mof_index": mof_idx1,
                        "framework_atom": atoms_complex[fi0].symbol,
                        "distance_A": d,
                        "site_type": self._site_type_for_species(atoms_complex[fi0].symbol),
                        "chemistry_unit": chemistry_unit,
                    }
                )

        contacts_sorted = sorted(contacts, key=lambda x: x["distance_A"])
        nearest_contacts = contacts_sorted[: max(1, int(top_k))]
        contacts_within_cutoff = [c for c in contacts_sorted if c["distance_A"] <= contact_cutoff_A]
        nearest_contact = contacts_within_cutoff[0] if contacts_within_cutoff else None

        site_counts: Dict[str, int] = {}
        for c in contacts_within_cutoff:
            site_counts[c["site_type"]] = site_counts.get(c["site_type"], 0) + 1
        unit_contact_summary = self._summarize_unit_contacts(contacts_within_cutoff)
        local_binding_environment = self._build_local_binding_environment(
            contacts_sorted,
            cutoff_A=contact_cutoff_A,
            top_k=top_k,
        )
        guest_pose_degrees_of_freedom = self._build_guest_pose_degrees_of_freedom(
            atoms_complex,
            guest_indices0,
            unit_by_mof_index,
            mapping,
            local_binding_environment,
            contacts_within_cutoff,
            top_k=4,
        )

        configuration_label = "no close framework contact within cutoff"
        if nearest_contact:
            site_type = nearest_contact["site_type"]
            guest_atom = nearest_contact["guest_atom"]
            fw_atom = nearest_contact["framework_atom"]
            d = nearest_contact["distance_A"]
            configuration_label = f"{guest_atom}-to-{fw_atom} nearest contact at {d:.2f} A ({site_type})"
            unit = nearest_contact.get("chemistry_unit")
            if unit:
                unit_label = unit.get("sbu_type") or unit.get("formula") or unit.get("unit_type")
                configuration_label = (
                    f"{guest_atom}-to-{fw_atom} nearest contact at {d:.2f} A near "
                    f"{unit.get('unit_type')} {unit_label}"
                )

        orientation = self._co2_orientation_summary(atoms_complex, guest_indices0, nearest_contact)
        if orientation.get("configuration_label"):
            configuration_label = orientation["configuration_label"]
            if nearest_contact and nearest_contact.get("chemistry_unit"):
                unit = nearest_contact["chemistry_unit"]
                unit_label = unit.get("sbu_type") or unit.get("formula") or unit.get("unit_type")
                configuration_label = f"{configuration_label} at {unit.get('unit_type')} {unit_label}"

        nearest_binding_region = None
        if nearest_contact and nearest_contact.get("chemistry_unit"):
            nearest_binding_region = {
                "unit": nearest_contact["chemistry_unit"],
                "contact_atom": nearest_contact.get("framework_atom"),
                "guest_atom": nearest_contact.get("guest_atom"),
                "distance_A": nearest_contact.get("distance_A"),
                "framework_mof_index": nearest_contact.get("framework_mof_index"),
                "framework_complex_index": nearest_contact.get("framework_complex_index"),
            }
        elif unit_contact_summary:
            nearest_binding_region = unit_contact_summary[0]

        visualization = self._build_binding_site_visualization(
            atoms_complex,
            guest_indices0,
            contacts_sorted,
            output_dir=complex_dir / "binding_site_visualization",
            mof_name=mof_name,
            guest_name=guest_name,
            radius_A=None,
            top_contact_lines=min(8, len(contacts_within_cutoff) or len(nearest_contacts)),
        )

        return {
            "status": "ok",
            "mof": mof_name,
            "guest": guest_name,
            "mof_structure": str(mof_structure),
            "complex_structure": str(complex_structure),
            "chemistry_source_structure": str(chemistry_source_path) if chemistry_source_path else None,
            "contact_cutoff_A": contact_cutoff_A,
            "counts": {
                "n_framework_atoms": len(framework_indices0),
                "n_guest_atoms": len(guest_indices0),
                "n_contacts_within_cutoff": len(contacts_within_cutoff),
                "site_type_counts_within_cutoff": site_counts,
                "n_framework_atoms_mapped_to_chemistry_units": len(unit_by_mof_index),
                "n_unit_contact_groups_within_cutoff": len(unit_contact_summary),
            },
            "guest": {
                "formula_from_unmatched_atoms": self._formula_from_symbols(guest_symbols),
                "complex_indices": [int(i + 1) for i in guest_indices0],
                "symbols": guest_symbols,
            },
            "guest_identification": guest_identification,
            "nearest_contacts": nearest_contacts,
            "contacts_within_cutoff": contacts_within_cutoff,
            "nearest_binding_region": nearest_binding_region,
            "unit_contact_summary": unit_contact_summary,
            "local_binding_environment": local_binding_environment,
            "guest_pose_degrees_of_freedom": guest_pose_degrees_of_freedom,
            "binding_site_visualization": visualization,
            "mof_chemistry": {
                "status": chemistry_summary.get("status"),
                "summary_json": chemistry_summary.get("summary_json"),
                "output_dir": chemistry_summary.get("output_dir"),
                "n_structures": chemistry_summary.get("n_structures"),
                "warnings": chemistry_summary.get("warnings", []),
            },
            "configuration_label": configuration_label,
            "orientation": orientation,
            "limitations": [
                "Distance-based contact analysis does not prove a chemical bond.",
                "Classification depends on the final optimized geometry and the contact cutoff.",
                "MOF chemistry unit assignment depends on coordinate matching between the decomposition source and VASP MOF structure.",
            ],
        }

    def _analyze_binding_configurations_any(self, context: Dict[str, Any]) -> Dict[str, Any]:
        pairs = self._extract_binding_structure_pairs_any(context)
        out: Dict[str, Any] = {}
        for plan_name, info in pairs.items():
            out[plan_name] = self._analyze_binding_configuration_for_pair(
                mof_dir=info["mof_dir"],
                complex_dir=info["complex_dir"],
                guest_dir=info.get("guest_dir"),
                mof_name=info.get("mof"),
                guest_name=info.get("guest"),
                mof_cif_path=info.get("mof_cif_path"),
                chemistry_output_root=Path(context.get("work_dir", "working_dir"))
                / "binding_configuration_chemistry"
                / self._safe_path_token(plan_name),
            )
        return out

    def _build_binding_pdos_evidence_any(self, context: Dict[str, Any]) -> Dict[str, Any]:
        from analysis.binding_pdos import analyze_binding_pdos_artifact

        upstream = context.get("upstream_plans", {}) or {}
        output_root = (
            Path(context.get("work_dir", "working_dir"))
            / "binding_evidence"
            / "projected_dos"
        )
        binding_pairs = self._extract_binding_structure_pairs_any(context)
        out: Dict[str, Any] = {}

        for plan_name, plan_blob in upstream.items():
            if not isinstance(plan_blob, dict):
                continue

            roles: Dict[str, Dict[str, Any]] = {}
            for job_id, job in plan_blob.items():
                if not isinstance(job, dict):
                    continue
                result = (job.get("results") or {}).get("projected_dos")
                if not isinstance(result, dict):
                    continue
                role = str(
                    result.get("role")
                    or job.get("projected_dos_role")
                    or job.get("vasp_role")
                    or ""
                ).lower()
                if role in {"mof", "guest", "complex"}:
                    roles[role] = {
                        "job_id": str(job_id),
                        "job": job,
                        "result": result,
                    }

            if not roles:
                continue
            if "complex" not in roles:
                out[plan_name] = {
                    "status": "missing_required_roles",
                    "available_roles": sorted(roles),
                    "required_roles": ["complex"],
                }
                continue

            complex_result = roles["complex"]["result"]
            complex_dir = Path(
                str(
                    complex_result.get("vasp_dir")
                    or roles["complex"]["job"].get("vasp_dir")
                    or ""
                )
            )
            complex_structure = self._preferred_vasp_structure_path(complex_dir)
            artifact = complex_result.get("artifact")

            source_binding_plan = None
            binding_info = None
            mof_structure = None
            target_mof = roles["complex"]["job"].get("mof")
            target_guest = roles["complex"]["job"].get("guest")
            matching_pairs = [
                (binding_plan_name, info)
                for binding_plan_name, info in binding_pairs.items()
                if info.get("mof") == target_mof
                and info.get("guest") == target_guest
            ]
            if len(matching_pairs) == 1:
                source_binding_plan, binding_info = matching_pairs[0]
            elif len(binding_pairs) == 1:
                source_binding_plan, binding_info = next(
                    iter(binding_pairs.items())
                )

            if "mof" in roles:
                mof_result = roles["mof"]["result"]
                mof_dir = Path(
                    str(
                        mof_result.get("vasp_dir")
                        or roles["mof"]["job"].get("vasp_dir")
                        or ""
                    )
                )
                mof_structure = self._preferred_vasp_structure_path(mof_dir)

            if mof_structure is None and binding_info is not None:
                mof_structure = self._preferred_vasp_structure_path(
                    binding_info["mof_dir"]
                )

            if (
                mof_structure is None
                or complex_structure is None
                or not artifact
            ):
                out[plan_name] = {
                    "status": "missing_structure_or_pdos_artifact",
                    "mof_structure": (
                        str(mof_structure) if mof_structure is not None else None
                    ),
                    "complex_structure": (
                        str(complex_structure)
                        if complex_structure is not None
                        else None
                    ),
                    "complex_pdos_artifact": artifact,
                }
                continue

            try:
                atoms = read(complex_structure)
                guest_identification: Dict[str, Any] = {}
                guest_structure = None
                if (
                    binding_info is not None
                    and binding_info.get("guest_dir") is not None
                ):
                    guest_structure = self._preferred_vasp_structure_path(
                        binding_info["guest_dir"]
                    )
                if guest_structure is not None:
                    guest_identification = self._match_guest_by_internal_geometry(
                        guest_structure,
                        complex_structure,
                    )

                if guest_identification.get("status") == "ok":
                    guest_indices1 = [
                        int(index)
                        for index in guest_identification["complex_indices"]
                    ]
                    guest_indices0 = sorted(
                        {int(index1) - 1 for index1 in guest_indices1}
                    )
                    framework_indices0 = sorted(
                        set(range(len(atoms))) - set(guest_indices0)
                    )
                else:
                    mapping, guest_indices1 = self._match_atoms_by_distance(
                        mof_structure,
                        complex_structure,
                        cutoff=0.5,
                    )
                    framework_indices0 = sorted(
                        {
                            int(complex_index1) - 1
                            for complex_index1 in mapping.values()
                        }
                    )
                    guest_indices0 = sorted(
                        {int(index1) - 1 for index1 in guest_indices1}
                    )

                all_contacts: List[Dict[str, Any]] = []
                for guest_index0 in guest_indices0:
                    for framework_index0 in framework_indices0:
                        all_contacts.append(
                            {
                                "guest_complex_index": int(guest_index0 + 1),
                                "guest_atom": atoms[guest_index0].symbol,
                                "framework_complex_index": int(framework_index0 + 1),
                                "framework_atom": atoms[framework_index0].symbol,
                                "distance_A": float(
                                    atoms.get_distance(
                                        guest_index0,
                                        framework_index0,
                                        mic=True,
                                    )
                                ),
                            }
                        )
                all_contacts.sort(key=lambda row: row["distance_A"])
                contacts = [
                    row for row in all_contacts if row["distance_A"] <= 3.5
                ][:12]
                if not contacts:
                    contacts = all_contacts[:12]

                summary_path = (
                    output_root
                    / self._safe_path_token(plan_name)
                    / "binding_pdos_summary.json"
                )
                summary = analyze_binding_pdos_artifact(
                    str(artifact),
                    guest_indices1,
                    contacts,
                    output_path=str(summary_path),
                )
                summary.update(
                    {
                        "mof": (
                            roles["complex"]["job"].get("mof")
                            or roles["mof"]["job"].get("mof")
                        ),
                        "guest": roles["complex"]["job"].get("guest"),
                        "source_binding_plan": source_binding_plan,
                        "guest_identification": guest_identification,
                        "calculation_artifacts": {
                            role: {
                                "status": entry["result"].get("status"),
                                "artifact": entry["result"].get("artifact"),
                                "vasp_dir": entry["result"].get("vasp_dir"),
                            }
                            for role, entry in sorted(roles.items())
                        },
                    }
                )
                summary_path.write_text(
                    json.dumps(summary, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                out[plan_name] = summary
            except Exception as exc:
                out[plan_name] = {
                    "status": "analysis_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }

        return out

    @staticmethod
    def _safe_path_token(text: Any) -> str:
        token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text or "structure")).strip("_")
        return token[:120] or "structure"

    def _collect_mof_cif_paths_any(self, context: Dict[str, Any], max_files: int = 20) -> List[Path]:
        candidates: List[Path] = []
        target_mof = context.get("mof") or context.get("MOF")

        def add_path(value: Any) -> None:
            if not value:
                return
            path = Path(str(value))
            if path.suffix.lower() == ".cif" and path.exists():
                candidates.append(path)

        for key in ("cif_path", "mof_path", "optimized_mof_path"):
            add_path(context.get(key))

        for key in ("cif_paths", "mof_paths"):
            values = context.get(key)
            if isinstance(values, (list, tuple)):
                for value in values:
                    add_path(value)

        upstream = context.get("upstream_plans", {}) or {}
        for plan_blob in upstream.values():
            if not isinstance(plan_blob, dict):
                continue
            for job in plan_blob.values():
                if not isinstance(job, dict):
                    continue
                job_mof = job.get("mof") or job.get("MOF")
                if target_mof and job_mof and job_mof != target_mof:
                    continue
                for key in ("mof_path", "optimized_mof_path", "cif_path"):
                    add_path(job.get(key))
                    results = job.get("results", {}) or {}
                    if isinstance(results, dict):
                        add_path(results.get(key))

        out: List[Path] = []
        seen = set()
        for path in candidates:
            try:
                resolved = path.resolve()
            except Exception:
                resolved = path
            if resolved in seen:
                continue
            seen.add(resolved)
            out.append(path)
            if len(out) >= max_files:
                break
        return out

    @staticmethod
    def _default_mofstructure_python_path() -> Path:
        project_root = Path(__file__).resolve().parents[1]
        return project_root / "working_dir" / ".venvs" / "mofstructure" / "bin" / "python"

    def _mofstructure_python_path(self, context: Dict[str, Any]) -> Path:
        override = (
            context.get("mofstructure_python")
            or context.get("mofstructure_python_path")
            or os.environ.get("SIMMOF_MOFSTRUCTURE_PY")
        )
        return Path(str(override)) if override else self._default_mofstructure_python_path()

    def _analyze_linker_chemistry_any(self, context: Dict[str, Any]) -> Dict[str, Any]:
        cif_paths = self._collect_mof_cif_paths_any(context)
        output_root = Path(
            context.get("linker_chemistry_output_dir")
            or Path(context.get("work_dir", "working_dir")) / "linker_chemistry"
        )
        output_root.mkdir(parents=True, exist_ok=True)

        summary_path = output_root / "linker_chemistry_summary.json"
        if not cif_paths:
            summary = {
                "method": "linker_chemistry_analysis",
                "status": "no_cif_paths_found",
                "n_structures": 0,
                "output_dir": str(output_root),
                "summary_json": str(summary_path),
                "structures": [],
                "note": "No MOF CIF path was found in the current calculation context.",
            }
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            return summary

        worker = Path(__file__).resolve().with_name("linker_chemistry_worker.py")
        mofstructure_python = self._mofstructure_python_path(context)
        if not mofstructure_python.exists():
            return {
                "method": "linker_chemistry_analysis",
                "status": "missing_mofstructure_env",
                "n_structures": len(cif_paths),
                "output_dir": str(output_root),
                "summary_json": str(summary_path),
                "mofstructure_python": str(mofstructure_python),
                "cif_paths": [str(path) for path in cif_paths],
                "error": "Create the isolated environment or set SIMMOF_MOFSTRUCTURE_PY.",
            }

        payload_path = output_root / "linker_chemistry_input.json"
        payload = {
            "cif_paths": [str(path) for path in cif_paths],
            "output_root": str(output_root),
        }
        with payload_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        try:
            proc = subprocess.run(
                [str(mofstructure_python), str(worker), "--input", str(payload_path), "--output", str(summary_path)],
                cwd=str(Path(__file__).resolve().parents[1]),
                text=True,
                capture_output=True,
                check=False,
            )
            if proc.returncode != 0:
                return {
                    "method": "linker_chemistry_analysis",
                    "status": "worker_error",
                    "n_structures": len(cif_paths),
                    "output_dir": str(output_root),
                    "summary_json": str(summary_path),
                    "mofstructure_python": str(mofstructure_python),
                    "stderr": proc.stderr[-4000:],
                    "stdout": proc.stdout[-2000:],
                }
            with summary_path.open(encoding="utf-8") as f:
                summary = json.load(f)
            summary["mofstructure_python"] = str(mofstructure_python)
            return summary
        except Exception as exc:
            return {
                "method": "linker_chemistry_analysis",
                "status": "worker_exception",
                "n_structures": len(cif_paths),
                "output_dir": str(output_root),
                "summary_json": str(summary_path),
                "mofstructure_python": str(mofstructure_python),
                "error": f"{type(exc).__name__}: {exc}",
            }

    @staticmethod
    def _default_mofstructure_analysis_python_path() -> Path:
        project_root = Path(__file__).resolve().parents[1]
        return (
            project_root
            / "working_dir"
            / ".venvs"
            / "mofstructure_analysis"
            / "bin"
            / "python"
        )

    def _mofstructure_analysis_python_path(self, context: Dict[str, Any]) -> Path:
        override = (
            context.get("mofstructure_analysis_python")
            or context.get("mofstructure_analysis_python_path")
            or os.environ.get("SIMMOF_MOFSTRUCTURE_ANALYSIS_PY")
        )
        return (
            Path(str(override))
            if override
            else self._default_mofstructure_analysis_python_path()
        )

    def _analyze_open_metal_sites_any(self, context: Dict[str, Any]) -> Dict[str, Any]:
        cif_paths = self._collect_mof_cif_paths_any(context)
        output_root = Path(
            context.get("open_metal_site_output_dir")
            or Path(context.get("work_dir", "working_dir"))
            / "mof_structure_chemistry"
            / "open_metal_sites"
        )
        output_root.mkdir(parents=True, exist_ok=True)
        summary_path = output_root / "open_metal_site_summary.json"
        if not cif_paths:
            summary = {
                "method": "open_metal_site_analysis",
                "status": "no_cif_paths_found",
                "n_structures": 0,
                "output_dir": str(output_root),
                "summary_json": str(summary_path),
                "structures": [],
            }
            with summary_path.open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, ensure_ascii=False)
            return summary

        worker = Path(__file__).resolve().with_name("open_metal_site_worker.py")
        python_path = self._mofstructure_analysis_python_path(context)
        if not python_path.exists():
            return {
                "method": "open_metal_site_analysis",
                "status": "missing_mofstructure_analysis_env",
                "n_structures": len(cif_paths),
                "cif_paths": [str(path) for path in cif_paths],
                "output_dir": str(output_root),
                "summary_json": str(summary_path),
                "mofstructure_analysis_python": str(python_path),
                "error": (
                    "Install the dedicated Python >=3.10 environment or set "
                    "SIMMOF_MOFSTRUCTURE_ANALYSIS_PY."
                ),
            }

        payload_path = output_root / "open_metal_site_input.json"
        payload = {
            "cif_paths": [str(path) for path in cif_paths],
            "output_root": str(output_root),
        }
        with payload_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        try:
            proc = subprocess.run(
                [
                    str(python_path),
                    str(worker),
                    "--input",
                    str(payload_path),
                    "--output",
                    str(summary_path),
                ],
                cwd=str(Path(__file__).resolve().parents[1]),
                text=True,
                capture_output=True,
                check=False,
            )
            if proc.returncode != 0:
                return {
                    "method": "open_metal_site_analysis",
                    "status": "worker_error",
                    "n_structures": len(cif_paths),
                    "output_dir": str(output_root),
                    "summary_json": str(summary_path),
                    "mofstructure_analysis_python": str(python_path),
                    "stderr": proc.stderr[-4000:],
                    "stdout": proc.stdout[-2000:],
                }
            with summary_path.open(encoding="utf-8") as handle:
                summary = json.load(handle)
            summary["mofstructure_analysis_python"] = str(python_path)
            return summary
        except Exception as exc:
            return {
                "method": "open_metal_site_analysis",
                "status": "worker_exception",
                "n_structures": len(cif_paths),
                "output_dir": str(output_root),
                "summary_json": str(summary_path),
                "mofstructure_analysis_python": str(python_path),
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _analyze_linker_functional_groups_any(
        self,
        context: Dict[str, Any],
        linker_chemistry_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        output_root = Path(
            context.get("linker_functional_group_output_dir")
            or Path(context.get("work_dir", "working_dir"))
            / "mof_structure_chemistry"
            / "linker_functional_groups"
        )
        output_root.mkdir(parents=True, exist_ok=True)
        summary_path = output_root / "linker_functional_group_summary.json"
        structures_out: List[Dict[str, Any]] = []
        global_counts: Dict[str, int] = {}
        global_linker_counts: Dict[str, int] = {}

        for structure in linker_chemistry_summary.get("structures", []) or []:
            group_counts: Dict[str, int] = {}
            linker_counts: Dict[str, int] = {}
            linkers = []
            ligand_records = structure.get("ligands", []) or []
            functional_units = ligand_records or (structure.get("linkers", []) or [])
            representation = (
                "metal_ligand_bond_cut_ligands"
                if ligand_records
                else "sbu_linker_fragments"
            )
            for linker in functional_units:
                fingerprint = linker.get("functional_group_fingerprint", {}) or {}
                features = fingerprint.get("features", {}) or {}
                present_groups = fingerprint.get("present_groups", []) or []
                for group, feature in features.items():
                    count = int(feature.get("count") or 0)
                    if count:
                        group_counts[group] = group_counts.get(group, 0) + count
                        global_counts[group] = global_counts.get(group, 0) + count
                for group in present_groups:
                    linker_counts[group] = linker_counts.get(group, 0) + 1
                    global_linker_counts[group] = (
                        global_linker_counts.get(group, 0) + 1
                    )
                linkers.append(
                    {
                        "index": linker.get("index"),
                        "formula": linker.get("formula"),
                        "smiles": linker.get("smiles"),
                        "inchikey": linker.get("inchikey"),
                        "source_atom_indices": linker.get(
                            "source_atom_indices",
                            [],
                        ),
                        "functional_group_fingerprint": fingerprint,
                    }
                )
            structures_out.append(
                {
                    "status": structure.get("status"),
                    "mof": structure.get("mof"),
                    "cif_path": structure.get("cif_path"),
                    "n_linkers": len(linkers),
                    "linker_representation": representation,
                    "functional_group_match_counts": dict(
                        sorted(group_counts.items())
                    ),
                    "linker_count_with_group": dict(
                        sorted(linker_counts.items())
                    ),
                    "linkers": linkers,
                }
            )

        if not linker_chemistry_summary:
            status = "missing_linker_chemistry_dependency"
        elif not structures_out:
            status = "no_linker_structures"
        elif any(
            not linker.get("functional_group_fingerprint")
            for structure in (linker_chemistry_summary.get("structures", []) or [])
            for linker in (
                (structure.get("ligands", []) or [])
                or (structure.get("linkers", []) or [])
            )
        ):
            status = "partial_missing_fingerprints"
        else:
            status = "ok"
        summary = {
            "method": "linker_functional_group_analysis",
            "status": status,
            "n_structures": len(structures_out),
            "engine": "RDKit named SMARTS substructure fingerprint",
            "output_dir": str(output_root),
            "summary_json": str(summary_path),
            "global_functional_group_match_counts": dict(
                sorted(global_counts.items())
            ),
            "global_linker_count_with_group": dict(
                sorted(global_linker_counts.items())
            ),
            "structures": structures_out,
            "interpretation_note": (
                "Match counts are molecular substructure counts in each "
                "disconnected linker SMILES, not framework-wide atom counts."
            ),
        }
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        return summary

    def _analyze_pore_surface_chemistry_any(
        self,
        context: Dict[str, Any],
        linker_chemistry_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        cif_paths = self._collect_mof_cif_paths_any(context)
        output_root = Path(
            context.get("pore_surface_chemistry_output_dir")
            or Path(context.get("work_dir", "working_dir"))
            / "mof_structure_chemistry"
            / "pore_surface_chemistry"
        )
        analysis_options = context.get("analysis_options", {}) or {}
        try:
            from analysis.pore_surface_chemistry import (
                run_pore_surface_chemistry_analysis,
            )

            return run_pore_surface_chemistry_analysis(
                cif_paths=cif_paths,
                output_dir=output_root,
                chemistry_summary=linker_chemistry_summary,
                probe_radius_A=float(
                    analysis_options.get("pore_surface_probe_radius_A", 1.86)
                ),
                samples_per_atom=int(
                    analysis_options.get(
                        "pore_surface_samples_per_atom",
                        256,
                    )
                ),
                spatial_bins=int(
                    analysis_options.get("pore_surface_spatial_bins", 8)
                ),
            )
        except Exception as exc:
            return {
                "method": "pore_surface_chemistry_analysis",
                "status": "analysis_error",
                "n_structures": len(cif_paths),
                "cif_paths": [str(path) for path in cif_paths],
                "output_dir": str(output_root),
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _extract_zeopp_summaries_any(self, context: Dict[str, Any]) -> Dict[str, Any]:
        upstream = context.get("upstream_plans", {}) or {}
        out: Dict[str, Any] = {}

        for plan_name, plan_blob in upstream.items():
            if not isinstance(plan_blob, dict):
                continue

            for job_id, job in plan_blob.items():
                if not isinstance(job, dict):
                    continue

                mof = job.get("mof") or job.get("MOF")
                prop = job.get("property") or job.get("property_name") or job.get("simulation_property")
                prop = re.sub(r"[\s-]+", "_", str(prop or "").strip().lower())
                res = job.get("results", {}) or {}
                zeopp = res.get("zeopp", {}) or {}
                raw = zeopp.get("raw", {}) or {}

                if not mof or not prop or not raw:
                    continue

                out.setdefault(mof, {})

                if prop == "pore_size_distribution":
                    out[mof]["pore_size_distribution"] = {
                        "summary": raw.get("summary", {}),
                        "metadata": raw.get("metadata", {}),
                        "histogram": raw.get("histogram", {}),
                        "file": raw.get("file"),
                        "note": raw.get(
                            "note",
                            "Zeo++ pore size distribution (-ha -psd).",
                        ),
                    }

                elif prop == "pore_volume":
                    out[mof]["pore_volume"] = {
                        "AV_cm3_g": raw.get("AV_cm3_g"),
                        "AV_volume_fraction": raw.get("AV_Volume_fraction"),
                        "AV_A3": raw.get("AV_A3"),
                        "probe_radius_A": (job.get("zeopp_info") or {}).get("probe_radius"),
                        "note": "Zeo++ accessible volume (-ha -vol).",
                    }

                elif prop == "pore_limiting_diameter":
                    out[mof]["pore_diameter"] = {
                        "PLD_free_sphere_A": raw.get("free_sphere"),
                        "LCD_included_sphere_A": raw.get("included_sphere"),
                        "LCD_along_free_path_A": raw.get("included_sphere_along_free_path"),
                        "note": "Zeo++ pore diameters (-ha -res). free_sphere is commonly used as PLD.",
                    }

                elif prop == "largest_cavity_diameter":

                    out[mof]["largest_cavity_diameter"] = {
                        "LCD_included_sphere_A": raw.get("included_sphere"),
                        "note": "Zeo++ largest cavity diameter (included_sphere).",
                    }

                elif prop == "surface_area":
                    out[mof]["surface_area"] = {
                        "ASA_m2_g": raw.get("ASA_m2_g"),
                        "NASA_m2_g": raw.get("NASA_m2_g"),
                        "note": "Zeo++ surface area if available.",
                    }

        return out

    @staticmethod
    def _merge_zeopp_summaries(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base or {})
        for mof, block in (extra or {}).items():
            if mof.startswith("_"):
                merged[mof] = block
                continue
            if not isinstance(block, dict):
                continue
            merged.setdefault(mof, {})
            for key, value in block.items():
                if key not in merged[mof] or not merged[mof].get(key):
                    merged[mof][key] = value
        return merged

    def _find_raspa_cif_for_zeopp(self, job: Dict[str, Any], mof: str) -> Optional[Path]:
        candidates: List[Path] = []
        for key in ("cif_path", "structure_path", "mof_cif_path"):
            value = job.get(key) or (job.get("results", {}) or {}).get(key)
            if value:
                candidates.append(Path(str(value)))

        work_dir_value = job.get("work_dir") or (job.get("results", {}) or {}).get("work_dir")
        if work_dir_value:
            work_dir = Path(str(work_dir_value))
            candidates.append(work_dir / f"{mof}.cif")
            candidates.extend(sorted(work_dir.glob("*.cif")))
            candidates.extend(sorted((work_dir / "Movies" / "System_0").glob("Framework_0_initial*_VASP.cif")))
            candidates.extend(sorted((work_dir / "Movies" / "System_0").glob("Framework_0_final*_VASP.cif")))

        for path in candidates:
            if path.exists() and path.is_file():
                return path
        return None

    def _run_zeopp_command_for_analysis(
        self,
        command: List[str],
        work_dir: Path,
        timeout_sec: int = 900,
    ) -> Dict[str, Any]:
        try:
            proc = subprocess.run(
                command,
                cwd=str(work_dir),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_sec,
                check=False,
            )
        except Exception as exc:
            return {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "command": command,
                "work_dir": str(work_dir),
            }

        return {
            "status": "ok" if proc.returncode == 0 else "run_failed",
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "command": command,
            "work_dir": str(work_dir),
        }

    def _compute_zeopp_descriptors_for_cif(
        self,
        mof: str,
        cif_path: Path,
        output_root: Path,
        probe_radius_A: float = 1.2,
        surface_probe_radius_A: float = 1.86,
        num_samples: int = 50000,
    ) -> Dict[str, Any]:
        try:
            from config import ZEOPP_BIN
            from output.zeopp_output import ZeoppOutputAgent
        except Exception as exc:
            return {"status": "import_failed", "mof": mof, "error": str(exc)}

        safe_mof = self._safe_path_token(mof)
        work_dir = output_root / safe_mof
        work_dir.mkdir(parents=True, exist_ok=True)
        local_cif = work_dir / f"{safe_mof}.cif"
        try:
            if cif_path.resolve() != local_cif.resolve():
                shutil.copy2(cif_path, local_cif)
        except Exception as exc:
            return {
                "status": "copy_failed",
                "mof": mof,
                "source_cif": str(cif_path),
                "error": str(exc),
            }

        zeopp_bin = str(ZEOPP_BIN)
        runs = {
            "pore_diameter": self._run_zeopp_command_for_analysis(
                [zeopp_bin, "-ha", "-res", str(local_cif)],
                work_dir,
            ),
            "pore_volume": self._run_zeopp_command_for_analysis(
                [zeopp_bin, "-ha", "-vol", str(probe_radius_A), str(probe_radius_A), str(int(num_samples)), str(local_cif)],
                work_dir,
            ),
            "surface_area": self._run_zeopp_command_for_analysis(
                [
                    zeopp_bin,
                    "-ha",
                    "-sa",
                    str(surface_probe_radius_A),
                    str(surface_probe_radius_A),
                    str(int(num_samples)),
                    str(local_cif),
                ],
                work_dir,
            ),
        }

        descriptor: Dict[str, Any] = {
            "status": "ok",
            "mof": mof,
            "source_cif": str(cif_path),
            "work_dir": str(work_dir),
            "probe_settings": {
                "pore_volume_probe_radius_A": probe_radius_A,
                "surface_area_probe_radius_A": surface_probe_radius_A,
                "num_samples": int(num_samples),
            },
            "runs": runs,
        }

        if runs["pore_diameter"].get("status") == "ok":
            try:
                raw = ZeoppOutputAgent._read_res_file(safe_mof, str(work_dir))
                descriptor["pore_diameter"] = {
                    "PLD_free_sphere_A": raw.get("free_sphere"),
                    "LCD_included_sphere_A": raw.get("included_sphere"),
                    "LCD_along_free_path_A": raw.get("included_sphere_along_free_path"),
                    "note": "Zeo++ pore diameters (-ha -res). free_sphere is commonly used as PLD.",
                }
                descriptor["largest_cavity_diameter"] = {
                    "LCD_included_sphere_A": raw.get("included_sphere"),
                    "note": "Zeo++ largest cavity diameter (included_sphere).",
                }
            except Exception as exc:
                descriptor.setdefault("warnings", []).append(f"Failed to parse .res: {exc}")

        if runs["pore_volume"].get("status") == "ok":
            try:
                raw = ZeoppOutputAgent._read_vol_file(safe_mof, str(work_dir))
                descriptor["pore_volume"] = {
                    "AV_cm3_g": raw.get("AV_cm3_g"),
                    "AV_volume_fraction": raw.get("AV_Volume_fraction"),
                    "AV_A3": raw.get("AV_A3"),
                    "probe_radius_A": probe_radius_A,
                    "note": "Zeo++ accessible volume (-ha -vol).",
                }
            except Exception as exc:
                descriptor.setdefault("warnings", []).append(f"Failed to parse .vol: {exc}")

        if runs["surface_area"].get("status") == "ok":
            try:
                raw = ZeoppOutputAgent._read_sa_file(safe_mof, str(work_dir))
                descriptor["surface_area"] = {
                    "ASA_m2_g": raw.get("ASA_m2_g"),
                    "ASA_m2_cm3": raw.get("ASA_m2_cm3"),
                    "ASA_A2": raw.get("ASA_A2"),
                    "probe_radius_A": surface_probe_radius_A,
                    "note": "Zeo++ surface area (-ha -sa).",
                }
            except Exception as exc:
                descriptor.setdefault("warnings", []).append(f"Failed to parse .sa: {exc}")

        parsed_keys = [k for k in ("pore_diameter", "largest_cavity_diameter", "pore_volume", "surface_area") if k in descriptor]
        if not parsed_keys:
            descriptor["status"] = "no_descriptors_parsed"
        return descriptor

    def _auto_compute_zeopp_for_uptake_context(
        self,
        context: Dict[str, Any],
        zeopp_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        upstream = context.get("upstream_plans", {}) or {}
        analysis_options = context.get("analysis_options", {}) or {}
        if analysis_options.get("auto_zeopp_for_uptake") is False:
            return {}

        output_root_value = analysis_options.get("auto_zeopp_output_dir")
        output_root = Path(str(output_root_value)) if output_root_value else None
        out: Dict[str, Any] = {}
        metadata: Dict[str, Any] = {
            "status": "ok",
            "trigger": "missing Zeo++ descriptors for RASPA uptake analysis",
            "jobs": {},
        }

        for plan_blob in upstream.values():
            if not isinstance(plan_blob, dict):
                continue
            for job_id, job in plan_blob.items():
                if not isinstance(job, dict):
                    continue
                if job.get("agent") != "RASPAAgent" or job.get("property") != "uptake":
                    continue

                mof = job.get("mof") or job.get("MOF")
                if not mof:
                    continue
                existing = (zeopp_summary or {}).get(mof, {}) or {}
                required = ("pore_diameter", "pore_volume", "surface_area")
                if all(existing.get(key) for key in required):
                    continue

                cif_path = self._find_raspa_cif_for_zeopp(job, str(mof))
                if cif_path is None:
                    metadata["jobs"][str(job_id)] = {
                        "mof": mof,
                        "status": "missing_cif",
                    }
                    continue

                if output_root is None:
                    work_dir_value = job.get("work_dir") or (job.get("results", {}) or {}).get("work_dir")
                    base_dir = Path(str(work_dir_value)).parent if work_dir_value else Path.cwd()
                    output_root = base_dir / "analysis_agent_zeopp"

                descriptor = self._compute_zeopp_descriptors_for_cif(
                    str(mof),
                    cif_path,
                    output_root,
                    probe_radius_A=float(analysis_options.get("zeopp_pore_volume_probe_radius_A", 1.2)),
                    surface_probe_radius_A=float(analysis_options.get("zeopp_surface_probe_radius_A", 1.86)),
                    num_samples=int(analysis_options.get("zeopp_num_samples", 50000)),
                )
                metadata["jobs"][str(job_id)] = {
                    "mof": mof,
                    "status": descriptor.get("status"),
                    "source_cif": str(cif_path),
                    "work_dir": descriptor.get("work_dir"),
                    "warnings": descriptor.get("warnings", []),
                }

                out.setdefault(str(mof), {})
                for key in ("pore_diameter", "largest_cavity_diameter", "pore_volume", "surface_area"):
                    if descriptor.get(key):
                        out[str(mof)][key] = descriptor[key]

        if output_root is not None:
            metadata["output_root"] = str(output_root)
            try:
                output_root.mkdir(parents=True, exist_ok=True)
                metadata_path = output_root / "auto_zeopp_summary.json"
                metadata_path.write_text(json.dumps({"metadata": metadata, "zeopp_summary": out}, indent=2), encoding="utf-8")
                metadata["summary_json"] = str(metadata_path)
            except Exception as exc:
                metadata.setdefault("warnings", []).append(f"Failed to write auto Zeo++ summary: {exc}")

        if out:
            out["_auto_zeopp"] = metadata
        return out

    def _extract_diffusivity_summaries_any(self, context: Dict[str, Any]) -> Dict[str, Any]:
        upstream = context.get("upstream_plans", {}) or {}
        out: Dict[str, Any] = {}

        for plan_name, plan_blob in upstream.items():
            if not isinstance(plan_blob, dict):
                continue

            for job_id, job in plan_blob.items():
                if not isinstance(job, dict):
                    continue

                mof = job.get("mof") or job.get("MOF")
                prop = job.get("property") or job.get("property_name") or job.get("simulation_property")
                guest = job.get("guest") or "guest"

                res = job.get("results", {}) or {}
                d = res.get("diffusivity", {}) or {}


                if prop != "diffusivity" or not mof or not d:
                    continue

                out.setdefault(mof, {})
                out[mof].setdefault(guest, {})
                out[mof][guest][str(job_id)] = {
                    "D_m2_per_s": d.get("D_m2_per_s"),
                    "r2": d.get("r2"),
                    "slope_A2_per_fs": d.get("slope_A2_per_fs"),
                    "std_err_slope": d.get("std_err_slope"),
                    "p_value": d.get("p_value"),
                    "time_range_fs": d.get("time_range_fs"),
                    "note": "From MSD linear fit (Einstein relation).",
                }

        return out

    def _extract_msd_summaries_any(self, context: Dict[str, Any]) -> Dict[str, Any]:
        upstream = context.get("upstream_plans", {}) or {}
        out: Dict[str, Any] = {}

        for plan_blob in upstream.values():
            if not isinstance(plan_blob, dict):
                continue
            for job_id, job in plan_blob.items():
                if not isinstance(job, dict) or job.get("agent") != "LAMMPSAgent":
                    continue

                results = job.get("results", {}) or {}
                msd = results.get("msd") or {}
                if not isinstance(msd, dict) or not msd:
                    continue

                mof = job.get("mof") or job.get("MOF") or "unknown_mof"
                guest = job.get("guest") or "guest"
                steps = msd.get("steps") or []
                values = msd.get("msd_A2") or []
                out.setdefault(str(mof), {})
                out[str(mof)].setdefault(str(guest), {})
                out[str(mof)][str(guest)][str(job_id)] = {
                    "summary": msd.get("summary") or {},
                    "n_points": min(len(steps), len(values)),
                    "source_file": msd.get("file"),
                    "work_dir": job.get("work_dir") or results.get("work_dir"),
                    "note": "MSD curve summary; diffusivity requires a justified linear-time fit.",
                }

        return out

    def _analyze_lammps_trajectories_any(self, context: Dict[str, Any]) -> Dict[str, Any]:
        upstream = context.get("upstream_plans", {}) or {}
        out: Dict[str, Any] = {}

        try:
            from analysis.lammps_trajectory import run_lammps_trajectory_analysis
        except Exception as exc:
            return {"status": "import_failed", "error": str(exc)}

        chemistry_summary_path = None
        linker_blob = ((context.get("analysis") or {}).get("linker_chemistry") or {})
        if isinstance(linker_blob, dict):
            chemistry_summary_path = linker_blob.get("summary_json")

        for plan_name, plan_blob in upstream.items():
            if not isinstance(plan_blob, dict):
                continue
            for job_id, job in plan_blob.items():
                if not isinstance(job, dict):
                    continue
                if job.get("agent") != "LAMMPSAgent":
                    continue

                work_dir_value = job.get("work_dir") or (job.get("results", {}) or {}).get("work_dir")
                if not work_dir_value:
                    continue
                work_dir = Path(str(work_dir_value))
                if not (work_dir / "traj.lammpstrj").exists():
                    continue

                mof = job.get("mof") or job.get("MOF") or work_dir.name.split("_")[0]
                guest = job.get("guest") or "guest"
                try:
                    result = run_lammps_trajectory_analysis(
                        work_dir=work_dir,
                        output_dir=work_dir / "lammps_trajectory_analysis",
                        max_frames=400,
                        chemistry_summary=Path(str(chemistry_summary_path)) if chemistry_summary_path else None,
                    )
                except Exception as exc:
                    out.setdefault(mof, {})
                    out[mof].setdefault(guest, {})
                    out[mof][guest][str(job_id)] = {
                        "status": "failed",
                        "work_dir": str(work_dir),
                        "error": str(exc),
                    }
                    continue

                out.setdefault(mof, {})
                out[mof].setdefault(guest, {})
                out[mof][guest][str(job_id)] = result

        return out

    def _analyze_lammps_diffusion_meta_any(self, context: Dict[str, Any]) -> Dict[str, Any]:
        upstream = context.get("upstream_plans", {}) or {}
        inputs = []
        for plan_name, plan_blob in upstream.items():
            if not isinstance(plan_blob, dict):
                continue
            for job in plan_blob.values():
                if not isinstance(job, dict) or job.get("agent") != "LAMMPSAgent":
                    continue
                work_dir_value = job.get("work_dir") or (job.get("results", {}) or {}).get("work_dir")
                if work_dir_value:
                    work_dir = Path(str(work_dir_value))
                    has_trajectory_summary = any(
                        work_dir.rglob("*_lammps_trajectory_summary.json")
                    ) or any(
                        work_dir.rglob("combined_lammps_trajectory_analysis_summary.json")
                    )
                    if has_trajectory_summary:
                        inputs.append(work_dir)

        if not inputs:
            return {}

        try:
            from analysis.lammps_diffusion_meta import run_lammps_diffusion_meta_analysis
        except Exception as exc:
            return {"status": "import_failed", "error": str(exc)}

        output_root = Path(str(context.get("working_dir") or context.get("work_dir") or "working_dir"))
        output_dir = output_root / "analysis_lammps_diffusion_meta"
        try:
            return run_lammps_diffusion_meta_analysis(inputs, output_dir)
        except Exception as exc:
            return {"status": "failed", "error": str(exc), "output_dir": str(output_dir)}

    def _extract_raspa_henry_summaries_any(self, context: Dict[str, Any]) -> Dict[str, Any]:
        upstream = context.get("upstream_plans", {}) or {}
        out: Dict[str, Any] = {}

        for plan_name, plan_blob in upstream.items():
            if not isinstance(plan_blob, dict):
                continue

            for job_id, job in plan_blob.items():
                if not isinstance(job, dict):
                    continue

                if (job.get("agent") != "RASPAAgent") or (job.get("property") != "henry_coefficient"):
                    continue

                mof = job.get("mof") or job.get("MOF")
                guest = job.get("guest") or "guest"
                res = job.get("results", {}) or {}

                henry = res.get("henry_constant")
                if henry is None:
                    continue

                out.setdefault(mof, {})
                out[mof][guest] = {
                    "henry_constant": henry,
                    "henry_error": res.get("henry_error"),
                    "henry_units": res.get("henry_units"),
                    "raspa_summary": res.get("raspa_summary"),
                    "raspa_output_file": res.get("raspa_output_file"),
                    "note": "RASPA parsed henry_constant (Widom insertion).",
                }

        return out

    def _run_bader_summaries_any(self, context: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
        context.setdefault("analysis", {})
        context["analysis"].setdefault("bader_summary", {})

        pairs = self._extract_bader_dirs_any(context)
        for plan_name, (mof_dir, complex_dir, guest_dir) in pairs.items():
            delta = self._build_bader_delta_q_for_mof_complex(
                mof_dir,
                complex_dir,
                guest_dir=guest_dir,
            )
            summary = self._summarize_delta_q(delta, top_k=top_k)
            has_reference_density = all(
                (directory / "CHGCAR_sum").exists()
                for directory in (mof_dir, complex_dir)
            )
            summary["quality_control"] = {
                "reference_density_mode": (
                    "aeccar0_plus_aeccar2"
                    if has_reference_density
                    else "unknown_or_missing"
                ),
                "mof_reference_density": str(mof_dir / "CHGCAR_sum")
                if (mof_dir / "CHGCAR_sum").exists()
                else None,
                "complex_reference_density": str(complex_dir / "CHGCAR_sum")
                if (complex_dir / "CHGCAR_sum").exists()
                else None,
            }
            context["analysis"]["bader_summary"][plan_name] = summary

        return context

    def _extract_raspa_uptake_summaries_any(self, context: Dict[str, Any]) -> Dict[str, Any]:
        upstream = context.get("upstream_plans", {}) or {}
        out: Dict[str, Any] = {}

        for plan_name, plan_blob in upstream.items():
            if not isinstance(plan_blob, dict):
                continue

            for job_id, job in plan_blob.items():
                if not isinstance(job, dict):
                    continue

                if (job.get("agent") != "RASPAAgent") or (job.get("property") != "uptake"):
                    continue

                mof = job.get("mof") or job.get("MOF")
                guest = job.get("guest")
                res = job.get("results", {}) or {}

                if not mof:
                    continue


                uptake = res.get("uptake_excess")
                units = res.get("uptake_units")

                if uptake is None:
                    continue

                out.setdefault(mof, {})
                out[mof][guest or "guest"] = {
                    "uptake_excess": uptake,
                    "uptake_error": res.get("uptake_error"),
                    "uptake_units": units,
                    "raspa_summary": res.get("raspa_summary"),
                    "raspa_output_file": res.get("raspa_output_file"),
                    "note": "RASPA parsed uptake_excess plus conditions, thermodynamics, energies, and Widom summaries when available.",
                }

        return out

    def _extract_raspa_selectivity_summaries_any(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        upstream = context.get("upstream_plans", {}) or {}
        out: Dict[str, Any] = {}

        for plan_blob in upstream.values():
            if not isinstance(plan_blob, dict):
                continue
            for job_id, job in plan_blob.items():
                if not isinstance(job, dict) or job.get("agent") != "RASPAAgent":
                    continue

                prop = (
                    job.get("property")
                    or job.get("property_name")
                    or job.get("simulation_property")
                )
                if prop not in {"selectivity", "binary_selectivity"}:
                    continue

                results = job.get("results", {}) or {}
                selectivity = self._clean_numeric(results.get("selectivity"))
                if selectivity is None:
                    continue

                mof = job.get("mof") or job.get("MOF") or "unknown_mof"
                guests = job.get("guests") or job.get("guest")
                if isinstance(guests, (list, tuple)):
                    pair_label = "/".join(str(guest) for guest in guests)
                elif guests:
                    pair_label = str(guests)
                else:
                    loadings = results.get("component_loadings_excess_molkg") or {}
                    pair_label = "/".join(str(name) for name in loadings) or "binary_mixture"

                out.setdefault(str(mof), {})
                out[str(mof)].setdefault(pair_label, {})
                out[str(mof)][pair_label][str(job_id)] = {
                    "selectivity": selectivity,
                    "definition": results.get("selectivity_definition"),
                    "component_loadings_excess_molkg": results.get(
                        "component_loadings_excess_molkg"
                    ),
                    "gas_fractions": job.get("gas_fractions"),
                    "raspa_summary": results.get("raspa_summary"),
                    "raspa_output_file": results.get("raspa_output_file"),
                    "note": "Equilibrium mixture selectivity, not kinetic selectivity.",
                }

        return out

    def _extract_raspa_isotherm_series_any(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        candidates: List[Tuple[str, Dict[str, Any]]] = []
        seen_candidates = set()

        def add_candidate(label: str, job: Any) -> None:
            if not isinstance(job, dict) or id(job) in seen_candidates:
                return
            results = job.get("results", {}) or {}
            is_raspa = job.get("agent") == "RASPAAgent"
            has_isotherm = bool(
                results.get("raspa_batch_summary")
                or results.get("raspa_batch_summary_path")
                or job.get("batch")
            )
            if not is_raspa and not has_isotherm:
                return
            seen_candidates.add(id(job))
            candidates.append((label, job))

        add_candidate(
            str(context.get("plan_name") or context.get("job_name") or "context"),
            context,
        )
        for plan_name, plan_blob in (context.get("upstream_plans", {}) or {}).items():
            if not isinstance(plan_blob, dict):
                continue
            for job_id, job in plan_blob.items():
                add_candidate(f"{plan_name}:{job_id}", job)

        grouped: Dict[Tuple[str, str, str, Optional[float], str], Dict[str, Any]] = {}
        for fallback_label, job in candidates:
            prop = str(
                job.get("property")
                or job.get("property_name")
                or job.get("simulation_property")
                or ""
            ).strip().lower().replace(" ", "_").replace("-", "_")
            if prop and prop not in {"uptake", "adsorption_isotherm", "isotherm"}:
                continue

            results = job.get("results", {}) or {}
            batch_summary = results.get("raspa_batch_summary")
            if not isinstance(batch_summary, dict):
                summary_path = results.get("raspa_batch_summary_path")
                if summary_path:
                    try:
                        batch_summary = json.loads(Path(str(summary_path)).read_text(encoding="utf-8"))
                    except (OSError, ValueError, TypeError):
                        batch_summary = None

            batch = job.get("batch")
            if isinstance(batch, list) and len(batch) >= 2:
                entries = batch
            elif isinstance(batch_summary, dict) and batch_summary.get("is_isotherm_batch"):
                entries = batch_summary.get("ranked") or []
            else:
                entries = [job]

            series_id = str(
                job.get("plan_name")
                or job.get("job_name")
                or job.get("job_id")
                or fallback_label
            )
            parent_mof = str(job.get("mof") or job.get("MOF") or "unknown_mof")
            parent_guest = job.get("guest")
            if not parent_guest:
                guests = job.get("guests")
                if isinstance(guests, list) and len(guests) == 1:
                    parent_guest = guests[0]
            parent_guest = str(parent_guest or "guest")
            parent_temperature = self._clean_numeric(
                job.get("temperature") or job.get("temperature_K")
            )

            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                entry_results = entry.get("results", {}) or {}
                if not entry_results and (
                    entry.get("uptake_excess") is not None
                    or entry.get("raspa_summary") is not None
                ):
                    entry_results = entry

                uptake = self._clean_numeric(entry_results.get("uptake_excess"))
                pressure = self._clean_numeric(entry.get("pressure_bar"))
                raspa_summary = entry_results.get("raspa_summary") or {}
                if pressure is None:
                    pressure = self._pressure_bar_from_raspa_summary(raspa_summary)
                if pressure is None or uptake is None:
                    continue

                conditions = raspa_summary.get("conditions") or {}
                temperature = self._clean_numeric(
                    entry.get("temperature")
                    or entry.get("temperature_K")
                    or conditions.get("temperature_K")
                )
                if temperature is None:
                    temperature = parent_temperature
                mof = str(entry.get("mof") or entry.get("MOF") or parent_mof)
                guest = entry.get("guest")
                if not guest:
                    guests = entry.get("guests")
                    if isinstance(guests, list) and len(guests) == 1:
                        guest = guests[0]
                guest = str(guest or parent_guest)
                units = str(entry_results.get("uptake_units") or "")

                key = (series_id, mof, guest, temperature, units)
                group = grouped.setdefault(
                    key,
                    {
                        "series_id": series_id,
                        "mof": mof,
                        "guest": guest,
                        "temperature_K": temperature,
                        "uptake_units": units or None,
                        "points": [],
                    },
                )
                group["points"].append(
                    {
                        "pressure_bar": pressure,
                        "uptake": uptake,
                        "uptake_error": self._clean_numeric(entry_results.get("uptake_error")),
                        "source_file": entry_results.get("raspa_output_file"),
                        "work_dir": entry.get("work_dir") or job.get("work_dir"),
                    }
                )

        return [
            series
            for series in grouped.values()
            if len({point["pressure_bar"] for point in series["points"]}) >= 2
        ]

    def _analyze_raspa_isotherms_any(self, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from analysis.raspa_isotherm import analyze_isotherm_collection
        except Exception as exc:
            return {
                "method": "isotherm_shape_analysis",
                "status": "import_failed",
                "error": str(exc),
                "series": [],
            }

        series = self._extract_raspa_isotherm_series_any(context)
        result = analyze_isotherm_collection(series)
        if not series:
            return result

        options = context.get("analysis_options", {}) or {}
        output_dir_value = options.get("isotherm_output_dir")
        if output_dir_value:
            output_dir = Path(str(output_dir_value))
        else:
            root = Path(
                str(
                    context.get("working_dir")
                    or context.get("work_dir")
                    or "working_dir"
                )
            )
            output_dir = root / "analysis_agent_results"

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "raspa_isotherm_shape_analysis.json"
            result["artifact_path"] = str(output_path)
            output_path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            result["artifact_write_error"] = str(exc)
        return result

    def _extract_raspa_thermodynamics_summaries_any(self, context: Dict[str, Any]) -> Dict[str, Any]:
        upstream = context.get("upstream_plans", {}) or {}
        out: Dict[str, Any] = {}

        for plan_name, plan_blob in upstream.items():
            if not isinstance(plan_blob, dict):
                continue

            for job_id, job in plan_blob.items():
                if not isinstance(job, dict):
                    continue
                if job.get("agent") != "RASPAAgent":
                    continue

                mof = job.get("mof") or job.get("MOF")
                guest = job.get("guest") or "guest"
                prop = job.get("property") or job.get("property_name") or job.get("simulation_property")
                res = job.get("results", {}) or {}
                if not mof:
                    continue

                raspa_summary = res.get("raspa_summary") or {}
                thermo = raspa_summary.get("thermodynamics") or {}
                qst_block = thermo.get("qst") or {}
                enthalpy_block = thermo.get("enthalpy_of_adsorption") or {}

                enthalpy = res.get("enthalpy_of_adsorption")
                enthalpy_error = res.get("enthalpy_of_adsorption_error")
                enthalpy_units = res.get("enthalpy_of_adsorption_units")
                if enthalpy is None:
                    enthalpy = enthalpy_block.get("value")
                    enthalpy_error = enthalpy_block.get("error")
                    enthalpy_units = enthalpy_block.get("unit")

                qst = res.get("qst")
                qst_error = res.get("qst_error")
                qst_units = res.get("qst_units")
                if qst is None:
                    qst = qst_block.get("value")
                    qst_error = qst_block.get("error")
                    qst_units = qst_block.get("unit")

                if qst is None and enthalpy is not None:
                    try:
                        qst = -float(enthalpy)
                    except (TypeError, ValueError):
                        qst = None
                    qst_error = enthalpy_error
                    qst_units = enthalpy_units

                if qst is None and enthalpy is None:
                    continue

                out.setdefault(mof, {})
                out[mof].setdefault(guest, {})
                out[mof][guest][str(job_id)] = {
                    "source_property": prop,
                    "qst": qst,
                    "qst_error": qst_error,
                    "qst_units": qst_units or "kJ/mol",
                    "enthalpy_of_adsorption": enthalpy,
                    "enthalpy_of_adsorption_error": enthalpy_error,
                    "enthalpy_of_adsorption_units": enthalpy_units or "kJ/mol",
                    "definition": "Qst = - enthalpy_of_adsorption; larger positive Qst means stronger exothermic adsorption.",
                    "raspa_output_file": res.get("raspa_output_file"),
                    "note": "RASPA thermodynamic adsorption strength descriptor parsed from the Enthalpy of adsorption block.",
                }

        return out

    @staticmethod
    def _clean_numeric(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            if isinstance(value, str) and not value.strip():
                return None
            x = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(x):
            return None
        return x

    @staticmethod
    def _rank_values(values: List[float]) -> List[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        return ranks

    def _correlate(self, xs: List[float], ys: List[float]) -> Dict[str, Any]:
        if len(xs) < 2 or len(ys) < 2:
            return {"n": len(xs), "pearson": None, "spearman": None, "status": "insufficient_data"}
        if len(set(xs)) < 2 or len(set(ys)) < 2:
            return {"n": len(xs), "pearson": None, "spearman": None, "status": "constant_values"}

        pearson = float(np.corrcoef(np.array(xs, dtype=float), np.array(ys, dtype=float))[0, 1])
        rx = self._rank_values(xs)
        ry = self._rank_values(ys)
        spearman = float(np.corrcoef(np.array(rx, dtype=float), np.array(ry, dtype=float))[0, 1])
        return {
            "n": len(xs),
            "pearson": pearson,
            "spearman": spearman,
            "status": "ok" if len(xs) >= 3 else "computed_from_two_points",
        }

    def _pairwise_uncertainty_comparisons(
        self,
        rows: List[Dict[str, Any]],
        value_key: str,
        error_key: str,
        confidence_z: float = 1.96,
    ) -> List[Dict[str, Any]]:
        comparisons: List[Dict[str, Any]] = []
        usable = [
            row
            for row in rows
            if self._clean_numeric(row.get(value_key)) is not None
            and self._clean_numeric(row.get(error_key)) is not None
        ]
        for left, right in itertools.combinations(usable, 2):
            left_value = float(left[value_key])
            right_value = float(right[value_key])
            left_error = abs(float(left[error_key]))
            right_error = abs(float(right[error_key]))
            combined_error = math.sqrt(left_error ** 2 + right_error ** 2)
            difference = left_value - right_value
            z_score = (
                abs(difference) / combined_error
                if combined_error > 0.0
                else None
            )
            resolved = z_score is not None and z_score >= confidence_z
            comparisons.append(
                {
                    "mof_a": left.get("mof"),
                    "mof_b": right.get("mof"),
                    "difference_a_minus_b": difference,
                    "combined_standard_uncertainty": combined_error,
                    "z_score": z_score,
                    "resolved_at_95_percent": resolved,
                    "higher_if_resolved": (
                        left.get("mof")
                        if resolved and difference > 0.0
                        else right.get("mof")
                        if resolved and difference < 0.0
                        else None
                    ),
                }
            )
        return comparisons

    def _iter_uptake_rows(self, uptake_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for mof, guest_map in sorted((uptake_summary or {}).items()):
            if not isinstance(guest_map, dict):
                continue
            for guest, rec in sorted(guest_map.items()):
                if not isinstance(rec, dict):
                    continue
                uptake = self._clean_numeric(rec.get("uptake_excess"))
                if uptake is None:
                    continue
                rows.append(
                    {
                        "mof": mof,
                        "guest": guest,
                        "uptake_excess": uptake,
                        "uptake_error": self._clean_numeric(rec.get("uptake_error")),
                        "uptake_units": rec.get("uptake_units"),
                        "uptake_source_file": rec.get("raspa_output_file"),
                    }
                )
        return rows

    def _raspa_guest_loadings(
        self,
        raspa_summary: Dict[str, Any],
        guest: str,
    ) -> Dict[str, Any]:
        all_loadings = (raspa_summary or {}).get("loadings") or {}
        if not isinstance(all_loadings, dict):
            return {}

        guest_loadings = all_loadings.get(guest) or all_loadings.get(str(guest).lower())
        if not guest_loadings:
            guest_key = str(guest).strip().lower()
            for label, loadings in all_loadings.items():
                if str(label).strip().lower() == guest_key:
                    guest_loadings = loadings
                    break
        if not guest_loadings and len(all_loadings) == 1:
            guest_loadings = next(iter(all_loadings.values()))
        return guest_loadings if isinstance(guest_loadings, dict) else {}

    def _raspa_loading_value(
        self,
        raspa_summary: Dict[str, Any],
        guest: str,
        unit: str,
    ) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        guest_loadings = self._raspa_guest_loadings(raspa_summary, guest)
        for mode in ("excess", "absolute"):
            block = ((guest_loadings.get(mode) or {}).get(unit) or {})
            value = self._clean_numeric(block.get("value"))
            error = self._clean_numeric(block.get("error"))
            if value is not None:
                return value, error, mode
        return None, None, None

    @staticmethod
    def _is_mass_based_henry_unit(unit: Any) -> bool:
        normalized = str(unit or "").lower().replace(" ", "")
        return normalized in {
            "mol/kg/pa",
            "mol/(kg*pa)",
            "molkg^-1pa^-1",
            "molkg-1pa-1",
        }

    def _uptake_normalization_fields(
        self,
        raspa_summary: Dict[str, Any],
        guest: str,
        henry_constant: Optional[float] = None,
        henry_error: Optional[float] = None,
        henry_units: Optional[str] = None,
    ) -> Dict[str, Any]:
        grav_mol_kg, grav_mol_kg_error, grav_mode = self._raspa_loading_value(
            raspa_summary,
            guest,
            "mol/kg framework",
        )
        grav_cm3_g, grav_cm3_g_error, grav_volume_mode = self._raspa_loading_value(
            raspa_summary,
            guest,
            "cm^3 (STP)/gr framework",
        )
        vol_cm3_cm3, vol_cm3_cm3_error, vol_mode = self._raspa_loading_value(
            raspa_summary,
            guest,
            "cm^3 (STP)/cm^3 framework",
        )

        density_g_cm3 = None
        if (
            grav_cm3_g is not None
            and grav_cm3_g > 0.0
            and vol_cm3_cm3 is not None
            and grav_volume_mode == vol_mode
        ):
            density_g_cm3 = vol_cm3_cm3 / grav_cm3_g
        density_kg_m3 = (
            density_g_cm3 * 1000.0
            if density_g_cm3 is not None
            else None
        )

        mass_henry = (
            henry_constant
            if henry_constant is not None
            and self._is_mass_based_henry_unit(henry_units)
            else None
        )
        volumetric_henry = (
            mass_henry * density_kg_m3
            if mass_henry is not None and density_kg_m3 is not None
            else None
        )
        volumetric_henry_error = (
            henry_error * density_kg_m3
            if henry_error is not None
            and mass_henry is not None
            and density_kg_m3 is not None
            else None
        )

        conditions = (raspa_summary or {}).get("conditions") or {}
        pressure_pa = self._clean_numeric(conditions.get("external_pressure_Pa"))
        if pressure_pa is None:
            pressure_bar = self._clean_numeric(conditions.get("external_pressure_bar"))
            pressure_pa = pressure_bar * 100000.0 if pressure_bar is not None else None
        predicted_gravimetric = (
            mass_henry * pressure_pa
            if mass_henry is not None and pressure_pa is not None
            else None
        )
        observed_to_prediction = (
            grav_mol_kg / predicted_gravimetric
            if grav_mol_kg is not None
            and predicted_gravimetric is not None
            and predicted_gravimetric > 0.0
            else None
        )

        return {
            "gravimetric_uptake_mol_kg": grav_mol_kg,
            "gravimetric_uptake_mol_kg_error": grav_mol_kg_error,
            "gravimetric_uptake_mol_kg_mode": grav_mode,
            "gravimetric_uptake_cm3STP_g": grav_cm3_g,
            "gravimetric_uptake_cm3STP_g_error": grav_cm3_g_error,
            "volumetric_uptake_cm3STP_cm3": vol_cm3_cm3,
            "volumetric_uptake_cm3STP_cm3_error": vol_cm3_cm3_error,
            "framework_density_g_cm3": density_g_cm3,
            "framework_density_kg_m3": density_kg_m3,
            "framework_density_source": (
                "Matched volumetric/gravimetric RASPA loading conversion ratio"
                if density_g_cm3 is not None
                else None
            ),
            "henry_mass_mol_kg_Pa": mass_henry,
            "henry_mass_error_mol_kg_Pa": (
                henry_error if mass_henry is not None else None
            ),
            "henry_volumetric_mol_m3_Pa": volumetric_henry,
            "henry_volumetric_error_mol_m3_Pa": volumetric_henry_error,
            "pressure_Pa": pressure_pa,
            "henry_law_predicted_gravimetric_mol_kg": predicted_gravimetric,
            "observed_to_henry_law_prediction_ratio": observed_to_prediction,
        }

    def _pressure_bar_from_raspa_summary(self, raspa_summary: Dict[str, Any]) -> Optional[float]:
        conditions = (raspa_summary or {}).get("conditions") or {}
        pressure_bar = self._clean_numeric(conditions.get("external_pressure_bar"))
        if pressure_bar is not None:
            return pressure_bar
        pressure_pa = self._clean_numeric(conditions.get("external_pressure_Pa"))
        if pressure_pa is not None:
            return pressure_pa / 100000.0
        return None

    def _classify_adsorption_pressure_regime(
        self,
        pressure_bar: Optional[float],
        guest: Optional[str] = None,
    ) -> Dict[str, Any]:
        base = {
            "pressure_bar": pressure_bar,
            "guest": guest,
            "threshold_basis": "heuristic",
            "requires_isotherm_for_confirmation": True,
        }
        if pressure_bar is None:
            return {
                **base,
                "regime": "unknown",
                "confidence": "low",
                "primary_interpretation_axis": "unknown",
                "secondary_axes": [],
                "zeopp_descriptor_role": "unknown",
                "reason": "Pressure was not available in the parsed RASPA summary.",
            }
        if pressure_bar <= 0.1:
            return {
                **base,
                "regime": "low_pressure_henry",
                "confidence": "moderate",
                "primary_interpretation_axis": "basis_matched_henry_coefficient",
                "secondary_axes": ["qst", "zeopp_accessibility_descriptors"],
                "zeopp_descriptor_role": "supporting_accessibility_descriptor_not_capacity_limit",
                "reason": (
                    "At low pressure, loading is expected to be close to the Henry limit, "
                    "so uptake should be interpreted through a Henry coefficient expressed on "
                    "the same mass or volume basis as the compared uptake."
                ),
            }
        if pressure_bar < 5.0:
            return {
                **base,
                "regime": "intermediate_pressure",
                "confidence": "moderate",
                "primary_interpretation_axis": "mixed_affinity_and_accessible_volume",
                "secondary_axes": ["henry_coefficient", "qst", "pore_volume", "surface_area"],
                "zeopp_descriptor_role": "capacity_and_accessibility_descriptor",
                "reason": "At intermediate pressure, both affinity and accessible pore capacity can affect uptake.",
            }
        return {
            **base,
            "regime": "high_pressure_capacity",
            "confidence": "moderate",
            "primary_interpretation_axis": "pore_volume_and_volumetric_capacity",
            "secondary_axes": ["surface_area", "packing", "qst", "henry_coefficient"],
            "zeopp_descriptor_role": "capacity_descriptor",
            "reason": "At high pressure, pore volume and packing capacity are expected to become more important than Henry-limit affinity.",
        }

    def _analyze_adsorption_regime(self, uptake_summary: Dict[str, Any]) -> Dict[str, Any]:
        rows: List[Dict[str, Any]] = []
        for base_row in self._iter_uptake_rows(uptake_summary):
            mof = base_row["mof"]
            guest = base_row["guest"]
            rec = ((uptake_summary or {}).get(mof) or {}).get(guest) or {}
            raspa_summary = rec.get("raspa_summary") or {}
            conditions = raspa_summary.get("conditions") or {}
            pressure_bar = self._pressure_bar_from_raspa_summary(raspa_summary)
            regime = self._classify_adsorption_pressure_regime(pressure_bar, guest=guest)
            rows.append(
                {
                    **base_row,
                    "temperature_K": self._clean_numeric(conditions.get("temperature_K")),
                    "pressure_bar": pressure_bar,
                    "pressure_regime": regime,
                }
            )

        regimes = sorted({(row.get("pressure_regime") or {}).get("regime") for row in rows if row.get("pressure_regime")})
        primary_axes = sorted(
            {
                (row.get("pressure_regime") or {}).get("primary_interpretation_axis")
                for row in rows
                if row.get("pressure_regime")
            }
        )
        if not rows:
            status = "insufficient_data"
            note = "No RASPA uptake rows were available for pressure-regime classification."
        elif len(regimes) == 1:
            status = "ok"
            note = f"All uptake rows are classified as {regimes[0]}."
        else:
            status = "mixed_regimes"
            note = "Compared uptake rows span multiple pressure regimes; do not interpret all points with a single evidence axis."

        return {
            "method": "adsorption_regime_analysis",
            "definition": {
                "goal": "Classify adsorption pressure regime so uptake interpretation uses the appropriate evidence axis.",
                "low_pressure_henry": (
                    "A basis-matched Henry coefficient is the primary uptake descriptor: "
                    "mass-based Henry for gravimetric uptake, or density-converted volumetric "
                    "Henry for volumetric uptake. Zeo++ descriptors support accessibility/entropy "
                    "discussion, not direct capacity limitation."
                ),
                "intermediate_pressure": "Affinity and accessible volume can both matter.",
                "high_pressure_capacity": "Pore volume, volumetric capacity, and packing become primary descriptors.",
            },
            "rows": rows,
            "regimes": regimes,
            "primary_interpretation_axes": primary_axes,
            "trend_status": status,
            "trend_note": note,
            "limitations": [
                "Regime thresholds are heuristics; pressure-dependent isotherms are needed for strict confirmation.",
                "Henry-regime classification assumes the parsed pressure corresponds to the compared uptake point.",
                "Capacity descriptors can correlate with low-pressure uptake through accessibility/entropy, but should not be called saturation-capacity evidence at low pressure.",
            ],
        }

    def _find_henry_for_uptake_row(
        self,
        henry_summary: Dict[str, Any],
        mof: str,
        guest: str,
    ) -> Dict[str, Any]:
        mof_block = (henry_summary or {}).get(mof, {}) or {}
        rec = mof_block.get(guest)
        if rec is None and len(mof_block) == 1:
            rec = next(iter(mof_block.values()))
        if not isinstance(rec, dict):
            return {}
        return rec

    def _find_qst_for_uptake_row(
        self,
        thermo_summary: Dict[str, Any],
        mof: str,
        guest: str,
    ) -> Dict[str, Any]:
        mof_block = (thermo_summary or {}).get(mof, {}) or {}
        guest_block = mof_block.get(guest)
        if guest_block is None and len(mof_block) == 1:
            guest_block = next(iter(mof_block.values()))
        if not isinstance(guest_block, dict):
            return {}

        for rec in guest_block.values():
            if isinstance(rec, dict) and rec.get("qst") is not None:
                return rec
        return {}

    @staticmethod
    def _guest_kinetic_diameter_A(guest: Optional[str]) -> Optional[float]:
        if not guest:
            return None
        key = str(guest).strip().upper().replace("-", "")
        diameters = {
            "H2": 2.89,
            "HE": 2.60,
            "CO2": 3.30,
            "N2": 3.64,
            "O2": 3.46,
            "CH4": 3.80,
            "AR": 3.40,
            "KR": 3.60,
            "XE": 4.10,
            "CO": 3.76,
            "H2O": 2.65,
            "NH3": 2.60,
        }
        return diameters.get(key)

    @staticmethod
    def _steric_fit_label(pld_ratio: Optional[float], lcd_ratio: Optional[float]) -> str:
        if pld_ratio is None:
            return "unknown"
        if pld_ratio < 1.0:
            return "kinetically_restricted"
        if pld_ratio < 1.25:
            return "tight_aperture"
        if lcd_ratio is not None and lcd_ratio > 4.0:
            return "very_open_or_diluted"
        if pld_ratio <= 3.0:
            return "well_matched"
        return "open_pore"

    def _find_energy_histogram_for_uptake_row(
        self,
        energy_histogram_summary: Dict[str, Any],
        mof: str,
        guest: str,
    ) -> Dict[str, Any]:
        mof_block = (energy_histogram_summary or {}).get(mof, {}) or {}
        guest_block = mof_block.get(guest)
        if guest_block is None and len(mof_block) == 1:
            guest_block = next(iter(mof_block.values()))
        if not isinstance(guest_block, dict):
            return {}

        for rec in guest_block.values():
            if isinstance(rec, dict) and rec.get("method") == "energy_histogram_analysis":
                return rec
        return {}

    def _trend_status_for_rows(self, rows: List[Dict[str, Any]], correlations: Dict[str, Any], target: str) -> Tuple[str, str]:
        interpretable = [
            key for key, val in correlations.items()
            if val.get("status") in {"ok", "computed_from_two_points"}
        ]
        if len(rows) < 2:
            return "insufficient_data", f"Need at least two matched MOF rows to compare {target} with uptake."
        if not interpretable:
            return "insufficient_descriptor_overlap", f"Matched uptake rows exist, but {target} values are missing or constant."
        if len(rows) < 3:
            return "qualitative_only", "Only two MOFs are available; correlation signs are direction checks, not robust trends."
        return "ok", "Correlations are exploratory and should be checked with more MOFs before treating them as causal."

    def _build_uptake_henry_evidence(
        self,
        uptake_summary: Dict[str, Any],
        henry_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        rows: List[Dict[str, Any]] = []
        for row in self._iter_uptake_rows(uptake_summary):
            henry = self._find_henry_for_uptake_row(henry_summary, row["mof"], row["guest"])
            rec = ((uptake_summary or {}).get(row["mof"]) or {}).get(row["guest"]) or {}
            raspa_summary = rec.get("raspa_summary") or {}
            henry_constant = self._clean_numeric(henry.get("henry_constant"))
            henry_error = self._clean_numeric(henry.get("henry_error"))
            henry_units = henry.get("henry_units")
            row = dict(row)
            row["henry_constant"] = henry_constant
            row["henry_error"] = henry_error
            row["henry_units"] = henry_units
            row["henry_source_file"] = henry.get("raspa_output_file")
            row.update(
                self._uptake_normalization_fields(
                    raspa_summary,
                    row["guest"],
                    henry_constant=henry_constant,
                    henry_error=henry_error,
                    henry_units=henry_units,
                )
            )
            rows.append(row)

        mass_pairs = [
            (
                float(row["henry_mass_mol_kg_Pa"]),
                float(row["gravimetric_uptake_mol_kg"]),
            )
            for row in rows
            if row.get("henry_mass_mol_kg_Pa") is not None
            and row.get("gravimetric_uptake_mol_kg") is not None
        ]
        volumetric_pairs = [
            (
                float(row["henry_volumetric_mol_m3_Pa"]),
                float(row["volumetric_uptake_cm3STP_cm3"]),
            )
            for row in rows
            if row.get("henry_volumetric_mol_m3_Pa") is not None
            and row.get("volumetric_uptake_cm3STP_cm3") is not None
        ]
        correlations = {
            "mass_henry_vs_gravimetric_uptake_mol_kg": self._correlate(
                [pair[0] for pair in mass_pairs],
                [pair[1] for pair in mass_pairs],
            ),
            "volumetric_henry_vs_volumetric_uptake_cm3STP_cm3": self._correlate(
                [pair[0] for pair in volumetric_pairs],
                [pair[1] for pair in volumetric_pairs],
            ),
        }
        trend_status, _ = self._trend_status_for_rows(
            rows,
            correlations,
            "basis-matched Henry coefficients",
        )

        mass_henry_values = [
            float(row["henry_mass_mol_kg_Pa"])
            for row in rows
            if row.get("henry_mass_mol_kg_Pa") is not None
            and float(row["henry_mass_mol_kg_Pa"]) > 0.0
        ]
        volumetric_henry_values = [
            float(row["henry_volumetric_mol_m3_Pa"])
            for row in rows
            if row.get("henry_volumetric_mol_m3_Pa") is not None
            and float(row["henry_volumetric_mol_m3_Pa"]) > 0.0
        ]
        mass_spread = (
            max(mass_henry_values) / min(mass_henry_values)
            if len(mass_henry_values) >= 2
            else None
        )
        volumetric_spread = (
            max(volumetric_henry_values) / min(volumetric_henry_values)
            if len(volumetric_henry_values) >= 2
            else None
        )
        density_compensation = (
            mass_spread is not None
            and volumetric_spread is not None
            and volumetric_spread < mass_spread
        )
        trend_note = (
            "Mass-based Henry coefficients are compared only with gravimetric uptake; "
            "density-converted volumetric Henry coefficients are compared only with volumetric uptake."
        )

        return {
            "method": "uptake_henry_relationship",
            "definition": {
                "mass_basis_relation": "q_mass [mol/kg] is approximately K_H,mass [mol/kg/Pa] times pressure [Pa].",
                "volume_basis_conversion": "K_H,vol [mol/m^3/Pa] = K_H,mass [mol/kg/Pa] times framework density [kg/m^3].",
                "basis_matching_rule": (
                    "Compare mass-based Henry only with gravimetric uptake and "
                    "density-converted volumetric Henry only with volumetric uptake."
                ),
            },
            "rows": rows,
            "uptake_ranking": sorted(rows, key=lambda r: r["uptake_excess"], reverse=True),
            "gravimetric_uptake_ranking": sorted(
                [row for row in rows if row.get("gravimetric_uptake_mol_kg") is not None],
                key=lambda row: row["gravimetric_uptake_mol_kg"],
                reverse=True,
            ),
            "volumetric_uptake_ranking": sorted(
                [row for row in rows if row.get("volumetric_uptake_cm3STP_cm3") is not None],
                key=lambda row: row["volumetric_uptake_cm3STP_cm3"],
                reverse=True,
            ),
            "correlations_with_uptake": correlations,
            "density_compensation": {
                "mass_henry_max_to_min_ratio": mass_spread,
                "volumetric_henry_max_to_min_ratio": volumetric_spread,
                "detected": density_compensation,
                "interpretation": (
                    "Framework density narrows the spread in Henry response on a volume basis."
                    if density_compensation
                    else "No clear density compensation was detected from the available rows."
                ),
            },
            "basis_matched_interpretation": (
                "Gravimetric uptake should be interpreted with K_H,mass. "
                "Volumetric uptake should be interpreted with K_H,vol = "
                "K_H,mass times framework density. Do not use K_H,mass alone "
                "to explain a volumetric loading trend."
            ),
            "reporting_summary": {
                "rows": [
                    {
                        "mof": row["mof"],
                        "framework_density_g_cm3": row.get("framework_density_g_cm3"),
                        "henry_mass_mol_kg_Pa": row.get("henry_mass_mol_kg_Pa"),
                        "henry_volumetric_mol_m3_Pa": row.get(
                            "henry_volumetric_mol_m3_Pa"
                        ),
                        "gravimetric_uptake_mol_kg": row.get(
                            "gravimetric_uptake_mol_kg"
                        ),
                        "volumetric_uptake_cm3STP_cm3": row.get(
                            "volumetric_uptake_cm3STP_cm3"
                        ),
                    }
                    for row in rows
                ],
                "mass_henry_max_to_min_ratio": mass_spread,
                "volumetric_henry_max_to_min_ratio": volumetric_spread,
                "density_compensation_detected": density_compensation,
                "required_conclusion": (
                    "Gravimetric uptake follows the mass-based Henry coefficient, "
                    "whereas framework-density differences compensate for much of "
                    "this trend on a volumetric basis."
                ),
            },
            "trend_status": trend_status,
            "trend_note": trend_note,
            "limitations": [
                "Henry coefficient explains low-pressure affinity, not saturation capacity.",
                "Framework density is recovered from the ratio of matched volumetric and gravimetric RASPA loading conversions.",
                "The Henry-law loading estimate uses pressure; fugacity should be used when non-ideality is appreciable.",
                "At higher pressure, pore volume and accessible surface can dominate uptake.",
            ],
        }

    def _build_uptake_qst_evidence(
        self,
        uptake_summary: Dict[str, Any],
        thermo_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        rows: List[Dict[str, Any]] = []
        for row in self._iter_uptake_rows(uptake_summary):
            qst = self._find_qst_for_uptake_row(thermo_summary, row["mof"], row["guest"])
            row = dict(row)
            row["qst"] = self._clean_numeric(qst.get("qst"))
            row["qst_error"] = self._clean_numeric(qst.get("qst_error"))
            row["qst_units"] = qst.get("qst_units")
            row["enthalpy_of_adsorption"] = self._clean_numeric(qst.get("enthalpy_of_adsorption"))
            row["enthalpy_of_adsorption_error"] = self._clean_numeric(qst.get("enthalpy_of_adsorption_error"))
            row["enthalpy_of_adsorption_units"] = qst.get("enthalpy_of_adsorption_units")
            row["qst_source_file"] = qst.get("raspa_output_file")
            rows.append(row)

        xs = [float(r["qst"]) for r in rows if r.get("qst") is not None]
        ys = [float(r["uptake_excess"]) for r in rows if r.get("qst") is not None]
        correlations = {"qst": self._correlate(xs, ys)}
        trend_status, trend_note = self._trend_status_for_rows(rows, correlations, "Qst")

        return {
            "method": "uptake_qst_relationship",
            "definition": {
                "target": "uptake_excess",
                "descriptor": "Qst = -enthalpy_of_adsorption",
                "interpretation": "positive correlation means uptake increases with stronger exothermic adsorption.",
            },
            "rows": rows,
            "uptake_ranking": sorted(rows, key=lambda r: r["uptake_excess"], reverse=True),
            "correlations_with_uptake": correlations,
            "trend_status": trend_status,
            "trend_note": trend_note,
            "limitations": [
                "Qst measures adsorption strength, not available pore volume.",
                "Very high Qst can improve low-pressure uptake but may be undesirable for regeneration.",
            ],
        }

    def _analyze_uptake_basis_comparison(self, uptake_summary: Dict[str, Any]) -> Dict[str, Any]:
        unit_specs = {
            "volumetric_cm3STP_cm3": "cm^3 (STP)/cm^3 framework",
            "gravimetric_cm3STP_g": "cm^3 (STP)/gr framework",
            "gravimetric_mol_kg": "mol/kg framework",
            "gravimetric_mg_g": "milligram/gram framework",
        }
        rows: List[Dict[str, Any]] = []

        for base_row in self._iter_uptake_rows(uptake_summary):
            mof = base_row["mof"]
            guest = base_row["guest"]
            rec = ((uptake_summary or {}).get(mof) or {}).get(guest) or {}
            raspa_summary = rec.get("raspa_summary") or {}

            row = dict(base_row)
            for key, unit in unit_specs.items():
                value, error, _ = self._raspa_loading_value(
                    raspa_summary,
                    guest,
                    unit,
                )
                row[key] = value
                row[f"{key}_error"] = error
                row[f"{key}_unit"] = unit
            row.update(self._uptake_normalization_fields(raspa_summary, guest))
            rows.append(row)

        ranking_by_basis: Dict[str, List[Dict[str, Any]]] = {}
        for key in unit_specs:
            ranked = [
                {
                    "mof": row["mof"],
                    "guest": row["guest"],
                    "value": row.get(key),
                    "error": row.get(f"{key}_error"),
                    "unit": row.get(f"{key}_unit"),
                }
                for row in rows
                if row.get(key) is not None
            ]
            ranking_by_basis[key] = sorted(ranked, key=lambda r: r["value"], reverse=True)

        volumetric_order = [r["mof"] for r in ranking_by_basis.get("volumetric_cm3STP_cm3", [])]
        gravimetric_order = [r["mof"] for r in ranking_by_basis.get("gravimetric_cm3STP_g", [])]
        order_match = bool(volumetric_order and gravimetric_order and volumetric_order == gravimetric_order)

        if len(rows) < 2:
            status = "insufficient_data"
            note = "Need at least two MOFs with RASPA loading tables to compare uptake bases."
        elif not volumetric_order or not gravimetric_order:
            status = "missing_basis"
            note = "RASPA uptake exists, but volumetric or gravimetric loading units are missing from the parsed loading table."
        elif order_match:
            status = "consistent_rankings"
            note = "Volumetric and gravimetric uptake rank the MOFs in the same order."
        else:
            status = "basis_dependent_rankings"
            note = "Volumetric and gravimetric uptake rank the MOFs differently; density or framework mass affects the apparent performance trend."

        return {
            "method": "uptake_basis_comparison",
            "definition": {
                "goal": "Compare adsorption capacity on the two standard bases: volumetric and gravimetric uptake.",
                "volumetric_basis": "cm^3 (STP) gas per cm^3 framework",
                "gravimetric_basis": [
                    "cm^3 (STP) gas per gram framework",
                    "mol gas per kg framework",
                    "mg gas per gram framework",
                ],
                "interpretation": (
                    "Volumetric uptake is relevant to fixed-bed/tank volume efficiency; "
                    "gravimetric uptake is relevant to adsorbent mass efficiency."
                ),
                "density_relation": (
                    "framework density [g/cm^3] = volumetric uptake [cm^3(STP)/cm^3] "
                    "/ gravimetric uptake [cm^3(STP)/g] for matched loading conversions."
                ),
            },
            "rows": rows,
            "ranking_by_basis": ranking_by_basis,
            "pairwise_resolution_95_percent": {
                "volumetric_cm3STP_cm3": self._pairwise_uncertainty_comparisons(
                    rows,
                    "volumetric_cm3STP_cm3",
                    "volumetric_cm3STP_cm3_error",
                ),
                "gravimetric_cm3STP_g": self._pairwise_uncertainty_comparisons(
                    rows,
                    "gravimetric_cm3STP_g",
                    "gravimetric_cm3STP_g_error",
                ),
            },
            "volumetric_vs_gravimetric_order_match": order_match,
            "trend_status": status,
            "trend_note": note,
            "limitations": [
                "Use the same temperature, pressure, force field, and framework convention before comparing bases.",
                "cm^3(STP)/g, mol/kg, and mg/g are all gravimetric variants and should not be treated as independent physical mechanisms.",
                "This analysis does not include accessible pore volume unless Zeo++ descriptors are separately available.",
            ],
        }

    def _build_uptake_evidence_synthesis(
        self,
        uptake_summary: Dict[str, Any],
        henry_summary: Dict[str, Any],
        thermo_summary: Dict[str, Any],
        zeopp_summary: Dict[str, Any],
        energy_histogram_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        rows: List[Dict[str, Any]] = []

        for base_row in self._iter_uptake_rows(uptake_summary):
            mof = base_row["mof"]
            guest = base_row["guest"]
            uptake_rec = ((uptake_summary or {}).get(mof) or {}).get(guest) or {}
            raspa_summary = uptake_rec.get("raspa_summary") or {}
            z = zeopp_summary.get(mof, {}) if isinstance(zeopp_summary, dict) else {}
            diam = z.get("pore_diameter", {}) or {}
            lcd = z.get("largest_cavity_diameter", {}) or {}
            pv = z.get("pore_volume", {}) or {}
            sa = z.get("surface_area", {}) or {}
            henry = self._find_henry_for_uptake_row(henry_summary, mof, guest)
            qst = self._find_qst_for_uptake_row(thermo_summary, mof, guest)
            hist = self._find_energy_histogram_for_uptake_row(energy_histogram_summary, mof, guest)
            hostguest = hist.get("hostguest_summary", {}) or {}
            per_mol = hist.get("hostguest_per_molecule_estimate", {}) or {}

            pld = self._clean_numeric(diam.get("PLD_free_sphere_A"))
            lcd_value = self._clean_numeric(diam.get("LCD_included_sphere_A") or lcd.get("LCD_included_sphere_A"))
            pore_volume = self._clean_numeric(pv.get("AV_cm3_g"))
            uptake = self._clean_numeric(base_row.get("uptake_excess"))
            guest_diameter = self._guest_kinetic_diameter_A(guest)
            pld_ratio = (pld / guest_diameter) if pld is not None and guest_diameter else None
            lcd_ratio = (lcd_value / guest_diameter) if lcd_value is not None and guest_diameter else None

            row = dict(base_row)
            row.update(
                {
                    "henry_constant": self._clean_numeric(henry.get("henry_constant")),
                    "henry_error": self._clean_numeric(henry.get("henry_error")),
                    "henry_units": henry.get("henry_units"),
                    "qst": self._clean_numeric(qst.get("qst")),
                    "qst_units": qst.get("qst_units") or "kJ/mol",
                    "hostguest_system_mean_kJ_mol": self._clean_numeric(hostguest.get("mean_kJ_mol")),
                    "hostguest_system_p50_kJ_mol": self._clean_numeric((hostguest.get("quantiles_kJ_mol") or {}).get("p50")),
                    "hostguest_system_fraction_le_minus20_kJ_mol": self._clean_numeric(
                        (hostguest.get("strong_binding_fractions") or {}).get("fraction_energy_le_20_kJ_mol")
                    ),
                    "hostguest_per_molecule_mean_estimate_kJ_mol": self._clean_numeric(
                        per_mol.get("mean_kJ_mol_per_molecule_estimate")
                    ),
                    "hostguest_per_molecule_p50_estimate_kJ_mol": self._clean_numeric(
                        (per_mol.get("quantiles_kJ_mol_per_molecule_estimate") or {}).get("p50")
                    ),
                    "avg_adsorbed_molecules_in_simulation_cell": self._clean_numeric(
                        (hist.get("loading_context") or {}).get("estimated_average_molecules_in_simulation_cell")
                    ),
                    "PLD_A": pld,
                    "LCD_A": lcd_value,
                    "pore_volume_cm3_g": pore_volume,
                    "accessible_volume_fraction": self._clean_numeric(pv.get("AV_volume_fraction")),
                    "surface_area_m2_g": self._clean_numeric(sa.get("ASA_m2_g")),
                    "uptake_per_pore_volume_index": (
                        uptake / pore_volume if uptake is not None and pore_volume and pore_volume > 0.0 else None
                    ),
                    "guest_kinetic_diameter_A": guest_diameter,
                    "PLD_to_guest_diameter_ratio": pld_ratio,
                    "LCD_to_guest_diameter_ratio": lcd_ratio,
                    "steric_fit_label": self._steric_fit_label(pld_ratio, lcd_ratio),
                }
            )
            row.update(
                self._uptake_normalization_fields(
                    raspa_summary,
                    guest,
                    henry_constant=self._clean_numeric(henry.get("henry_constant")),
                    henry_error=self._clean_numeric(henry.get("henry_error")),
                    henry_units=henry.get("henry_units"),
                )
            )
            rows.append(row)

        descriptor_keys = [
            "qst",
            "hostguest_per_molecule_mean_estimate_kJ_mol",
            "hostguest_system_fraction_le_minus20_kJ_mol",
            "pore_volume_cm3_g",
            "uptake_per_pore_volume_index",
            "PLD_to_guest_diameter_ratio",
            "LCD_to_guest_diameter_ratio",
            "surface_area_m2_g",
        ]
        correlations: Dict[str, Any] = {}
        for key in descriptor_keys:
            xs = []
            ys = []
            for row in rows:
                x = row.get(key)
                y = row.get("uptake_excess")
                if x is None or y is None:
                    continue
                xs.append(float(x))
                ys.append(float(y))
            correlations[key] = self._correlate(xs, ys)

        mass_henry_pairs = [
            (
                float(row["henry_mass_mol_kg_Pa"]),
                float(row["gravimetric_uptake_mol_kg"]),
            )
            for row in rows
            if row.get("henry_mass_mol_kg_Pa") is not None
            and row.get("gravimetric_uptake_mol_kg") is not None
        ]
        volumetric_henry_pairs = [
            (
                float(row["henry_volumetric_mol_m3_Pa"]),
                float(row["volumetric_uptake_cm3STP_cm3"]),
            )
            for row in rows
            if row.get("henry_volumetric_mol_m3_Pa") is not None
            and row.get("volumetric_uptake_cm3STP_cm3") is not None
        ]
        correlations["mass_henry_vs_gravimetric_uptake_mol_kg"] = self._correlate(
            [pair[0] for pair in mass_henry_pairs],
            [pair[1] for pair in mass_henry_pairs],
        )
        correlations[
            "volumetric_henry_vs_volumetric_uptake_cm3STP_cm3"
        ] = self._correlate(
            [pair[0] for pair in volumetric_henry_pairs],
            [pair[1] for pair in volumetric_henry_pairs],
        )

        ranked = sorted(rows, key=lambda r: r["uptake_excess"], reverse=True)
        if len(rows) < 2:
            status = "insufficient_data"
            note = "Need at least two MOFs with matched uptake data for a multifactor comparison."
        elif len(rows) < 3:
            status = "qualitative_only"
            note = "Only two MOFs are available; use this as structured evidence, not a robust trend."
        else:
            status = "ok"
            note = "This table is designed for LLM interpretation across affinity, energy, capacity, and steric-fit evidence."

        return {
            "method": "uptake_evidence_synthesis",
            "definition": {
                "goal": "Collect single-pressure uptake evidence into one LLM-readable table.",
                "affinity_axis": [
                    "henry_mass_mol_kg_Pa matched to gravimetric_uptake_mol_kg",
                    "henry_volumetric_mol_m3_Pa matched to volumetric_uptake_cm3STP_cm3",
                    "qst",
                ],
                "energy_axis": [
                    "hostguest_per_molecule_mean_estimate_kJ_mol",
                    "hostguest_system_fraction_le_minus20_kJ_mol",
                ],
                "capacity_axis": ["pore_volume_cm3_g", "surface_area_m2_g", "uptake_per_pore_volume_index"],
                "geometry_fit_axis": [
                    "guest_kinetic_diameter_A",
                    "PLD_to_guest_diameter_ratio",
                    "LCD_to_guest_diameter_ratio",
                    "steric_fit_label",
                ],
            },
            "rows": rows,
            "uptake_ranking": ranked,
            "correlations_with_uptake": correlations,
            "trend_status": status,
            "trend_note": note,
            "limitations": [
                "Single-pressure uptake mixes affinity, loading state, and capacity.",
                "Mass-based Henry coefficients must not be used directly to explain volumetric uptake; framework-density conversion is required.",
                "uptake_per_pore_volume_index is a comparison index unless uptake units are explicitly normalized.",
                "hostguest_per_molecule_mean_estimate is approximate because the RASPA histogram is system-level.",
                "Steric-fit labels use simple kinetic-diameter heuristics and should not replace diffusion calculations.",
            ],
        }

    def _build_uptake_structure_evidence(
        self,
        uptake_summary: Dict[str, Any],
        zeopp_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        rows: List[Dict[str, Any]] = []
        for row in self._iter_uptake_rows(uptake_summary):
            z = zeopp_summary.get(row["mof"], {}) if isinstance(zeopp_summary, dict) else {}
            diam = z.get("pore_diameter", {}) or {}
            lcd = z.get("largest_cavity_diameter", {}) or {}
            pv = z.get("pore_volume", {}) or {}
            sa = z.get("surface_area", {}) or {}

            row = dict(row)
            row["PLD_A"] = self._clean_numeric(diam.get("PLD_free_sphere_A"))
            row["LCD_A"] = self._clean_numeric(
                diam.get("LCD_included_sphere_A") or lcd.get("LCD_included_sphere_A")
            )
            row["pore_volume_cm3_g"] = self._clean_numeric(pv.get("AV_cm3_g"))
            row["accessible_volume_fraction"] = self._clean_numeric(pv.get("AV_volume_fraction"))
            row["surface_area_m2_g"] = self._clean_numeric(sa.get("ASA_m2_g"))
            rows.append(row)

        descriptor_keys = ["PLD_A", "LCD_A", "pore_volume_cm3_g", "accessible_volume_fraction", "surface_area_m2_g"]
        correlations: Dict[str, Any] = {}
        for key in descriptor_keys:
            xs = []
            ys = []
            for row in rows:
                x = row.get(key)
                y = row.get("uptake_excess")
                if x is None or y is None:
                    continue
                xs.append(float(x))
                ys.append(float(y))
            correlations[key] = self._correlate(xs, ys)

        trend_status, trend_note = self._trend_status_for_rows(rows, correlations, "Zeo++ structure descriptors")

        return {
            "method": "uptake_structure_relationship",
            "definition": {
                "target": "uptake_excess",
                "descriptors": descriptor_keys,
                "interpretation": "positive correlation means uptake increases as the pore descriptor increases.",
            },
            "rows": rows,
            "uptake_ranking": sorted(rows, key=lambda r: r["uptake_excess"], reverse=True),
            "correlations_with_uptake": correlations,
            "trend_status": trend_status,
            "trend_note": trend_note,
            "limitations": [
                "Zeo++ descriptors are global geometric descriptors and do not identify chemical adsorption sites.",
                "Single-pressure uptake mixes affinity and capacity effects; pressure-dependent isotherms separate them better.",
            ],
        }

    def _analyze_energy_histograms_any(self, context: Dict[str, Any]) -> Dict[str, Any]:
        upstream = context.get("upstream_plans", {}) or {}
        out: Dict[str, Any] = {}

        try:
            from analysis.energy_histogram import run_energy_histogram_analysis
        except Exception as exc:
            return {"status": "import_failed", "error": str(exc)}

        for plan_name, plan_blob in upstream.items():
            if not isinstance(plan_blob, dict):
                continue

            for job_id, job in plan_blob.items():
                if not isinstance(job, dict):
                    continue
                if job.get("agent") != "RASPAAgent":
                    continue

                work_dir_value = job.get("work_dir")
                if not work_dir_value:
                    work_dir_value = (job.get("results", {}) or {}).get("work_dir")
                if not work_dir_value:
                    continue

                work_dir = Path(str(work_dir_value))
                hist_dir = work_dir / "EnergyHistograms" / "System_0"
                if not hist_dir.exists():
                    continue

                mof = job.get("mof") or job.get("MOF") or work_dir.name.split("_")[0]
                guest = job.get("guest") or "guest"
                output_dir = work_dir / "energy_histogram_analysis"

                try:
                    result = run_energy_histogram_analysis(
                        work_dir=work_dir,
                        mof=mof,
                        output_dir=output_dir,
                        make_plot=True,
                    )
                except Exception as exc:
                    out.setdefault(mof, {})
                    out[mof].setdefault(guest, {})
                    out[mof][guest][str(job_id)] = {
                        "status": "failed",
                        "work_dir": str(work_dir),
                        "error": str(exc),
                    }
                    continue

                out.setdefault(mof, {})
                out[mof].setdefault(guest, {})
                out[mof][guest][str(job_id)] = result

        return out

    def _analyze_adsorption_site_density_any(self, context: Dict[str, Any]) -> Dict[str, Any]:
        upstream = context.get("upstream_plans", {}) or {}
        analysis_options = context.get("analysis_options", {}) or {}
        out: Dict[str, Any] = {}
        seen_work_dirs = set()

        try:
            from analysis.adsorption_site_density import run_adsorption_site_density_analysis
        except Exception as exc:
            return {"status": "import_failed", "error": str(exc)}

        chemistry_summary_path = None
        linker_blob = ((context.get("analysis") or {}).get("linker_chemistry") or {})
        if isinstance(linker_blob, dict) and linker_blob.get("summary_json"):
            candidate = Path(str(linker_blob["summary_json"]))
            if candidate.exists():
                chemistry_summary_path = candidate

        guest_label_overrides = analysis_options.get("adsorption_site_guest_labels", {}) or {}
        common_guest_labels = {
            "CH4": "methane",
            "CO2": "CO2",
            "N2": "N2",
            "H2": "H2",
        }

        for plan_blob in upstream.values():
            if not isinstance(plan_blob, dict):
                continue
            for job_id, job in plan_blob.items():
                if not isinstance(job, dict) or job.get("agent") != "RASPAAgent":
                    continue

                work_dir_value = job.get("work_dir") or (job.get("results", {}) or {}).get("work_dir")
                if not work_dir_value:
                    continue
                work_dir = Path(str(work_dir_value))
                work_dir_key = str(work_dir.resolve())
                if work_dir_key in seen_work_dirs:
                    continue

                vtk_root = work_dir / "VTK" / "System_0"
                if not vtk_root.exists() or not any(vtk_root.glob("COMDensityProfile_*.vtk")):
                    continue
                seen_work_dirs.add(work_dir_key)

                mof = job.get("mof") or job.get("MOF") or work_dir.name.split("_")[0]
                guest = str(job.get("guest") or "guest")
                guest_label = (
                    guest_label_overrides.get(guest)
                    or common_guest_labels.get(guest.upper())
                    or guest
                )

                try:
                    result = run_adsorption_site_density_analysis(
                        work_dir=work_dir,
                        mof=str(mof),
                        guest_label=str(guest_label),
                        percentile=float(analysis_options.get("adsorption_site_density_percentile", 99.75)),
                        max_points=int(analysis_options.get("adsorption_site_density_max_points", 1000)),
                        cutoff_A=float(analysis_options.get("adsorption_site_contact_cutoff_A", 4.0)),
                        top_contacts_per_point=int(
                            analysis_options.get("adsorption_site_top_contacts_per_point", 8)
                        ),
                        top_k=int(analysis_options.get("adsorption_site_top_k", 15)),
                        chemistry_summary_json=chemistry_summary_path,
                        output_dir=work_dir / "adsorption_site_density_analysis",
                    )
                except Exception as exc:
                    out.setdefault(str(mof), {})
                    out[str(mof)].setdefault(guest, {})
                    out[str(mof)][guest][str(job_id)] = {
                        "status": "failed",
                        "work_dir": str(work_dir),
                        "error": str(exc),
                    }
                    continue

                out.setdefault(str(mof), {})
                out[str(mof)].setdefault(guest, {})
                out[str(mof)][guest][str(job_id)] = result

        return out

    def _extract_lammps_material_properties_any(self, context: Dict[str, Any]) -> Dict[str, Any]:
        upstream = context.get("upstream_plans", {}) or {}
        out: Dict[str, Any] = {}

        for plan_blob in upstream.values():
            if not isinstance(plan_blob, dict):
                continue
            for job_id, job in plan_blob.items():
                if not isinstance(job, dict) or job.get("agent") != "LAMMPSAgent":
                    continue

                results = job.get("results", {}) or {}
                thermal = results.get("thermal_expansion")
                youngs = results.get("youngs_modulus")
                if not thermal and not youngs:
                    continue

                mof = job.get("mof") or job.get("MOF") or "unknown_mof"
                record = {
                    "property": job.get("property") or job.get("simulation_property"),
                    "work_dir": job.get("work_dir") or results.get("work_dir"),
                }
                if thermal:
                    record["thermal_expansion"] = thermal
                if youngs:
                    record["youngs_modulus"] = youngs
                out.setdefault(str(mof), {})[str(job_id)] = record

        return out

    def _build_binding_structure_evidence(
        self,
        binding_energy_summary: Dict[str, Any],
        zeopp_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        rows: List[Dict[str, Any]] = []

        for mof, be in sorted((binding_energy_summary or {}).items()):
            if not isinstance(be, dict):
                continue
            z = zeopp_summary.get(mof, {}) if isinstance(zeopp_summary, dict) else {}
            diam = z.get("pore_diameter", {}) or {}
            lcd = z.get("largest_cavity_diameter", {}) or {}
            pv = z.get("pore_volume", {}) or {}
            sa = z.get("surface_area", {}) or {}

            e_bind = self._clean_numeric(be.get("E_bind_ev"))
            if e_bind is None:
                continue

            row = {
                "mof": mof,
                "E_bind_ev": e_bind,
                "binding_strength_ev": -e_bind,
                "PLD_A": self._clean_numeric(diam.get("PLD_free_sphere_A")),
                "LCD_A": self._clean_numeric(
                    diam.get("LCD_included_sphere_A") or lcd.get("LCD_included_sphere_A")
                ),
                "pore_volume_cm3_g": self._clean_numeric(pv.get("AV_cm3_g")),
                "surface_area_m2_g": self._clean_numeric(sa.get("ASA_m2_g")),
            }
            rows.append(row)

        descriptor_keys = ["PLD_A", "LCD_A", "pore_volume_cm3_g", "surface_area_m2_g"]
        correlations: Dict[str, Any] = {}
        for key in descriptor_keys:
            xs = []
            ys = []
            for row in rows:
                x = row.get(key)
                y = row.get("binding_strength_ev")
                if x is None or y is None:
                    continue
                xs.append(float(x))
                ys.append(float(y))
            correlations[key] = self._correlate(xs, ys)

        ranked = sorted(rows, key=lambda row: row["E_bind_ev"])
        strongest = ranked[0]["mof"] if ranked else None
        weakest = ranked[-1]["mof"] if ranked else None

        interpretable = [k for k, v in correlations.items() if v.get("status") in {"ok", "computed_from_two_points"}]
        if len(rows) < 2:
            trend_status = "insufficient_data"
            trend_note = "Need binding energies for at least two MOFs to compare structure-binding trends."
        elif not interpretable:
            trend_status = "insufficient_descriptor_overlap"
            trend_note = "Binding energies were available, but matching Zeo++ descriptors were missing or constant."
        elif len(rows) < 3:
            trend_status = "qualitative_only"
            trend_note = "Only two MOFs are available; correlations are direction checks, not robust trends."
        else:
            trend_status = "ok"
            trend_note = "Correlations are exploratory and should not be treated as causal proof."

        return {
            "method": "binding_structure_relationship",
            "definition": {
                "binding_strength_ev": "-E_bind_ev; larger means stronger binding",
                "correlation_target": "binding_strength_ev",
                "interpretation": "positive correlation means the descriptor increases with stronger binding",
            },
            "rows": rows,
            "binding_energy_ranking": ranked,
            "strongest_binding_mof": strongest,
            "weakest_binding_mof": weakest,
            "correlations_with_binding_strength": correlations,
            "trend_status": trend_status,
            "trend_note": trend_note,
            "limitations": [
                "Binding energy is adsorption-geometry and site dependent.",
                "Zeo++ descriptors are global pore descriptors and may not identify the local binding site.",
                "Small MOF counts support qualitative trends only.",
            ],
        }

    
    
    
    def run(
        self,
        context_or_contexts: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        
        if isinstance(context_or_contexts, dict):
            return self._run_single(context_or_contexts)

        
        
        batch_ctx = self._build_batch_context(context_or_contexts)
        return self._run_single(batch_ctx)

    @staticmethod
    def _render_interpretation_report(
        report_sections: Sequence[Union[ReportSectionModel, Dict[str, Any]]],
    ) -> str:
        rendered: List[str] = []
        for section in report_sections:
            if isinstance(section, BaseModel):
                section_data = _pydantic_dump(section)
            elif isinstance(section, dict):
                section_data = section
            else:
                continue
            heading = " ".join(str(section_data.get("heading") or "").split())
            body = str(section_data.get("body") or "").strip()
            if not heading or not body:
                continue
            rendered.append(f"### {heading}\n\n{body}")
        return "\n\n".join(rendered)

    def _store_interpretation(
        self,
        context: Dict[str, Any],
        trace: List[Dict[str, Any]],
        interpretation: InterpretationModel,
    ) -> None:
        headings = [
            section.heading.strip().lower()
            for section in interpretation.report_sections
        ]

        def is_limitations_heading(heading: str) -> bool:
            return heading.startswith(("limitation", "uncertaint"))

        has_overall = any(
            "overall" in heading and "interpret" in heading
            for heading in headings
        )
        has_limitations = any(is_limitations_heading(heading) for heading in headings)

        if not has_overall and interpretation.summary.strip():
            overall_section = ReportSectionModel(
                heading="Overall Interpretation",
                body=interpretation.summary.strip(),
            )
            limitation_index = next(
                (
                    index
                    for index, heading in enumerate(headings)
                    if is_limitations_heading(heading)
                ),
                len(interpretation.report_sections),
            )
            interpretation.report_sections.insert(
                limitation_index,
                overall_section,
            )

        if not has_limitations:
            limitation_parts = [
                str(item).strip()
                for item in interpretation.uncertainties
                if str(item).strip()
            ]
            if interpretation.next_best_step.strip():
                limitation_parts.append(
                    f"Next step: {interpretation.next_best_step.strip()}"
                )
            if limitation_parts:
                interpretation.report_sections.append(
                    ReportSectionModel(
                        heading="Limitations and Next Step",
                        body=" ".join(limitation_parts),
                    )
                )

        interp_dump = _pydantic_dump(interpretation)
        context["analysis"]["interpretation"] = interp_dump
        formatted_report = self._render_interpretation_report(
            interpretation.report_sections
        )
        if formatted_report:
            context["analysis"]["formatted_report"] = formatted_report
        else:
            context["analysis"].pop("formatted_report", None)
        self._trace(trace, "interpretation", interp_dump)

    def _run_single(self, context: Dict[str, Any]) -> Dict[str, Any]:
        context.setdefault("analysis", {})
        trace = context["analysis"].setdefault("trace", [])
        self._trace(trace, "start", {"agent": self.agent_name})
        

        interpret_only = bool(context.get("interpret_only") or context["analysis"].get("interpret_only"))
        try:
            if interpret_only:
                recommendation = context.get("analysis_recommendation", {}) or {}
                plan_blob = recommendation.get("analysis_plan", {}) or {}
                try:
                    analysis_plan = self._validate_selected_plan(
                        _pydantic_validate(
                            SimulationPlanModel,
                            plan_blob,
                        )
                    )
                except (TypeError, ValueError, ValidationError):
                    analysis_plan = SimulationPlanModel(steps=[])
                context["analysis"]["plan"] = _pydantic_dump(analysis_plan)
                evidence: Dict[str, Any] = {"mode": "interpret_only"}

                interp = self._step_interpretation(
                    context,
                    goal="",
                    hypothesis="",
                    plan=analysis_plan,
                    evidence=evidence,
                )
                self._store_interpretation(context, trace, interp)
                self._trace(trace, "end", {"agent": self.agent_name})
                return context

            goal = self._step_goal(context)
            context["analysis"]["goal"] = goal.goal
            self._trace(trace, "goal", _pydantic_dump(goal))

            hyp = self._step_hypothesis(context, goal.goal)
            context["analysis"]["hypothesis"] = hyp.hypothesis
            self._trace(trace, "hypothesis", _pydantic_dump(hyp))

            plan = self._step_plan(context, goal.goal, hyp.hypothesis)
            plan_dump = _pydantic_dump(plan)
            context["analysis"]["plan"] = plan_dump
            self._trace(trace, "plan", plan_dump)

            evidence: Dict[str, Any] = {}

            context["analysis"]["evidence"] = evidence

            interp = self._step_interpretation(context, goal.goal, hyp.hypothesis, plan, evidence)
            self._store_interpretation(context, trace, interp)

        except ValidationError as ve:
            context["analysis"]["error"] = {"type": "ValidationError", "message": str(ve)}
            self._trace(trace, "error", context["analysis"]["error"])
        except Exception as e:
            context["analysis"]["error"] = {"type": type(e).__name__, "message": str(e)}
            self._trace(trace, "error", context["analysis"]["error"])

        self._trace(trace, "end", {"agent": self.agent_name})
        return context

    def _build_batch_context(
        self,
        contexts: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not contexts:
            raise ValueError("AnalysisAgent: no contexts provided for batch mode")

        first = contexts[0]

        query_text = first.get("query_text") or first.get("QueryText") or ""
        job_name   = first.get("batch_job_name") or first.get("job_name", "")
        prop       = (
            first.get("property")
            or first.get("property_name")
            or first.get("simulation_property")
            or ""
        )
        guest      = first.get("guest")

        batch_results: Dict[str, Any] = {}
        per_mof_info: Dict[str, Any] = {}

        for idx, ctx in enumerate(contexts):
            results = ctx.get("results", {}) or {}

            
            mof_name = (
                ctx.get("mof")
                or ctx.get("MOF")
                or ctx.get("job_name")
                or f"system_{idx}"
            )

            batch_results[mof_name] = results

            per_mof_info[mof_name] = {
                "work_dir": ctx.get("work_dir"),
                "job_name": ctx.get("job_name"),
                "mof": ctx.get("mof", mof_name),
                "guest": ctx.get("guest", guest),
                "property": ctx.get("property", prop),
            }

        batch_context: Dict[str, Any] = {
            "job_name": job_name,
            "property": prop,
            "guest": guest,
            "query_text": query_text,
            "results": batch_results,
            "per_mof_info": per_mof_info,
        }

        return batch_context
    
    
    
    def _call_llm(self, messages: List[Any], label: str = "analysis_llm_call") -> str:
        from core.llm_logging import set_llm_context
        set_llm_context("AnalysisAgent", label)
        llm_obj = self.llm


        if callable(llm_obj) and not hasattr(llm_obj, "invoke"):
            try:
                candidate = llm_obj()
                if hasattr(candidate, "invoke"):
                    llm_obj = candidate
            except TypeError:

                pass

        if hasattr(llm_obj, "invoke"):
            resp = llm_obj.invoke(messages)
            return str(getattr(resp, "content", resp))
        else:
            resp = llm_obj(messages)
            if hasattr(resp, "content"):
                return str(resp.content)
            return str(resp)

    def _safe_json_loads(self, text: str) -> Dict[str, Any]:
        t = str(text).strip()
        if t.startswith("```"):
            lines = t.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            t = "\n".join(lines).strip()
        return json.loads(t)

    def _invoke_llm_json(self, prompt: str, model_cls):
        messages = [SystemMessage(content=DEFAULT_SYSTEM), HumanMessage(content=prompt)]
        raw = self._call_llm(messages, label="invoke_llm_json")
        obj = self._safe_json_loads(raw)
        return _pydantic_validate(model_cls, obj)

    
    
    
    def _step_goal(self, context: Dict[str, Any]) -> ExplanationGoalModel:
        q = context.get("query_text", context.get("QueryText", ""))
        results = context.get("results", {})
        prompt = f"""{DEFAULT_SYSTEM}

Task: Define a concise analysis goal based on the user's question and any available results.

User question:
{q}

Available results keys:
{list(results.keys())}

Return JSON:
{{"goal":"..."}}"""
        return self._invoke_llm_json(prompt, ExplanationGoalModel)

    
    
    
    def _step_hypothesis(self, context: Dict[str, Any], goal: str) -> HypothesisModel:
        q = context.get("query_text", context.get("QueryText", ""))
        prompt = f"""{DEFAULT_SYSTEM}

Task: Propose exactly ONE testable hypothesis.

Goal:
{goal}

User question:
{q}

The hypothesis must:
- address the user's stated scope without adding a different scientific question;
- identify one observable mechanism or relationship that could be supported or rejected;
- avoid assuming a result that has not been computed;
- remain testable using one or more tools from the catalog that will be shown in the next planning step.

Return JSON:
{{"hypothesis":"..."}}"""
        return self._invoke_llm_json(prompt, HypothesisModel)

    
    
    
    def _step_plan(self, context: Dict[str, Any], goal: str, hypothesis: str) -> SimulationPlanModel:
        q = context.get("query_text", context.get("QueryText", ""))
        tool_catalog = [
            {
                "method": method,
                "engine": spec["engine"],
                "category": spec.get("category") or (
                    "Calculation"
                    if spec["engine"] in ENGINE_AGENT_MAP
                    else "Post-processing"
                ),
                "description": spec["description"],
                "data_needs": spec["data_needs"],
                "produces": spec["produces"],
                "cost": spec["cost"],
                **({"implementation": spec["tool"]} if spec.get("tool") else {}),
            }
            for method, spec in ANALYSIS_METHODS.items()
        ]
        prompt = f"""{DEFAULT_SYSTEM}

Select the tools needed to answer the question and test the hypothesis by reasoning
from the catalog descriptions. There are no automatically inserted prerequisite
tools. Therefore:

- Explicitly select every calculation, post-processing, structure/chemistry, and
  interpretation tool whose output will be used.
- Infer needed inputs from each tool's `data_needs` description. If an input is not
  already available, select a catalog tool that can produce it.
- Prefer the smallest scientifically sufficient set, considering the stated cost.
- Do not select a tool merely because it is commonly paired with another tool.
- Keep each method at most once and use only catalog method names.
- For every step, give a concrete `reason` tied to the user question or hypothesis.

Tool catalog:
{json.dumps(tool_catalog, ensure_ascii=False, indent=2)}

Goal:
{goal}

Hypothesis:
{hypothesis}

User question:
{q}

Return JSON:
{{
  "steps": [
    {{"name": "...", "method": "...", "reason": "..."}}
  ]
}}"""
        plan = self._invoke_llm_json(prompt, SimulationPlanModel)

        from config import ask_user_confirmation
        import json as _json

        plan = self._validate_selected_plan(plan)
        plan_summary = "\n".join(
            f"  Step {i+1}: {s.name} ({s.method})"
            + (f" - {s.reason}" if s.reason else "")
            for i, s in enumerate(plan.steps)
        )
        print(f"\n[AnalysisAgent] Proposed simulation plan:\n{plan_summary}")

        def _reinvoke_plan(instruction: str) -> str:
            from langchain.schema import HumanMessage as HM, SystemMessage as SM
            revised = prompt + f"\n\nUser instruction: {instruction}\nRevise your plan accordingly."
            return self._call_llm([HM(content=revised)], label="reinvoke_plan")

        action, revised_text = ask_user_confirmation(
            "AnalysisAgent", plan_summary, reinvoke_fn=_reinvoke_plan, required=True
        )
        if action == "apply" and revised_text != plan_summary:
            try:
                obj2 = self._safe_json_loads(revised_text)
                plan = self._validate_selected_plan(
                    _pydantic_validate(SimulationPlanModel, obj2)
                )
                print("[AnalysisAgent] Plan updated per user instruction.")
            except Exception:
                pass

        return plan

    @staticmethod
    def _validate_selected_plan(plan: SimulationPlanModel) -> SimulationPlanModel:
        selected: List[PlanStepModel] = []
        seen = set()
        for step in plan.steps:
            if step.method not in ANALYSIS_METHODS:
                raise ValueError(f"Unknown analysis method: {step.method}")
            if step.method in seen:
                continue
            selected.append(step)
            seen.add(step.method)
        return SimulationPlanModel(steps=selected)

    
    
    
    def _trace(self, trace_list: List[Dict[str, Any]], event: str, data: Dict[str, Any]) -> None:
        trace_list.append({"t": time.time(), "event": event, "data": data})

    
    
    
    
    def _step_interpretation(
        self,
        context: Dict[str, Any],
        goal: str,
        hypothesis: str,
        plan: SimulationPlanModel,
        evidence: Dict[str, Any],
    ) -> InterpretationModel:
        q = context.get("query_text") or context.get("QueryText") or ""
        plan_methods = {step.method for step in plan.steps}
        precomputed_results = context.get("results", {}) or {}

        if "bader_charge" in plan_methods:
            context = self._run_bader_summaries_any(context, top_k=5)

        analysis_blob = context.get("analysis", {}) or {}
        bader_summary = analysis_blob.get("bader_summary", {}) or {}

        
        upstream = context.get("upstream_plans", {}) or {}
        binding_energy_summary: Dict[str, Any] = {}

        for plan_name, plan_blob in (
            upstream.items() if "binding_energy" in plan_methods else []
        ):
            if not isinstance(plan_blob, dict):
                continue
            if not plan_name.endswith("_binding_energy"):
                continue

            
            jm = plan_blob.get(f"{plan_name}_mof", {})
            jg = plan_blob.get(f"{plan_name}_guest", {})
            jc = plan_blob.get(f"{plan_name}_complex", {})

            try:
                Emof = float(jm.get("results", {}).get("vasp_energy_ev"))
                Eguest = float(jg.get("results", {}).get("vasp_energy_ev"))
                Ecomplex = float(jc.get("results", {}).get("vasp_energy_ev"))
            except (TypeError, ValueError):
                
                continue

            
            mof_name = jm.get("mof") or plan_blob.get("mof") or plan_name.split("_CO2_")[0]

            Ebind = Ecomplex - (Emof + Eguest)
            complex_results = jc.get("results", {}) or {}
            interaction = complex_results.get("interaction_energy") or {}
            deformation = complex_results.get("structure_deformation") or {}
            energy_record = {
                "E_bind_ev": Ebind,
                "E_ads_relaxed_ev": Ebind,
                "E_mof_opt_ev": Emof,
                "E_guest_opt_ev": Eguest,
                "E_complex_opt_ev": Ecomplex,
                "relaxed_equation": (
                    "E_ads_relaxed = E(MOF+guest,opt) - E(MOF,opt) - E(guest,opt)"
                ),
                "relaxed_energy_includes": [
                    "direct host-guest interaction",
                    "MOF deformation",
                    "guest deformation",
                    "cell deformation when the cell was relaxed",
                ],
                "structure_deformation": deformation,
                "deformation_threshold_exceeded": bool(
                    deformation.get("threshold_exceeded")
                ),
                "interaction_energy_status": interaction.get("status"),
                "convention": (
                    "more negative relaxed adsorption or interaction energy "
                    "means stronger stabilization"
                ),
            }
            if interaction.get("status") == "ok" and interaction.get("E_int_ev") is not None:
                Eint = float(interaction["E_int_ev"])
                energy_record.update(
                    {
                        "E_int_ev": Eint,
                        "interaction_equation": interaction.get("equation"),
                        "deformation_contribution_ev": Ebind - Eint,
                    }
                )
            elif interaction:
                energy_record["interaction_energy"] = interaction
            binding_energy_summary[mof_name] = energy_record

        if binding_energy_summary:
            context.setdefault("results", {})[
                "vasp_adsorption_energies"
            ] = binding_energy_summary
        
        needs_zeopp_evidence = any(
            (ANALYSIS_METHODS.get(method) or {}).get("engine") == "Zeo++"
            for method in plan_methods
        )
        needs_open_metal_sites = "open_metal_site_analysis" in plan_methods
        needs_pore_surface_chemistry = (
            "pore_surface_chemistry_analysis" in plan_methods
        )
        needs_linker_functional_groups = (
            "linker_functional_group_analysis" in plan_methods
        )
        needs_linker_chemistry = "linker_chemistry_analysis" in plan_methods
        linker_chemistry_summary = (
            self._analyze_linker_chemistry_any(context) if needs_linker_chemistry else {}
        )
        if linker_chemistry_summary:
            context.setdefault("analysis", {})["linker_chemistry"] = linker_chemistry_summary
        open_metal_site_summary = (
            self._analyze_open_metal_sites_any(context)
            if needs_open_metal_sites
            else {}
        )
        linker_functional_group_summary = (
            self._analyze_linker_functional_groups_any(
                context,
                linker_chemistry_summary,
            )
            if needs_linker_functional_groups
            else {}
        )
        pore_surface_chemistry_summary = (
            self._analyze_pore_surface_chemistry_any(
                context,
                linker_chemistry_summary,
            )
            if needs_pore_surface_chemistry
            else {}
        )
        if open_metal_site_summary:
            context.setdefault("analysis", {})[
                "open_metal_sites"
            ] = open_metal_site_summary
        if linker_functional_group_summary:
            context.setdefault("analysis", {})[
                "linker_functional_groups"
            ] = linker_functional_group_summary
        if pore_surface_chemistry_summary:
            context.setdefault("analysis", {})[
                "pore_surface_chemistry"
            ] = pore_surface_chemistry_summary

        zeopp_summary = (
            self._extract_zeopp_summaries_any(context)
            if needs_zeopp_evidence
            else {}
        )
        if needs_zeopp_evidence and not zeopp_summary:
            zeopp_summary = (
                precomputed_results.get("zeopp_summary")
                or precomputed_results.get("auto_zeopp_summary")
                or {}
            )
        diff_summary = (
            self._extract_diffusivity_summaries_any(context)
            if "diffusivity" in plan_methods
            else {}
        )
        msd_summary = (
            self._extract_msd_summaries_any(context)
            if "msd" in plan_methods
            else {}
        )
        material_methods = {"thermal_expansion", "youngs_modulus"}
        lammps_material_properties = (
            self._extract_lammps_material_properties_any(context)
            if plan_methods & material_methods
            else {}
        )
        trajectory_methods = {
            "anisotropic_diffusion",
            "rdf_guest_host_contact",
            "residence_hopping",
            "van_hove_non_gaussian",
            "velocity_autocorrelation",
            "energy_autocorrelation",
            "node_linker_contact",
            "pore_network_hopping_graph",
        }
        lammps_trajectory_analysis = (
            self._analyze_lammps_trajectories_any(context)
            if plan_methods & trajectory_methods
            else {}
        )
        diffusion_meta_methods = {
            "diffusion_activation_barrier",
            "diffusion_replicate_consistency",
        }
        lammps_diffusion_meta = (
            self._analyze_lammps_diffusion_meta_any(context)
            if plan_methods & diffusion_meta_methods
            else {}
        )
        uptake_summary = (
            self._extract_raspa_uptake_summaries_any(context)
            if "uptake" in plan_methods
            else {}
        )
        if "uptake" in plan_methods and not uptake_summary:
            uptake_summary = precomputed_results.get("uptake") or {}
        selectivity_summary = (
            self._extract_raspa_selectivity_summaries_any(context)
            if "selectivity" in plan_methods
            else {}
        )
        isotherm_shape_analysis = (
            self._analyze_raspa_isotherms_any(context)
            if "isotherm_shape_analysis" in plan_methods
            else {}
        )
        if isotherm_shape_analysis.get("series"):
            context.setdefault("analysis", {})["raspa_isotherm_shape"] = isotherm_shape_analysis
        henry_summary = (
            self._extract_raspa_henry_summaries_any(context)
            if "henry_coefficient" in plan_methods
            else {}
        )
        if "henry_coefficient" in plan_methods and not henry_summary:
            henry_summary = (
                precomputed_results.get("henry")
                or precomputed_results.get("henry_coefficient")
                or {}
            )
        thermo_summary = (
            self._extract_raspa_thermodynamics_summaries_any(context)
            if "heat_of_adsorption" in plan_methods
            else {}
        )
        if "heat_of_adsorption" in plan_methods and not thermo_summary:
            thermo_summary = (
                precomputed_results.get("thermodynamics")
                or precomputed_results.get("heat_of_adsorption")
                or {}
            )
        adsorption_regime_analysis = (
            self._analyze_adsorption_regime(uptake_summary)
            if "adsorption_regime_analysis" in plan_methods
            else {}
        )
        uptake_henry_evidence = self._build_uptake_henry_evidence(
            uptake_summary,
            henry_summary,
        )
        uptake_qst_evidence = self._build_uptake_qst_evidence(
            uptake_summary,
            thermo_summary,
        )
        uptake_structure_evidence = self._build_uptake_structure_evidence(
            uptake_summary,
            zeopp_summary,
        )
        energy_histogram_analysis = (
            self._analyze_energy_histograms_any(context)
            if "energy_histogram_analysis" in plan_methods
            else {}
        )
        adsorption_site_density_analysis = (
            self._analyze_adsorption_site_density_any(context)
            if "adsorption_site_density_analysis" in plan_methods
            else {}
        )
        uptake_basis_analysis = (
            self._analyze_uptake_basis_comparison(uptake_summary)
            if "uptake_basis_comparison" in plan_methods
            else {}
        )
        uptake_evidence_synthesis = (
            self._build_uptake_evidence_synthesis(
                uptake_summary,
                henry_summary,
                thermo_summary,
                zeopp_summary,
                energy_histogram_analysis,
            )
            if "uptake_analysis" in plan_methods
            else {}
        )
        binding_structure_evidence = (
            self._build_binding_structure_evidence(
                binding_energy_summary,
                zeopp_summary,
            )
            if "binding_analysis" in plan_methods
            else {}
        )
        binding_configuration_summary = (
            self._analyze_binding_configurations_any(context)
            if "binding_configuration_analysis" in plan_methods
            else {}
        )
        binding_pdos_summary = (
            self._build_binding_pdos_evidence_any(context)
            if "projected_dos" in plan_methods
            else {}
        )

        def material_property_summary(property_name: str) -> Dict[str, Any]:
            selected: Dict[str, Any] = {}
            for mof, jobs in lammps_material_properties.items():
                if not isinstance(jobs, dict):
                    continue
                for job_id, record in jobs.items():
                    if not isinstance(record, dict) or not record.get(property_name):
                        continue
                    selected.setdefault(mof, {})[job_id] = {
                        property_name: record[property_name],
                        "property": record.get("property"),
                        "work_dir": record.get("work_dir"),
                    }
            return selected

        thermal_expansion_summary = material_property_summary("thermal_expansion")
        mechanical_response_summary = material_property_summary("youngs_modulus")

        uptake_evidence_bundle: Dict[str, Any] = {
            "primary_result": uptake_summary,
            "independent_evidence": {
                "henry_coefficient": henry_summary,
                "thermodynamics": thermo_summary,
                "pore_descriptors": zeopp_summary,
                "energy_histogram": energy_histogram_analysis,
                "adsorption_site_density": adsorption_site_density_analysis,
                "loading_basis": uptake_basis_analysis,
                "adsorption_regime": adsorption_regime_analysis,
                "isotherm_shape": isotherm_shape_analysis,
            },
            "derived_relationships": {
                "henry_affinity": uptake_henry_evidence,
                "adsorption_heat": uptake_qst_evidence,
                "pore_structure": uptake_structure_evidence,
            },
            "evidence_synthesis": uptake_evidence_synthesis,
        }
        binding_evidence_bundle: Dict[str, Any] = {
            "primary_result": binding_energy_summary,
            "independent_evidence": {
                "configuration": binding_configuration_summary,
                "open_metal_sites": open_metal_site_summary,
                "bader_charge": bader_summary,
                "projected_dos": binding_pdos_summary,
                "pore_descriptors": zeopp_summary,
                "linker_chemistry": linker_chemistry_summary,
                "linker_functional_groups": linker_functional_group_summary,
                "pore_surface_chemistry": pore_surface_chemistry_summary,
            },
            "derived_relationships": {
                "pore_structure": binding_structure_evidence,
            },
        }
        charge_transfer_evidence_bundle: Dict[str, Any] = {
            "primary_result": bader_summary,
            "independent_evidence": {
                "binding_energy": binding_energy_summary,
                "configuration": binding_configuration_summary,
                "projected_dos": binding_pdos_summary,
                "open_metal_sites": open_metal_site_summary,
                "linker_functional_groups": linker_functional_group_summary,
                "pore_surface_chemistry": pore_surface_chemistry_summary,
            },
        }
        electronic_structure_evidence_bundle: Dict[str, Any] = {
            "primary_result": binding_pdos_summary,
            "independent_evidence": {
                "binding_energy": binding_energy_summary,
                "bader_charge": bader_summary,
                "configuration": binding_configuration_summary,
                "open_metal_sites": open_metal_site_summary,
                "linker_chemistry": linker_chemistry_summary,
            },
        }
        henry_evidence_bundle: Dict[str, Any] = {
            "primary_result": henry_summary,
            "independent_evidence": {
                "uptake": uptake_summary,
                "thermodynamics": thermo_summary,
                "binding_energy": binding_energy_summary,
                "pore_descriptors": zeopp_summary,
                "open_metal_sites": open_metal_site_summary,
                "pore_surface_chemistry": pore_surface_chemistry_summary,
            },
            "derived_relationships": {
                "uptake_at_low_pressure": uptake_henry_evidence,
            },
        }
        heat_of_adsorption_evidence_bundle: Dict[str, Any] = {
            "primary_result": thermo_summary,
            "independent_evidence": {
                "uptake": uptake_summary,
                "henry_coefficient": henry_summary,
                "energy_histogram": energy_histogram_analysis,
                "binding_energy": binding_energy_summary,
                "open_metal_sites": open_metal_site_summary,
                "pore_surface_chemistry": pore_surface_chemistry_summary,
            },
            "derived_relationships": {
                "uptake": uptake_qst_evidence,
            },
        }
        selectivity_evidence_bundle: Dict[str, Any] = {
            "primary_result": selectivity_summary,
            "independent_evidence": {
                "component_uptake": uptake_summary,
                "henry_coefficient": henry_summary,
                "thermodynamics": thermo_summary,
                "pore_descriptors": zeopp_summary,
                "adsorption_site_density": adsorption_site_density_analysis,
                "pore_surface_chemistry": pore_surface_chemistry_summary,
                "diffusivity": diff_summary,
            },
        }
        pore_structure_evidence_bundle: Dict[str, Any] = {
            "primary_result": zeopp_summary,
            "independent_evidence": {
                "linker_chemistry": linker_chemistry_summary,
                "open_metal_sites": open_metal_site_summary,
                "linker_functional_groups": linker_functional_group_summary,
                "pore_surface_chemistry": pore_surface_chemistry_summary,
            },
        }
        diffusion_evidence_bundle: Dict[str, Any] = {
            "primary_result": diff_summary,
            "independent_evidence": {
                "msd": msd_summary,
                "trajectory": lammps_trajectory_analysis,
                "diffusion_meta": lammps_diffusion_meta,
                "pore_descriptors": zeopp_summary,
                "pore_surface_chemistry": pore_surface_chemistry_summary,
                "linker_chemistry": linker_chemistry_summary,
            },
        }
        thermal_expansion_evidence_bundle: Dict[str, Any] = {
            "primary_result": thermal_expansion_summary,
            "independent_evidence": {
                "pore_descriptors": zeopp_summary,
                "linker_chemistry": linker_chemistry_summary,
                "pore_surface_chemistry": pore_surface_chemistry_summary,
            },
        }
        mechanical_response_evidence_bundle: Dict[str, Any] = {
            "primary_result": mechanical_response_summary,
            "independent_evidence": {
                "thermal_expansion": thermal_expansion_summary,
                "pore_descriptors": zeopp_summary,
                "linker_chemistry": linker_chemistry_summary,
                "open_metal_sites": open_metal_site_summary,
            },
        }

        results_for_prompt: Dict[str, Any] = {
            "critical_basis_normalization_check": uptake_henry_evidence.get(
                "reporting_summary",
                {},
            ),
            "critical_uptake_uncertainty_check": uptake_basis_analysis.get(
                "pairwise_resolution_95_percent",
                {},
            ),
            "evidence_bundles": {
                "uptake_analysis": uptake_evidence_bundle,
                "binding_analysis": binding_evidence_bundle,
                "charge_transfer_analysis": charge_transfer_evidence_bundle,
                "electronic_structure_analysis": electronic_structure_evidence_bundle,
                "henry_analysis": henry_evidence_bundle,
                "heat_of_adsorption_analysis": heat_of_adsorption_evidence_bundle,
                "selectivity_analysis": selectivity_evidence_bundle,
                "pore_structure_analysis": pore_structure_evidence_bundle,
                "diffusion_analysis": diffusion_evidence_bundle,
                "thermal_expansion_analysis": thermal_expansion_evidence_bundle,
                "mechanical_response_analysis": mechanical_response_evidence_bundle,
                "structure_chemistry": {
                    "linker_chemistry": linker_chemistry_summary,
                    "open_metal_sites": open_metal_site_summary,
                    "linker_functional_groups": linker_functional_group_summary,
                    "pore_surface_chemistry": pore_surface_chemistry_summary,
                    "pore_descriptors": zeopp_summary,
                },
            },
        }

        
        raw_results = context.get("results", {}) or {}
        if raw_results:
            results_for_prompt["additional_results"] = raw_results
        if (
            not henry_summary
            and not uptake_summary
            and not selectivity_summary
            and not thermo_summary
            and not uptake_henry_evidence.get("rows")
            and not uptake_qst_evidence.get("rows")
            and not uptake_structure_evidence.get("rows")
            and not energy_histogram_analysis
            and not adsorption_site_density_analysis
            and not uptake_basis_analysis.get("rows")
            and not adsorption_regime_analysis.get("rows")
            and not isotherm_shape_analysis.get("series")
            and not uptake_evidence_synthesis.get("rows")
            and not binding_energy_summary
            and not binding_structure_evidence.get("rows")
            and not binding_configuration_summary
            and not binding_pdos_summary
            and not linker_chemistry_summary
            and not open_metal_site_summary
            and not linker_functional_group_summary
            and not pore_surface_chemistry_summary
            and not bader_summary
            and not zeopp_summary
            and not diff_summary
            and not msd_summary
            and not lammps_material_properties
            and not lammps_trajectory_analysis
            and not lammps_diffusion_meta
        ):
            if raw_results:
                results_for_prompt = raw_results
            else:
                results_for_prompt = {"upstream_plans": context.get("upstream_plans", {})}

        
        prompt = f"""{DEFAULT_SYSTEM}

    Task:
    - The following "User query" is the primary instruction.
    - Interpret the available results to answer the query.
    - Use ONLY the provided results. Do NOT fabricate numbers.
    - If results are insufficient, state uncertainties and propose the single best next step.

    User query (PRIMARY INSTRUCTION):
    {q}

    Available results:
    {json.dumps(results_for_prompt, indent=2, ensure_ascii=False)}

    Optional context (may be empty; do not depend on it):
    - goal: {goal}
    - hypothesis: {hypothesis}
    - plan: {json.dumps(_pydantic_dump(plan), indent=2, ensure_ascii=False)}
    - evidence: {json.dumps(evidence, indent=2, ensure_ascii=False)}

    Notes:
    - VASP adsorption energies:
    * E_ads_relaxed includes MOF, guest, and possible cell deformation effects; do not call it a direct interaction energy.
    * E_int is the frozen-geometry host-guest interaction energy at the optimized complex geometry.
    * If both are present, report both with their distinct meanings. More negative values mean stronger stabilization.
    * If deformation_threshold_exceeded is true, explicitly warn that structural deformation reached or exceeded 20%.
    - Evidence bundles:
    * Select the bundle matching the requested Interpretation-category method.
    * Available primary bundles cover binding, charge transfer, electronic structure, uptake, Henry affinity,
      adsorption heat, selectivity, pore structure, diffusion, thermal expansion, and mechanical response.
    * Independent evidence remains distinct from derived_relationships; integrate it only when it is relevant to the user query.
    * Missing or empty evidence must not be treated as a negative result.
    - Bader summary under charge_transfer_analysis.primary_result or binding_analysis.independent_evidence.bader_charge:
    * ACF CHARGE is the integrated electron population in a Bader basin, not net atomic charge or formal oxidation state.
    * framework.delta_q_total is the framework electron-count change, complex minus isolated MOF. Positive means electron gain; negative means electron loss.
    * co2.guest_net_bader_charge_from_conservation_e is the guest net Bader charge. Negative means the guest gained electrons.
    * co2.guest_electron_gain_from_conservation_e is the same transfer expressed as electron gain, so positive means electron accumulation on the guest.
    * co2.guest_charge_sum_in_complex is only the guest's total Bader electron population in the complex. Do not call it net charge or compare it as polarization strength by itself.
    * Use framework.by_species for mapping-independent species aggregates.
    * Use framework.top_atoms_by_abs_delta_q, framework.all_atoms, and metal.top_sites_by_abs_delta_q only when framework.atom_mapping.status is "ok".
    * When atom mapping is unavailable, do not infer site-specific charge changes; use only total/species aggregates and state the mapping limitation.
    * Use metal.delta_q_total as the mapping-independent aggregate electron-count change over metal species.
    * Prefer quality_control.reference_density_mode="aeccar0_plus_aeccar2"; otherwise flag the Bader partition as insufficiently verified.
    - Binding structure relationship:
    * Use evidence_bundles.binding_analysis.derived_relationships.pore_structure.rows for the comparison table.
    * Use correlations_with_binding_strength only as exploratory trend evidence.
    * Do not claim causality from Zeo++ descriptors alone.
    - Zeo++ pore size distribution:
    * Use the pore_descriptors evidence for PSD peak, median, spread, and sampled accessibility.
    * Report the probe radius and sample count when available, and do not treat a PSD peak as a unique crystallographic pore diameter.
    - Binding configuration:
    * Prefer binding_analysis.independent_evidence.configuration nearest_binding_region and unit_contact_summary to describe where the guest sits chemically.
    * Use local_binding_environment.fingerprint as the distance-weighted contact evidence.
    * Use guest_pose_degrees_of_freedom to discuss guest orientation/position/internal distortion relative to nearby node/linker units.
    * For linker-relative poses, use plane offset, in-plane distance, and guest-axis-to-linker-plane/normal angles when available.
    * Use nearest_contacts only as atom-level supporting evidence.
    * Describe contacts as node/linker environments when chemistry_unit is available, e.g. metal paddlewheel node or imidazolate-like linker.
    * Treat distance-based contacts as coordination/contact candidates, not proof of chemical bond formation.
    * If CO2 orientation is present, use orientation.configuration_label and axis_to_site_angle_deg.
    - Projected DOS:
    * Use binding_analysis.independent_evidence.projected_dos.normalized_spectral_overlap as qualitative evidence for guest/site orbital energy coincidence.
    * Use guest_orbital_fraction and contact_site_orbital_fraction to identify the dominant s/p/d/f character in the stated energy window.
    * Do not claim bond formation or quantitative bond strength from PDOS overlap alone.
    * Do not directly compare isolated-system peak energies unless an explicit common energy alignment is available.
    - Linker chemistry:
    * Use structure_chemistry.linker_chemistry.structures[].linkers and nodes to describe decomposed building units.
    * SMILES/InChIKey and extension points are more reliable than visual interpretation alone.
    * Mention exported PNG/XYZ/CIF paths when the user wants VLM-ready artifacts.
    * Remember that cut fragments may omit carboxylate atoms if those atoms were assigned to the node side.
    - Open metal sites:
    * Use structure_chemistry.open_metal_sites.structures[].sites for the source atom index, coordination number, neighboring species, and OMS detector classification.
    * Treat is_open as the external OMS detector's geometric classification; do not replace it with a fixed coordination-number threshold.
    * Report problematic=true and mapping/library failures as uncertainty.
    - Linker functional groups:
    * Use structure_chemistry.linker_functional_groups structures[].linkers[].functional_group_fingerprint as the source of truth.
    * Distinguish SMARTS match counts from the number of linker types carrying a group.
    * Fingerprints describe disconnected linker SMILES, so cutting-point protonation or bond-order uncertainty must be retained.
    - Pore surface chemistry:
    * Use structure_chemistry.pore_surface_chemistry.structures[].target_surface_fractions for metal, O/N/S, and aromatic pore-facing surface comparisons.
    * Use surface_fraction_by_category and spatial_distribution for detailed composition and location.
    * This is probe-radius-dependent local accessibility, not proof that every sampled surface belongs to a percolating pore network.
    - RASPA thermodynamics:
    * Use heat_of_adsorption_analysis.primary_result when Qst/adsorption enthalpy is the main requested result.
    * Qst is reported as -enthalpy_of_adsorption; larger positive Qst means stronger exothermic adsorption.
    * If uptake/Henry and Qst are both available, use Qst as adsorption-strength evidence, not as a direct capacity by itself.
    - Henry affinity:
    * Use henry_analysis.primary_result for the infinite-dilution Henry coefficient.
    * Henry coefficient is an affinity descriptor and is not a saturation-capacity measurement.
    * Match normalization bases before comparing Henry coefficients with uptake.
    * A Henry coefficient in mol/kg/Pa directly corresponds to gravimetric loading in mol/kg, not volumetric loading.
    * For volumetric comparison, use K_H,vol = K_H,mass times framework density in kg/m^3.
    * When density narrows or reverses the mass-based trend, describe this explicitly as density compensation.
    * When derived Henry rows contain framework_density_g_cm3 and henry_volumetric_mol_m3_Pa,
      report those values for every compared framework rather than describing the conversion only qualitatively.
    * State the basis-specific conclusion explicitly: gravimetric uptake follows mass-based Henry,
      whereas volumetric uptake follows density-converted volumetric Henry.
    - Selectivity:
    * Use selectivity_analysis.primary_result and preserve the parsed definition (x_A/x_B)/(y_A/y_B).
    * Do not replace mixture selectivity with a ratio of separately simulated pure-component uptakes.
    * Treat diffusivity as selectivity evidence only for explicitly kinetic or membrane-separation questions.
    - Uptake explanation analyses:
    * Treat critical_basis_normalization_check as a required quantitative reporting block:
      report every listed framework density and volumetric Henry coefficient, then state its
      required_conclusion without changing its scientific meaning.
    * Use critical_uptake_uncertainty_check for pairwise ranking claims. A difference is
      statistically resolved at approximately 95% only when resolved_at_95_percent is true.
    * When independent_evidence.isotherm_shape.series is available, use its curve-derived regime, knee, initial slope, plateau status, and working capacity before applying a fixed pressure-regime heuristic.
    * Treat the single-site Langmuir fit as a shape descriptor, not proof that the MOF has one adsorption-site type.
    * If the fitted knee is unreliable or outside the measured range, state that saturation and the Henry-to-capacity crossover remain unconfirmed.
    * Use independent_evidence.adsorption_regime.rows[].pressure_regime to decide whether Henry affinity, mixed affinity/capacity, or pore-capacity evidence should be primary.
    * Use derived_relationships.henry_affinity to test low-pressure uptake only on matched bases: mass Henry versus gravimetric uptake, and volumetric Henry versus volumetric uptake.
    * Never claim that mass-based Henry directly controls volumetric uptake.
    * Use derived_relationships.adsorption_heat to test whether uptake follows adsorption heat/interaction strength.
    * Use derived_relationships.pore_structure to test whether uptake follows pore geometry/capacity descriptors.
    * In a confirmed or likely Henry regime, do not claim that pore volume or surface-area capacity
      dominates uptake from a single pressure point. Treat Zeo++ descriptors as supporting structural
      covariates unless pressure-dependent data demonstrate a capacity-controlled regime.
    * Qst measures mean interaction strength, whereas Henry affinity includes energetic and
      configurational contributions. An inverse Qst trend does not by itself imply that capacity dominates.
    * Use independent_evidence.energy_histogram.hostguest_summary to discuss sampled host-guest energy distributions, strong-site fractions, broadness, and multimodality.
    * Use independent_evidence.adsorption_site_density.*.fingerprint to describe preferred node/linker/site-type environments from RASPA density hotspots.
    * Treat adsorption-site contacts as density-weighted proximity, not proof of a chemical bond or a free-energy minimum.
    * Use uptake_analysis.evidence_synthesis.rows as the integrated evidence table when explaining uptake differences overall.
    * In evidence_synthesis, separate affinity/energy evidence from capacity/geometry-fit evidence instead of reducing everything to one number.
    - LAMMPS trajectory analyses:
    * Use diffusion_analysis.primary_result for diffusivity and independent_evidence.msd for the MSD curve summary.
    * Use diffusion_analysis.independent_evidence.trajectory.anisotropic_diffusion.fit_axes to discuss directional diffusion.
    * Use diffusion_analysis.independent_evidence.trajectory.van_hove_non_gaussian to discuss heterogeneous/caged/hopping-like diffusion beyond MSD.
    * Use diffusion_analysis.independent_evidence.trajectory.velocity_autocorrelation to discuss short-time dynamical memory; warn if dump stride is coarse.
    * Use diffusion_analysis.independent_evidence.trajectory.energy_autocorrelation only when status is ok; otherwise say the thermo energy column was unavailable.
    * Use diffusion_analysis.independent_evidence.trajectory.rdf_contact.contact_type_fraction and nearest_distance_A to discuss guest-framework contacts from MD.
    * Use diffusion_analysis.independent_evidence.trajectory.residence_hopping to discuss residence time and hopping frequency.
    * Use diffusion_analysis.independent_evidence.trajectory.chemistry_unit_contact when available to describe node/linker contact fractions.
    * Use diffusion_analysis.independent_evidence.trajectory.pore_network_hopping_graph to describe site-to-site transitions as a qualitative hopping network.
    * Use diffusion_analysis.independent_evidence.diffusion_meta.activation_barrier for Arrhenius apparent activation barriers across temperatures.
    * Use diffusion_analysis.independent_evidence.diffusion_meta.replicate_consistency to report repeated-diffusivity scatter and coefficient of variation.
    * Use thermal_expansion_analysis.primary_result for dV/dT and alpha_V, including fit quality.
    * Use mechanical_response_analysis.primary_result for the fitted uniaxial modulus, stress units, strain range, and fit quality.
    * Treat two-MOF correlations as qualitative direction checks only.
    - Sectioned report:
    * Return 3 to 7 report_sections tailored to the requested analysis and available evidence.
    * Use specific headings such as "Binding Energy Comparison", "Bader Charge Transfer",
      or "Limitations and Next Step" when those topics are relevant. Do not force irrelevant headings.
    * Each body must be a polished prose paragraph, not a bullet list or JSON-like field dump.
    * Preserve the meaning and uncertainty of summary/key_findings; do not introduce new evidence or stronger causal claims.
    * Use readable numeric precision while retaining units and scientifically meaningful differences.
    * If reported uncertainty intervals overlap, do not call the corresponding ordering evident,
      resolved, or definitive.
    * Always include "Overall Interpretation" after the evidence sections.
    * Always include "Limitations and Next Step" as the final section.

    Return JSON:
    {{
    "summary": "",
    "key_findings": [],
    "uncertainties": [],
    "next_best_step": "",
    "report_sections": [
      {{"heading": "", "body": ""}}
    ]
    }}
    """
        interpretation = self._invoke_llm_json(prompt, InterpretationModel)
        self._ensure_basis_conversion_in_report(
            interpretation,
            uptake_henry_evidence.get("reporting_summary", {}),
        )
        return interpretation

    @staticmethod
    def _ensure_basis_conversion_in_report(
        interpretation: InterpretationModel,
        reporting_summary: Dict[str, Any],
    ) -> None:
        rows = [
            row
            for row in (reporting_summary.get("rows") or [])
            if row.get("mof")
            and row.get("framework_density_g_cm3") is not None
            and row.get("henry_volumetric_mol_m3_Pa") is not None
        ]
        if not rows:
            return

        for section in interpretation.report_sections:
            heading = section.heading.strip().lower()
            if "henry" not in heading and "basis" not in heading:
                continue
            section_text = section.body.lower()
            has_density_units = (
                "g/cm^3" in section_text or "g/cm3" in section_text
            )
            has_density_context = "densit" in section_text
            has_all_frameworks = all(
                (
                    str(row["mof"]).lower() in section_text
                    or str(row["mof"]).split("_", 1)[0].lower() in section_text
                )
                for row in rows
            )
            if has_density_units and has_density_context and has_all_frameworks:
                return

        density_values = ", ".join(
            f"{row['mof']} {float(row['framework_density_g_cm3']):.6f}"
            for row in rows
        )
        density_sentence = (
            "The framework densities are "
            f"{density_values} g/cm^3, respectively; after conversion to kg/m^3, "
            "these values are used in K_H,vol = K_H,mass x density."
        )

        for section in interpretation.report_sections:
            heading = section.heading.strip().lower()
            if "henry" in heading or "basis" in heading:
                section.body = f"{section.body.rstrip()} {density_sentence}"
                return

        new_section = ReportSectionModel(
            heading="Framework Density and Henry-Basis Conversion",
            body=density_sentence,
        )
        insertion_index = next(
            (
                index
                for index, section in enumerate(interpretation.report_sections)
                if "overall" in section.heading.strip().lower()
            ),
            len(interpretation.report_sections),
        )
        interpretation.report_sections.insert(insertion_index, new_section)

    @staticmethod
    def calculation_requests_for_plan(
        plan: SimulationPlanModel,
    ) -> List[CalculationRequestModel]:
        requests: Dict[str, CalculationRequestModel] = {}

        for step in plan.steps:
            method = step.method
            spec = ANALYSIS_METHODS.get(method)
            if spec is None:
                raise ValueError(f"Unknown analysis method: {method}")

            engine = str(spec.get("engine") or "")
            agent = ENGINE_AGENT_MAP.get(engine)
            if agent is None:
                continue

            existing = requests.get(method)
            if existing is None:
                requests[method] = CalculationRequestModel(
                    method=method,
                    engine=engine,
                    agent=agent,
                    requested_by=[method],
                )

        return list(requests.values())

    def recommend_analysis_tasks(
        self,
        context: Dict[str, Any],
    ) -> AnalysisRecommendationModel:
        
        goal_obj = self._step_goal(context)
        goal = goal_obj.goal
        print("goal:", goal)

        
        hyp_obj = self._step_hypothesis(context, goal)
        hypothesis = hyp_obj.hypothesis
        print("hypothesis:", hypothesis)

        
        analysis_plan = self._step_plan(context, goal, hypothesis)
        requests = self.calculation_requests_for_plan(analysis_plan)
        return AnalysisRecommendationModel(
            analysis_plan=analysis_plan,
            calculation_requests=requests,
        )
