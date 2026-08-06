import os
import json
import re
import shutil
import time
import ase.io
import ase.io.vasp

from pathlib import Path
from typing import Dict, Any, Optional, List

import config
from config import working_dir, LLM_DEFAULT
from core.pipeline import make_pipeline_chain
from core.resource_allocator import ResourceAllocator, ResourceSpec
from core.timing import timed_call

from structure.agent import VASPStructureAgent
from input.vasp_input import VASPInputAgent
from file.agent import VASPFileAgent
from VASP.runner import VASPRunner
from error.vasp_error import VASPErrorAgent
from error.vasp_structure_precheck import VASPStructurePrecheckAgent
from error.structure_regeneration import clear_structure_regeneration_request
from output.vasp_output import VASPOutputAgent
from VASP.bader_reuse import (
    is_valid_chgcar,
    make_charge_dir_from_source,
    run_bader,
    parse_acf,
)
from VASP.adsorption import build_frozen_fragments, compact_magmom


class VASPAgent:

    def __init__(self, llm=None, debug_dump: bool = True):
        self.llm = llm or LLM_DEFAULT
        self.debug_dump = debug_dump

        self.structure = VASPStructureAgent()
        self.input = VASPInputAgent(llm=self.llm)
        self.runner = VASPRunner()
        self.error = VASPErrorAgent(llm=self.llm)
        self.structure_precheck = VASPStructurePrecheckAgent()
        self.output = VASPOutputAgent()

        
        
        
        self.mof_chain = make_pipeline_chain(
            steps=[
                ("mof_structure", self._timed_step("mof_structure", self.structure.run_mof_only)),
                ("mof_input", self._timed_step("mof_input", self.input.run)),
                ("mof_structure_precheck", self._timed_step("mof_structure_precheck", self.structure_precheck.run)),
                ("mof_pre_run_review", self._timed_step("mof_pre_run_review", self.error.pre_run_review)),
                ("mof_submit", self._timed_step("mof_submit", self.runner.run)),
                ("mof_error", self._timed_step("mof_error", self.error.run)),
                ("mof_output", self._timed_step("mof_output", self.output.run)),
            ],
            dump_step=(lambda ctx, n, k: self._dump_step_lcel(ctx, prefix="mof", step_agent=n, step_order=k))
            if self.debug_dump
            else None,
        )

        self.guest_chain = make_pipeline_chain(
            steps=[
                ("guest_prepare_structure", self._timed_step("guest_prepare_structure", self._guest_prepare_structure)),
                ("guest_input", self._timed_step("guest_input", self.input.run)),
                ("guest_structure_precheck", self._timed_step("guest_structure_precheck", self.structure_precheck.run)),
                ("guest_pre_run_review", self._timed_step("guest_pre_run_review", self.error.pre_run_review)),
                ("guest_submit", self._timed_step("guest_submit", self.runner.run)),
                ("guest_error", self._timed_step("guest_error", self.error.run)),
                ("guest_output", self._timed_step("guest_output", self.output.run)),
            ],
            dump_step=(lambda ctx, n, k: self._dump_step_lcel(ctx, prefix="guest", step_agent=n, step_order=k))
            if self.debug_dump
            else None,
        )

        self.complex_chain = make_pipeline_chain(
            steps=[
                ("complex_prepare_optimized_mof", self._timed_step("complex_prepare_optimized_mof", self._complex_prepare_optimized_mof)),
                ("complex_structure", self._timed_step("complex_structure", self.structure.run_guest_and_complex_from_optimized)),
                ("complex_prescreen", self._timed_step("complex_prescreen", self._prescreen_complex_candidates_with_mlip)),
                ("complex_input", self._timed_step("complex_input", self.input.run)),
                ("complex_structure_precheck", self._timed_step("complex_structure_precheck", self.structure_precheck.run)),
                ("complex_pre_run_review", self._timed_step("complex_pre_run_review", self.error.pre_run_review)),
                ("complex_submit", self._timed_step("complex_submit", self.runner.run)),
                ("complex_error", self._timed_step("complex_error", self.error.run)),
                ("complex_output", self._timed_step("complex_output", self.output.run)),
            ],
            dump_step=(lambda ctx, n, k: self._dump_step_lcel(ctx, prefix="complex", step_agent=n, step_order=k))
            if self.debug_dump
            else None,
        )

    
    
    
    def _timed_step(self, step_name: str, fn):
        def wrapper(ctx: Dict[str, Any]) -> Dict[str, Any]:
            return timed_call(
                step_name,
                fn,
                ctx,
                category="vasp_step",
                context=ctx,
                extra={"parent_agent": "VASPAgent"},
            )
        return wrapper

    def _ensure_defaults(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        ctx.setdefault("results", {})
        if "job_name" not in ctx and "plan_name" in ctx:
            ctx["job_name"] = ctx["plan_name"]
        ctx.setdefault("query_text", "")

        if not ctx.get("work_dir"):
            base = Path(working_dir) / (ctx.get("job_name") or "vasp_job")
            base.mkdir(parents=True, exist_ok=True)
            ctx["work_dir"] = str(base)

        return ctx

    def _dump_step(self, ctx: Dict[str, Any], tag: str):
        if not self.debug_dump:
            return
        d = Path(ctx.get("work_dir", working_dir)) / "_debug"
        d.mkdir(parents=True, exist_ok=True)
        out = d / f"ctx_{tag}.json"
        try:
            with open(out, "w", encoding="utf-8") as f:
                json.dump(ctx, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"[VASPAgent] dump failed {tag}: {e}")

    def _dump_step_lcel(self, ctx: Dict[str, Any], prefix: str, step_agent: str, step_order: int):
        if not self.debug_dump:
            return
        d = Path(ctx.get("work_dir", working_dir)) / "_debug"
        d.mkdir(parents=True, exist_ok=True)
        job_id = ctx.get("job_id", "unknown_job")
        out = d / f"ctx_{prefix}_{step_order:02d}_{step_agent}_{job_id}.json"
        try:
            with open(out, "w", encoding="utf-8") as f:
                json.dump(ctx, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"[VASPAgent] dump failed {prefix}/{step_agent}: {e}")

    def _get_ctx_vasp_system(self, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        sys_info = ctx.get("vasp_system")
        if not (isinstance(sys_info, dict) and sys_info.get("dir")):
            vasp_dir = ctx.get("vasp_dir")
            if not vasp_dir:
                return None
            sys_info = {"dir": vasp_dir}
            if ctx.get("vasp_label"):
                sys_info["label"] = ctx.get("vasp_label")
            if ctx.get("vasp_role"):
                sys_info["role"] = ctx.get("vasp_role")

        if ctx.get("vasp_label") and not sys_info.get("label"):
            sys_info["label"] = ctx.get("vasp_label")
        if ctx.get("vasp_role") and not sys_info.get("role"):
            sys_info["role"] = ctx.get("vasp_role")

        ctx["vasp_system"] = sys_info
        ctx["vasp_dir"] = sys_info["dir"]
        if sys_info.get("label"):
            ctx["vasp_label"] = sys_info["label"]
        if sys_info.get("role"):
            ctx["vasp_role"] = sys_info["role"]

        paths = ctx.get("paths")
        if isinstance(paths, dict):
            paths.setdefault("vasp", {})
            paths["vasp"]["run_dir"] = sys_info["dir"]

        return sys_info

    def _get_ctx_vasp_dir(self, ctx: Dict[str, Any]) -> Optional[str]:
        sys_info = ctx.get("vasp_system")
        if isinstance(sys_info, dict):
            vasp_dir = sys_info.get("dir")
            if vasp_dir:
                return vasp_dir
        return ctx.get("vasp_dir")

    
    def _prescreen_complex_candidates_with_mlip(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        candidates = ctx.get("complex_cif_paths") or []
        if not candidates:
            raise RuntimeError("[VASPAgent] no complex candidates found for MLIP prescreen")

        mlip_dir = Path(ctx["work_dir"]) / "mlip_prescreen"

        prescreen = self._run_mlip_complex_prescreen(
            complex_cif_paths=candidates,
            mlip_dir=mlip_dir,
            device=ctx.get("mlip_device", "cpu"),
            top_n=1,
        )

        best = prescreen["best_result"]

        ctx["complex_candidates"] = prescreen["all_results"]
        ctx["complex_selection_method"] = "mlip_lowest_energy_after_packmol"
        ctx["mlip_selected_idx"] = best["index"]
        ctx["mlip_selected_energy_ev"] = best["energy_ev"]

        ctx["complex_cif_path"] = best["relaxed_cif"]
        ctx["complex_path"] = ctx["complex_cif_path"]

        complex_label = Path(ctx["complex_cif_path"]).stem
        ctx["vasp_label"] = complex_label
        ctx.setdefault("vasp_system", {})
        ctx["vasp_system"]["label"] = complex_label

        return ctx
    
    def _fetch_initial_mof_cif(self, mof: str, target_dir: str, ctx: Dict[str, Any] = None) -> str:
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        for key in ("mof_path", "cif_path"):
            src = (ctx or {}).get(key)
            if src and os.path.isfile(src):
                dst = os.path.join(target_dir, f"{mof}.cif")
                shutil.copy2(src, dst)
                return dst
        self.structure.get_mof(mof, target_dir)
        return os.path.join(target_dir, f"{mof}.cif")

    def _optimized_mof_contcar_path(self, plan_root: str) -> str:
        return os.path.join(plan_root, "vasp", "mof", "CONTCAR")

    def _find_binding_plan_name(self, ctx: Dict[str, Any]) -> str:
        ups = ctx.get("upstream_plans") or {}
        if not ups:
            raise RuntimeError(
                "[VASPAgent] this calculation requires upstream binding_energy results"
            )

        my_mof = ctx.get("mof")
        my_guest = ctx.get("guest")
        exact_candidates = []
        binding_candidates = []
        for pname, pres in ups.items():
            if not isinstance(pres, dict):
                continue
            for _, jctx in pres.items():
                if not isinstance(jctx, dict):
                    continue
                if jctx.get("property") != "binding_energy":
                    continue
                binding_candidates.append(pname)
                if (
                    jctx.get("mof") == my_mof
                    and jctx.get("guest") == my_guest
                ):
                    exact_candidates.append(pname)
                break
        exact_candidates = list(dict.fromkeys(exact_candidates))
        binding_candidates = list(dict.fromkeys(binding_candidates))
        if len(exact_candidates) == 1:
            return exact_candidates[0]
        if len(binding_candidates) == 1:
            return binding_candidates[0]

        if len(ups) == 1:
            return next(iter(ups.keys()))

        candidates = []
        for pname, pres in ups.items():
            if not isinstance(pres, dict):
                continue
            for _, jctx in pres.items():
                if isinstance(jctx, dict) and jctx.get("mof") == my_mof and jctx.get("guest") == my_guest:
                    candidates.append(pname)
                    break
        if len(candidates) == 1:
            return candidates[0]

        raise RuntimeError(
            "[VASPAgent] cannot uniquely identify binding_energy plan from upstream_plans: %s"
            % list(ups.keys())
        )

    def _get_source_vasp_dir_from_upstream_plans(self, ctx: Dict[str, Any], role: str) -> str:
        binding_plan = self._find_binding_plan_name(ctx)
        pres = ctx["upstream_plans"][binding_plan]
        if not isinstance(pres, dict):
            raise RuntimeError(f"[VASPAgent] upstream_plans[{binding_plan}] is not a dict")

        for job_id, jctx in pres.items():
            if isinstance(jctx, dict) and jctx.get("vasp_role") == role:
                upstream_vasp_dir = self._get_ctx_vasp_dir(jctx)
                if upstream_vasp_dir:
                    return upstream_vasp_dir

        for job_id, jctx in pres.items():
            if isinstance(jctx, dict) and job_id.endswith(f"_{role}"):
                upstream_vasp_dir = self._get_ctx_vasp_dir(jctx)
                if upstream_vasp_dir:
                    return upstream_vasp_dir

        raise RuntimeError(
            f"[VASPAgent] cannot find binding_energy upstream vasp_dir for role={role} in plan={binding_plan}"
        )

    def _run_bader_from_source(self, ctx: Dict[str, Any], role: str) -> Dict[str, Any]:
        ctx.setdefault("results", {})
        results = ctx["results"]

        source_vasp_dir = Path(self._get_source_vasp_dir_from_upstream_plans(ctx, role=role))
        ctx["bader_source_vasp_dir"] = str(source_vasp_dir)

        source_chgcar = source_vasp_dir / "CHGCAR"
        check0 = is_valid_chgcar(str(source_chgcar))

        if check0.get("ok"):
            b0 = run_bader(source_vasp_dir, require_reference=True)
            if b0.get("status") == "ok":
                acf_path = Path(b0["ACF"])
                results["bader_charge"] = {
                    "status": "ok",
                    "phase": "reuse_source",
                    "role": role,
                    "source_vasp_dir": str(source_vasp_dir),
                    "bader_dir": str(source_vasp_dir),
                    "CHGCAR_check": check0,
                    "ACF": str(acf_path),
                    "idx_to_value": parse_acf(acf_path),
                    "reference_mode": b0.get("reference_mode"),
                    "reference_density": b0.get("reference_density"),
                    "bader_command": b0.get("command"),
                }
                return ctx

            results["bader_charge"] = {
                "status": "error",
                "phase": "reuse_source_bader_failed",
                "role": role,
                "source_vasp_dir": str(source_vasp_dir),
                "bader_dir": str(source_vasp_dir),
                "CHGCAR_check": check0,
                "bader_run": b0,
            }

        charge_dir = Path(str(source_vasp_dir) + "_charge")

        submit_label = ctx.get("vasp_label") or ctx.get("job_id") or f"bader_{role}"
        prep = make_charge_dir_from_source(
            source_vasp_dir=source_vasp_dir,
            charge_dir=charge_dir,
            submit_label=submit_label,
        )
        if not prep.get("ok"):
            results["bader_charge"] = {
                "status": "error",
                "phase": "charge_dir_prep_failed",
                "role": role,
                "source_vasp_dir": str(source_vasp_dir),
                "charge_dir": str(charge_dir),
                "CHGCAR_check_source": check0,
                "prep": prep,
            }
            return ctx

        old_vasp_dir = ctx.get("vasp_dir")
        old_vasp_system = ctx.get("vasp_system")
        old_vasp_submit = ctx.get("vasp_submit")
        old_vasp_job_id = ctx.get("vasp_job_id")
        old_vasp_submitted = ctx.get("vasp_submitted")

        try:
            ctx["vasp_dir"] = str(charge_dir)
            if isinstance(old_vasp_system, dict):
                sys2 = dict(old_vasp_system)
                sys2["dir"] = str(charge_dir)
                ctx["vasp_system"] = sys2

            ctx = timed_call("bader_submit", self.runner.run, ctx, category="vasp_step", context=ctx, extra={"parent_agent": "VASPAgent"})
            ctx = timed_call("bader_error", self.error.run, ctx, category="vasp_step", context=ctx, extra={"parent_agent": "VASPAgent"})

            check1 = is_valid_chgcar(str(charge_dir / "CHGCAR"))
            if not check1.get("ok"):
                results["bader_charge"] = {
                    "status": "error",
                    "phase": "charge_run_no_chgcar",
                    "role": role,
                    "source_vasp_dir": str(source_vasp_dir),
                    "bader_dir": str(charge_dir),
                    "prep": prep,
                    "CHGCAR_check_source": check0,
                    "CHGCAR_check_charge": check1,
                    "vasp_submit_charge": ctx.get("vasp_submit"),
                }
                return ctx

            b1 = run_bader(charge_dir, require_reference=True)
            if b1.get("status") != "ok":
                results["bader_charge"] = {
                    "status": "error",
                    "phase": "charge_bader_failed",
                    "role": role,
                    "source_vasp_dir": str(source_vasp_dir),
                    "bader_dir": str(charge_dir),
                    "prep": prep,
                    "CHGCAR_check_charge": check1,
                    "bader_run": b1,
                }
                return ctx

            acf_path = Path(b1["ACF"])
            results["bader_charge"] = {
                "status": "ok",
                "phase": "charge_dir",
                "role": role,
                "source_vasp_dir": str(source_vasp_dir),
                "bader_dir": str(charge_dir),
                "prep": prep,
                "CHGCAR_check_source": check0,
                "CHGCAR_check_charge": check1,
                "ACF": str(acf_path),
                "idx_to_value": parse_acf(acf_path),
                "reference_mode": b1.get("reference_mode"),
                "reference_density": b1.get("reference_density"),
                "bader_command": b1.get("command"),
            }
            return ctx

        finally:
            ctx["vasp_dir"] = old_vasp_dir
            ctx["vasp_system"] = old_vasp_system
            ctx["vasp_submit"] = old_vasp_submit
            ctx["vasp_job_id"] = old_vasp_job_id
            ctx["vasp_submitted"] = old_vasp_submitted

    def _make_optimized_mof_cif_from_upstream_dir(
        self,
        ctx: Dict[str, Any],
        mof_vasp_dir: str,
        mof_ctx: Optional[Dict[str, Any]] = None,
    ) -> str:
        contcar = os.path.join(mof_vasp_dir, "CONTCAR")
        if not os.path.exists(contcar):
            if mof_ctx is not None:
                raise FileNotFoundError(
                    f"[VASPAgent] optimized MOF CONTCAR not found: {contcar}\n"
                    f"  upstream mof_vasp_dir={mof_vasp_dir}\n"
                    f"  upstream mof_state={mof_ctx.get('vasp_state')}\n"
                    f"  upstream mof_outcar={mof_ctx.get('results', {}).get('vasp_outcar')}"
                )
            raise FileNotFoundError(f"[VASPAgent] optimized MOF CONTCAR not found: {contcar}")

        opt_cif = os.path.join(ctx["work_dir"], f"{ctx['mof']}_opt.cif")
        try:
            atoms = ase.io.read(contcar)
            ase.io.write(opt_cif, atoms, format="cif")
        except Exception as e:
            raise RuntimeError(f"[VASPAgent] failed to convert CONTCAR -> CIF: {contcar} ({e})")

        ctx["optimized_mof_path"] = opt_cif
        return opt_cif

    
    
    
    def _guest_prepare_structure(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        ctx["vasp_stage"] = "guest"

        work_dir = ctx["work_dir"]
        mof = ctx["mof"]
        guest = ctx["guest"]

        mof_cif = self._fetch_initial_mof_cif(mof, work_dir, ctx=ctx)
        ctx["mof_cell_source"] = mof_cif

        guest_xyz, guest_cif = self.structure.get_guest(guest, work_dir, mof_path=mof_cif)
        ctx["guest_path"] = guest_xyz
        ctx["guest_cif_path"] = guest_cif

        return ctx

    def _complex_prepare_optimized_mof(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        ctx["vasp_stage"] = "complex"
        ctx.setdefault("results", {})

        plan_root = ctx.get("plan_root")
        if not plan_root:
            plan_root = str(Path(working_dir) / (ctx.get("plan_name") or ctx.get("job_name") or "vasp_plan"))
            ctx["plan_root"] = plan_root

        if not ctx.get("work_dir"):
            ctx["work_dir"] = plan_root

        upstream_jobs = ctx.get("upstream_jobs") or {}
        if len(upstream_jobs) != 1:
            raise RuntimeError(
                f"[VASPAgent] complex job expects exactly 1 upstream mof job, got {list(upstream_jobs.keys())}"
            )

        mof_ctx = next(iter(upstream_jobs.values()))
        mof_vasp_dir = self._get_ctx_vasp_dir(mof_ctx)
        if not mof_vasp_dir:
            raise RuntimeError("[VASPAgent] upstream mof ctx missing vasp_dir")

        opt_cif = self._make_optimized_mof_cif_from_upstream_dir(ctx, mof_vasp_dir, mof_ctx=mof_ctx)
        ctx["mof_path"] = opt_cif  

        return ctx

    
    
    
    def _run_dos_subrun_from_contcar(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        ctx.setdefault("results", {})

        mof_vasp_dir = self._get_ctx_vasp_dir(ctx)
        if not mof_vasp_dir:
            raise RuntimeError("[VASPAgent] DOS requires ctx['vasp_dir'] from mof optimization")

        opt_cif = self._make_optimized_mof_cif_from_upstream_dir(ctx, mof_vasp_dir)

        ctx2 = dict(ctx)
        ctx2.pop("vasp_system", None)
        ctx2.pop("vasp_dir", None)
        ctx2.pop("vasp_label", None)

        ctx2["vasp_state"] = "pending"
        ctx2["vasp_retry"] = 0
        ctx2.pop("vasp_submit", None)
        ctx2.pop("vasp_job_id", None)
        ctx2["vasp_submitted"] = False

        ctx2["vasp_stage"] = "dos"
        ctx2["vasp_calc_type"] = "dos"
        ctx2["optimized_mof_path"] = opt_cif
        ctx2["dos_has_chgcar"] = os.path.exists(os.path.join(mof_vasp_dir, "CHGCAR"))

        ctx2.pop("incar_overrides", None)

        ctx2 = timed_call("dos_input", self.input.run, ctx2, category="vasp_step", context=ctx2, extra={"parent_agent": "VASPAgent"})
        self._dump_step(ctx2, "dos_input")

        dos_dir = self._get_ctx_vasp_dir(ctx2)
        if dos_dir:
            for fn in ["CHGCAR", "WAVECAR"]:
                src = os.path.join(mof_vasp_dir, fn)
                dst = os.path.join(dos_dir, fn)
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)

        ctx2 = timed_call("dos_submit", self.runner.run, ctx2, category="vasp_step", context=ctx2, extra={"parent_agent": "VASPAgent"})
        self._dump_step(ctx2, "dos_submit")

        ctx2 = timed_call("dos_error", self.error.run, ctx2, category="vasp_step", context=ctx2, extra={"parent_agent": "VASPAgent"})
        self._dump_step(ctx2, "dos_error")

        ctx2 = timed_call("dos_output", self.output.run, ctx2, category="vasp_step", context=ctx2, extra={"parent_agent": "VASPAgent"})
        self._dump_step(ctx2, "dos_output")

        ctx.setdefault("results", {})
        ctx["results"]["dos"] = ctx2.get("results", {}).get("dos", {})
        ctx["dos_vasp_dir"] = self._get_ctx_vasp_dir(ctx2)

        return ctx

    def _run_projected_dos_from_source(
        self,
        ctx: Dict[str, Any],
        role: str,
    ) -> Dict[str, Any]:
        source_vasp_dir = Path(
            self._get_source_vasp_dir_from_upstream_plans(ctx, role=role)
        )
        source_structure = source_vasp_dir / "CONTCAR"
        if not source_structure.exists() or source_structure.stat().st_size == 0:
            source_structure = source_vasp_dir / "POSCAR"
        if not source_structure.exists():
            raise RuntimeError(
                f"[VASPAgent] projected_dos source structure is missing for {role}: "
                f"{source_vasp_dir}"
            )

        ctx["vasp_stage"] = "projected_dos"
        ctx["vasp_calc_type"] = "projected_dos"
        ctx["projected_dos_role"] = role
        ctx["projected_dos_source_vasp_dir"] = str(source_vasp_dir)
        ctx["projected_dos_structure_path"] = str(source_structure)
        source_fft_grid = self._read_vasp_fft_grid(source_vasp_dir / "OUTCAR")
        ctx["projected_dos_fft_grid"] = source_fft_grid
        ctx["projected_dos_has_chgcar"] = (
            (source_vasp_dir / "CHGCAR").exists()
            and source_fft_grid is not None
        )
        ctx["projected_dos_has_wavecar"] = (source_vasp_dir / "WAVECAR").exists()

        ctx = timed_call(
            "projected_dos_input",
            self.input.run,
            ctx,
            category="vasp_step",
            context=ctx,
            extra={"parent_agent": "VASPAgent"},
        )
        self._dump_step(ctx, f"projected_dos_{role}_input")

        pdos_dir = self._get_ctx_vasp_dir(ctx)
        if not pdos_dir:
            raise RuntimeError("[VASPAgent] projected_dos input did not set vasp_dir")
        pdos_dir_path = Path(pdos_dir)
        for filename in ("CHGCAR", "WAVECAR"):
            source = source_vasp_dir / filename
            target = pdos_dir_path / filename
            if source.exists() and not target.exists():
                shutil.copy2(source, target)

        ctx = timed_call(
            "projected_dos_submit",
            self.runner.run,
            ctx,
            category="vasp_step",
            context=ctx,
            extra={"parent_agent": "VASPAgent"},
        )
        self._dump_step(ctx, f"projected_dos_{role}_submit")
        ctx = timed_call(
            "projected_dos_error",
            self.error.run,
            ctx,
            category="vasp_step",
            context=ctx,
            extra={"parent_agent": "VASPAgent"},
        )
        ctx = timed_call(
            "projected_dos_output",
            self.output.run,
            ctx,
            category="vasp_step",
            context=ctx,
            extra={"parent_agent": "VASPAgent"},
        )
        self._dump_step(ctx, f"projected_dos_{role}_output")

        pdos_result = ctx.setdefault("results", {}).setdefault(
            "projected_dos",
            {},
        )
        pdos_result.update(
            {
                "role": role,
                "source_vasp_dir": str(source_vasp_dir),
                "vasp_dir": str(pdos_dir_path),
            }
        )
        return ctx

    @staticmethod
    def _read_vasp_fft_grid(outcar_path: Path) -> Optional[Dict[str, int]]:
        if not outcar_path.exists():
            return None
        try:
            text = outcar_path.read_text(errors="ignore")
        except OSError:
            return None

        coarse = re.findall(
            r"dimension x,y,z NGX\s*=\s*(\d+)\s+NGY\s*=\s*(\d+)\s+NGZ\s*=\s*(\d+)",
            text,
        )
        fine = re.findall(
            r"dimension x,y,z NGXF\s*=\s*(\d+)\s+NGYF\s*=\s*(\d+)\s+NGZF\s*=\s*(\d+)",
            text,
        )
        if not coarse or not fine:
            return None
        ngx, ngy, ngz = coarse[0]
        ngxf, ngyf, ngzf = fine[0]
        return {
            "NGX": int(ngx),
            "NGY": int(ngy),
            "NGZ": int(ngz),
            "NGXF": int(ngxf),
            "NGYF": int(ngyf),
            "NGZF": int(ngzf),
        }

    def _run_bandgap_subrun_from_contcar(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        ctx.setdefault("results", {})

        mof_vasp_dir = self._get_ctx_vasp_dir(ctx)
        if not mof_vasp_dir:
            raise RuntimeError("[VASPAgent] bandgap requires ctx['vasp_dir'] from mof optimization")

        opt_cif = self._make_optimized_mof_cif_from_upstream_dir(ctx, mof_vasp_dir)

        ctx2 = dict(ctx)
        ctx2.pop("vasp_system", None)
        ctx2.pop("vasp_dir", None)
        ctx2.pop("vasp_label", None)

        ctx2["vasp_state"] = "pending"
        ctx2["vasp_retry"] = 0
        ctx2.pop("vasp_submit", None)
        ctx2.pop("vasp_job_id", None)
        ctx2["vasp_submitted"] = False

        ctx2["vasp_stage"] = "bandgap"
        ctx2["vasp_calc_type"] = "bandgap"
        ctx2["optimized_mof_path"] = opt_cif

        ctx2 = timed_call("bandgap_input", self.input.run, ctx2, category="vasp_step", context=ctx2, extra={"parent_agent": "VASPAgent"})
        self._dump_step(ctx2, "bandgap_input")

        bandgap_dir = self._get_ctx_vasp_dir(ctx2)
        if bandgap_dir:
            for fn in ["CHGCAR", "WAVECAR"]:
                src = os.path.join(mof_vasp_dir, fn)
                dst = os.path.join(bandgap_dir, fn)
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)

        ctx2 = timed_call("bandgap_submit", self.runner.run, ctx2, category="vasp_step", context=ctx2, extra={"parent_agent": "VASPAgent"})
        self._dump_step(ctx2, "bandgap_submit")

        ctx2 = timed_call("bandgap_error", self.error.run, ctx2, category="vasp_step", context=ctx2, extra={"parent_agent": "VASPAgent"})
        self._dump_step(ctx2, "bandgap_error")

        ctx2 = timed_call("bandgap_output", self.output.run, ctx2, category="vasp_step", context=ctx2, extra={"parent_agent": "VASPAgent"})
        self._dump_step(ctx2, "bandgap_output")

        ctx.setdefault("results", {})
        ctx["results"]["bandgap"] = ctx2.get("results", {}).get("bandgap", ctx2.get("results", {}))
        ctx["bandgap_vasp_dir"] = self._get_ctx_vasp_dir(ctx2)

        return ctx

    
    
    
    def _run_mof_job(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        ctx["vasp_stage"] = "mof_opt"

        if ctx.get("property") == "projected_dos":
            ctx["vasp_role"] = "mof"
            return self._run_projected_dos_from_source(ctx, role="mof")

        if ctx.get("property") == "bader_charge":
            ctx["vasp_role"] = "mof"
            return self._run_bader_from_source(ctx, role="mof")

        ctx = self.mof_chain.invoke(ctx)

        if ctx.get("vasp_status") == "needs_structure_from_user":
            return ctx

        if ctx.get("property") in ["dos", "electronic_density_of_states", "density_of_states"]:
            ctx = self._run_dos_subrun_from_contcar(ctx)

        if ctx.get("property") in ["bandgap", "electronic_bandgap"]:
            ctx = self._run_bandgap_subrun_from_contcar(ctx)

        if ctx.get("property") in ["geometry_optimization", "opt", "relax", "optimized_structure"]:
            vasp_dir = self._get_ctx_vasp_dir(ctx)
            if vasp_dir and os.path.exists(os.path.join(vasp_dir, "CONTCAR")):
                opt_cif = self._make_optimized_mof_cif_from_upstream_dir(ctx, vasp_dir)
                ctx["results"].setdefault("optimized_structure", {})
                ctx["results"]["optimized_structure"].update(
                    {
                        "status": "ok",
                        "vasp_dir": vasp_dir,
                        "CONTCAR": os.path.join(vasp_dir, "CONTCAR"),
                        "optimized_cif": opt_cif,
                    }
                )

        return ctx

    def _run_guest_job(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        if ctx.get("property") == "projected_dos":
            ctx["vasp_role"] = "guest"
            return self._run_projected_dos_from_source(ctx, role="guest")

        return self.guest_chain.invoke(ctx)

    def _vasp_finished_ok(self, ctx: Dict[str, Any]) -> bool:
        results = ctx.get("results") or {}
        return (
            ctx.get("vasp_state") == "done_ok"
            or results.get("vasp_status") == "ok"
            or results.get("vasp_output_status") == "ok"
        )

    def _read_failure_text(self, ctx: Dict[str, Any], max_chars: int = 50000) -> str:
        texts: List[str] = []
        results = ctx.get("results") or {}
        precheck = results.get("vasp_structure_precheck")
        if precheck:
            texts.append(json.dumps(precheck, ensure_ascii=False, default=str))

        vasp_dir = None
        sys_info = ctx.get("vasp_system")
        if isinstance(sys_info, dict):
            vasp_dir = sys_info.get("dir")
        vasp_dir = vasp_dir or ctx.get("vasp_dir")
        if not vasp_dir:
            return "\n".join(texts)

        for name in ("out.txt", "OUTCAR", "vasp.log", "stderr.txt", "qsub.err"):
            path = Path(vasp_dir) / name
            if not path.is_file():
                continue
            try:
                raw = path.read_text(errors="ignore")
            except Exception:
                continue
            texts.append(raw[-max_chars:])
        return "\n".join(texts)

    def _is_structure_related_complex_failure(self, ctx: Dict[str, Any]) -> bool:
        if self._vasp_finished_ok(ctx):
            return False

        if ctx.get("vasp_needs_structure_regeneration") is True:
            return True

        results = ctx.get("results") or {}
        precheck = results.get("vasp_structure_precheck") or {}
        if ctx.get("vasp_structure_precheck_status") == "failed":
            return True
        if isinstance(precheck, dict) and precheck.get("status") == "failed":
            return True
        submit = ctx.get("vasp_submit") or {}
        if submit.get("status") == "failed_structure_precheck":
            return True

        return False

    def _archive_existing_complex_vasp_dir(self, ctx: Dict[str, Any], retry_dir: Path) -> Optional[str]:
        old_dir = Path(ctx["work_dir"]) / "vasp" / "complex"
        if not old_dir.exists():
            return None
        archive_root = retry_dir / "archived_vasp"
        archive_root.mkdir(parents=True, exist_ok=True)
        archive_dir = archive_root / f"complex_{time.strftime('%Y%m%d_%H%M%S')}"
        if archive_dir.exists():
            archive_dir = archive_root / f"{archive_dir.name}_{int(time.time() * 1000)}"
        shutil.move(str(old_dir), str(archive_dir))
        return str(archive_dir)

    def _reset_complex_vasp_runtime_state(self, ctx: Dict[str, Any]) -> None:
        for key in (
            "vasp_system",
            "vasp_dir",
            "vasp_submit",
            "vasp_job_id",
            "scheduler_job_id",
            "vasp_structure_precheck_status",
            "vasp_energy",
            "vasp_needs_structure_regeneration",
            "vasp_structure_regeneration_reason",
            "vasp_recovery_route",
            "vasp_recovery_policy",
        ):
            ctx.pop(key, None)

        ctx["vasp_state"] = "pending"
        ctx["vasp_retry"] = 0
        ctx["vasp_submitted"] = False
        ctx["vasp_stage"] = "complex"
        ctx["vasp_role"] = "complex"

        results = ctx.setdefault("results", {})
        for key in (
            "vasp_status",
            "vasp_input_status",
            "vasp_structure_precheck",
            "vasp_output_status",
            "vasp_run_status",
            "vasp_energy_ev",
            "vasp_label",
            "vasp_role",
            "vasp_outcar",
            "vasp_recovery_policy",
        ):
            results.pop(key, None)

    def _regenerate_complex_structure(self, ctx: Dict[str, Any], attempt: int) -> Dict[str, Any]:
        mof_path = ctx.get("mof_path") or ctx.get("optimized_mof_path")
        guest_xyz_path = ctx.get("guest_path")
        if not mof_path or not os.path.exists(mof_path):
            raise FileNotFoundError(f"[VASPAgent] cannot regenerate complex: mof_path missing ({mof_path})")

        retry_dir = Path(ctx["work_dir"]) / "structure_regeneration" / f"attempt_{attempt:02d}"
        retry_dir.mkdir(parents=True, exist_ok=True)

        if not guest_xyz_path or not os.path.exists(guest_xyz_path):
            guest_name = ctx.get("guest")
            if not guest_name:
                raise ValueError("[VASPAgent] cannot regenerate complex: guest is missing")
            guest_xyz_path, guest_cif_path = self.structure.get_guest(
                guest_name,
                str(retry_dir),
                mof_path=mof_path,
                src_path=ctx.get("guest_src_path"),
            )
            ctx["guest_path"] = guest_xyz_path
            ctx["guest_cif_path"] = guest_cif_path

        previous_vasp_dir = self._archive_existing_complex_vasp_dir(ctx, retry_dir)
        self._reset_complex_vasp_runtime_state(ctx)

        complex_cif_paths = self.structure.get_complex(
            mof_path=mof_path,
            guest_xyz_path=guest_xyz_path,
            save_dir=str(retry_dir),
        )
        if not complex_cif_paths:
            raise RuntimeError("[VASPAgent] complex regeneration produced no CIF candidates")

        prescreen = self._run_mlip_complex_prescreen(
            complex_cif_paths=complex_cif_paths,
            mlip_dir=retry_dir / "mlip_prescreen",
            device=ctx.get("mlip_device", "cpu"),
            top_n=1,
        )
        best = prescreen["best_result"]

        ctx["complex_cif_paths"] = complex_cif_paths
        ctx["complex_candidates"] = prescreen["all_results"]
        ctx["complex_selection_method"] = f"mlip_lowest_energy_after_packmol_regeneration_{attempt}"
        ctx["mlip_selected_idx"] = best["index"]
        ctx["mlip_selected_energy_ev"] = best["energy_ev"]
        ctx["complex_cif_path"] = best["relaxed_cif"]
        ctx["complex_path"] = ctx["complex_cif_path"]
        ctx["vasp_label"] = Path(ctx["complex_cif_path"]).stem

        ctx.setdefault("results", {}).setdefault("vasp_structure_regeneration", []).append(
            {
                "attempt": attempt,
                "retry_dir": str(retry_dir),
                "previous_vasp_dir": previous_vasp_dir,
                "num_candidates": len(complex_cif_paths),
                "selected_cif": ctx["complex_cif_path"],
                "selected_energy_ev": best.get("energy_ev"),
            }
        )
        clear_structure_regeneration_request(ctx, status="handled")
        return ctx

    def _run_complex_vasp_attempt(self, ctx: Dict[str, Any], attempt: int) -> Dict[str, Any]:
        prefix = f"complex_regen_{attempt:02d}"
        for step_name, fn in (
            ("input", self.input.run),
            ("structure_precheck", self.structure_precheck.run),
            ("pre_run_review", self.error.pre_run_review),
            ("submit", self.runner.run),
            ("error", self.error.run),
            ("output", self.output.run),
        ):
            full_name = f"{prefix}_{step_name}"
            ctx = timed_call(
                full_name,
                fn,
                ctx,
                category="vasp_step",
                context=ctx,
                extra={"parent_agent": "VASPAgent"},
            )
            self._dump_step(ctx, full_name)
        return ctx

    @staticmethod
    def _automation_mode_enabled(ctx: Dict[str, Any]) -> bool:
        mode = str(ctx.get("interaction_mode") or config.INTERACTION_MODE).strip().lower()
        return mode in {"autonomous", "automation", "automatic"}

    @staticmethod
    def _expand_magmom(value: str) -> Optional[List[float]]:
        expanded: List[float] = []
        for token in value.replace(",", " ").split():
            token = token.strip()
            if not token:
                continue
            try:
                if "*" in token:
                    count_text, number_text = token.split("*", 1)
                    expanded.extend([float(number_text)] * int(count_text))
                else:
                    expanded.append(float(token))
            except (TypeError, ValueError):
                return None
        return expanded

    @staticmethod
    def _remove_incar_keys(incar_text: str, keys: List[str]) -> str:
        patterns = [
            re.compile(rf"^\s*{re.escape(key)}\s*=", re.IGNORECASE)
            for key in keys
        ]
        return "\n".join(
            line
            for line in incar_text.splitlines()
            if not any(pattern.search(line) for pattern in patterns)
        )

    @staticmethod
    def _subset_incar_species_vectors(
        incar_text: str,
        source_species: List[str],
        target_species: List[str],
    ) -> str:
        for key in ("LDAUL", "LDAUU", "LDAUJ", "RWIGS", "ROPT"):
            pattern = re.compile(
                rf"^(\s*{key}\s*=\s*)([^!#]*)(.*)$",
                flags=re.IGNORECASE | re.MULTILINE,
            )
            match = pattern.search(incar_text)
            if not match:
                continue
            values = match.group(2).split()
            if len(values) != len(source_species):
                continue
            by_species = dict(zip(source_species, values))
            if any(species not in by_species for species in target_species):
                continue
            replacement = (
                match.group(1)
                + " ".join(by_species[species] for species in target_species)
                + match.group(3)
            )
            incar_text = incar_text[: match.start()] + replacement + incar_text[match.end() :]
        return incar_text

    def _frozen_resource_spec(
        self,
        parent_ctx: Dict[str, Any],
        role: str,
        atom_count: int,
    ) -> ResourceSpec:
        allocation = parent_ctx.get("resource_allocation") or {}
        try:
            nodes = int(allocation["nodes"])
            ppn = int(allocation["ppn"])
            queue = str(allocation["queue"])
            if nodes > 0 and ppn > 0 and queue:
                return ResourceSpec(
                    nodes=nodes,
                    ppn=ppn,
                    np=nodes * ppn,
                    queue=queue,
                    rationale=(
                        f"Reuse the converged complex allocation for the {role} "
                        "frozen single-point calculation."
                    ),
                )
        except (KeyError, TypeError, ValueError):
            pass

        complex_dir_value = self._get_ctx_vasp_dir(parent_ctx)
        if complex_dir_value:
            for qsub_path in sorted(Path(complex_dir_value).glob("*.qsub")):
                try:
                    qsub_text = qsub_path.read_text(errors="replace")
                except OSError:
                    continue
                resource_match = re.search(
                    r"^#PBS\s+-l\s+nodes=(\d+):ppn=(\d+)(?::\S+)?\s*$",
                    qsub_text,
                    flags=re.MULTILINE,
                )
                queue_match = re.search(
                    r"^#PBS\s+-q\s+(\S+)\s*$",
                    qsub_text,
                    flags=re.MULTILINE,
                )
                if resource_match and queue_match:
                    nodes = int(resource_match.group(1))
                    ppn = int(resource_match.group(2))
                    return ResourceSpec(
                        nodes=nodes,
                        ppn=ppn,
                        np=nodes * ppn,
                        queue=queue_match.group(1),
                        rationale=(
                            f"Reuse scheduler resources from {qsub_path.name} for "
                            f"the {role} frozen single point."
                        ),
                    )
        return ResourceAllocator(llm=self.llm).recommend(
            "VASP",
            "frozen_interaction_single_point",
            atom_count,
            parent_ctx,
        )

    def _prepare_frozen_component_inputs(
        self,
        parent_ctx: Dict[str, Any],
        atoms,
        role: str,
        source_indices0: List[int],
    ) -> Dict[str, Any]:
        complex_dir_value = self._get_ctx_vasp_dir(parent_ctx)
        if not complex_dir_value:
            raise RuntimeError("complex VASP directory is unavailable")
        complex_dir = Path(complex_dir_value)
        source_poscar = complex_dir / "POSCAR"
        source_incar = complex_dir / "INCAR"
        if not source_poscar.is_file() or not source_incar.is_file():
            raise FileNotFoundError("complex POSCAR/INCAR is required for frozen single points")

        source_atoms = ase.io.read(str(source_poscar))
        source_symbols = source_atoms.get_chemical_symbols()
        source_species = sorted(set(source_symbols))

        fragment_symbols = atoms.get_chemical_symbols()
        order = sorted(range(len(atoms)), key=lambda index: (fragment_symbols[index], index))
        sorted_atoms = atoms[order]
        sorted_source_indices0 = [int(source_indices0[index]) for index in order]

        run_dir = complex_dir / "frozen_interaction" / role
        run_dir.mkdir(parents=True, exist_ok=True)
        out_dir = str(run_dir) + os.sep
        ase.io.vasp.write_vasp(
            str(run_dir / "POSCAR"),
            sorted_atoms,
            direct=True,
            sort=False,
            vasp5=True,
        )
        VASPFileAgent.atoms_to_potcar(sorted_atoms, out_dir, str(config.VASP_POTENTIAL_DIR_PATH) + os.sep)

        raw_label = f"{parent_ctx.get('vasp_label') or parent_ctx.get('job_id')}_{role}_sp"
        label = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_label).strip("_") or f"{role}_sp"
        spec = self._frozen_resource_spec(parent_ctx, role, len(sorted_atoms))
        VASPFileAgent.make_qsub(out_dir, label, spec=spec)

        source_kpoints = complex_dir / "KPOINTS"
        if source_kpoints.is_file():
            shutil.copy2(source_kpoints, run_dir / "KPOINTS")
        source_kernel = complex_dir / "vdw_kernel.bindat"
        if source_kernel.is_file():
            shutil.copy2(source_kernel, run_dir / "vdw_kernel.bindat")

        incar_text = source_incar.read_text(errors="replace")
        source_magmom = None
        magmom_match = re.search(
            r"^\s*MAGMOM\s*=\s*([^!#]*)",
            incar_text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if magmom_match:
            expanded = self._expand_magmom(magmom_match.group(1))
            if expanded is not None and len(expanded) == len(source_atoms):
                source_magmom = [expanded[index] for index in sorted_source_indices0]

        incar_text = self._remove_incar_keys(
            incar_text,
            ["MAGMOM", "M_CONSTR", "NELECT", "NUPDOWN"],
        )
        target_species = sorted(set(sorted_atoms.get_chemical_symbols()))
        incar_text = self._subset_incar_species_vectors(
            incar_text,
            source_species,
            target_species,
        )
        incar_text = self.input._upsert_incar_settings(
            incar_text,
            {
                "SYSTEM": label,
                "IBRION": "-1",
                "NSW": "0",
                "ISTART": "0",
                "ICHARG": "2",
                "LWAVE": ".FALSE.",
                "LCHARG": ".FALSE.",
            },
        )
        if source_magmom is not None:
            incar_text = self.input._upsert_incar_settings(
                incar_text,
                {"MAGMOM": compact_magmom(source_magmom)},
            )
        (run_dir / "INCAR").write_text(incar_text.rstrip() + "\n")

        resource_allocation = {
            "software": "VASP",
            "calc_type": "frozen_interaction_single_point",
            "n_atoms": len(sorted_atoms),
            "nodes": spec.nodes,
            "ppn": spec.ppn,
            "np": spec.np,
            "queue": spec.queue,
            "rationale": spec.rationale,
        }
        return {
            "dir": str(run_dir),
            "label": label,
            "role": role,
            "atom_count": len(sorted_atoms),
            "source_complex_indices_1based": [index + 1 for index in sorted_source_indices0],
            "resource_allocation": resource_allocation,
        }

    def _execute_frozen_component(
        self,
        parent_ctx: Dict[str, Any],
        atoms,
        role: str,
        source_indices0: List[int],
    ) -> Dict[str, Any]:
        prepared = self._prepare_frozen_component_inputs(
            parent_ctx,
            atoms,
            role,
            source_indices0,
        )
        subctx = {
            "job_name": parent_ctx.get("job_name"),
            "job_id": f"{parent_ctx.get('job_id')}_{role}_sp",
            "agent": "VASPAgent",
            "mof": parent_ctx.get("mof"),
            "guest": parent_ctx.get("guest"),
            "property": "interaction_energy",
            "query_text": parent_ctx.get("query_text", ""),
            "interaction_mode": parent_ctx.get("interaction_mode"),
            "vasp_stage": "frozen_single_point",
            "vasp_calc_type": "frozen_interaction_single_point",
            "vasp_role": role,
            "vasp_dir": prepared["dir"],
            "vasp_label": prepared["label"],
            "vasp_system": {
                "dir": prepared["dir"],
                "label": prepared["label"],
                "role": role,
            },
            "resource_allocation": prepared["resource_allocation"],
            "results": {},
        }
        for step_name, fn in (
            (f"{role}_structure_precheck", self.structure_precheck.run),
            (f"{role}_submit", self.runner.run),
            (f"{role}_error", self.error.run),
            (f"{role}_output", self.output.run),
        ):
            subctx = timed_call(
                step_name,
                fn,
                subctx,
                category="vasp_step",
                context=subctx,
                extra={"parent_agent": "VASPAgent", "purpose": "frozen_interaction"},
            )
        return {
            **prepared,
            "status": subctx.get("results", {}).get("vasp_output_status"),
            "energy_ev": subctx.get("results", {}).get("vasp_energy_ev"),
            "outcar": subctx.get("results", {}).get("vasp_outcar"),
        }

    def _maybe_run_frozen_interaction_calculations(
        self,
        ctx: Dict[str, Any],
    ) -> Dict[str, Any]:
        results = ctx.setdefault("results", {})
        if (ctx.get("property") or "").lower() != "binding_energy":
            return ctx
        if (ctx.get("vasp_role") or "").lower() != "complex":
            return ctx
        deformation = results.get("structure_deformation") or {}
        if not deformation.get("threshold_exceeded"):
            return ctx
        if (
            results.get("vasp_output_status") != "ok"
            or results.get("vasp_energy_ev") is None
        ):
            return ctx
        previous = results.get("interaction_energy") or {}
        if previous.get("status") == "ok":
            return ctx

        threshold = deformation.get("threshold_percent", 20.0)
        if not self._automation_mode_enabled(ctx):
            results["interaction_energy"] = {
                "status": "recommended_not_run_interactive",
                "reason": (
                    f"structure deformation reached "
                    f"{deformation.get('overall_deformation_percent', 0.0):.2f}% "
                    f"(threshold {float(threshold):.2f}%)"
                ),
                "recommended_calculation": (
                    "frozen MOF and frozen guest single-point energies at the "
                    "optimized complex geometry"
                ),
            }
            return ctx

        try:
            upstream_jobs = ctx.get("upstream_jobs") or {}
            mof_ctx = next(
                (
                    job_ctx
                    for job_id, job_ctx in upstream_jobs.items()
                    if isinstance(job_ctx, dict)
                    and (
                        job_ctx.get("vasp_role") == "mof"
                        or str(job_id).endswith("_mof")
                    )
                ),
                None,
            )
            if not isinstance(mof_ctx, dict):
                raise RuntimeError("optimized MOF upstream context is unavailable")
            mof_dir_value = self._get_ctx_vasp_dir(mof_ctx)
            complex_dir_value = self._get_ctx_vasp_dir(ctx)
            if not mof_dir_value or not complex_dir_value:
                raise RuntimeError("MOF or complex VASP directory is unavailable")
            mof_dir = Path(mof_dir_value)
            complex_dir = Path(complex_dir_value)
            optimized_mof = mof_dir / "CONTCAR"
            complex_initial = complex_dir / "POSCAR"
            complex_final = complex_dir / "CONTCAR"

            frozen_mof, frozen_guest, mapping = build_frozen_fragments(
                str(optimized_mof),
                str(complex_initial),
                str(complex_final),
            )
            results["frozen_component_mapping"] = mapping
            framework_indices0 = [
                int(index) - 1
                for index in mapping["framework_complex_indices_1based"]
            ]
            guest_indices0 = [
                int(index) - 1 for index in mapping["guest_complex_indices_1based"]
            ]
            mof_sp = self._execute_frozen_component(
                ctx,
                frozen_mof,
                "frozen_mof",
                framework_indices0,
            )
            guest_sp = self._execute_frozen_component(
                ctx,
                frozen_guest,
                "frozen_guest",
                guest_indices0,
            )
            complex_energy = results.get("vasp_energy_ev")
            if (
                complex_energy is None
                or mof_sp.get("energy_ev") is None
                or guest_sp.get("energy_ev") is None
            ):
                raise RuntimeError("one or more frozen single-point energies are missing")
            interaction_energy = (
                float(complex_energy)
                - float(mof_sp["energy_ev"])
                - float(guest_sp["energy_ev"])
            )
            results["interaction_energy"] = {
                "status": "ok",
                "trigger": "structure_deformation_threshold_exceeded",
                "deformation_percent": deformation.get("overall_deformation_percent"),
                "deformation_threshold_percent": threshold,
                "E_int_ev": interaction_energy,
                "E_complex_opt_ev": float(complex_energy),
                "E_mof_frozen_ev": float(mof_sp["energy_ev"]),
                "E_guest_frozen_ev": float(guest_sp["energy_ev"]),
                "equation": (
                    "E_int = E(MOF+guest,opt) - E(MOF,frozen) - E(guest,frozen)"
                ),
                "interpretation": "direct host-guest interaction at the optimized complex geometry",
                "frozen_mof": mof_sp,
                "frozen_guest": guest_sp,
            }
        except Exception as exc:
            results["interaction_energy"] = {
                "status": "failed",
                "trigger": "structure_deformation_threshold_exceeded",
                "deformation_percent": deformation.get("overall_deformation_percent"),
                "deformation_threshold_percent": threshold,
                "reason": f"{type(exc).__name__}: {exc}",
            }
            print(f"[VASPAgent] frozen interaction-energy calculation failed: {exc}")
        return ctx

    def _run_complex_job(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        if ctx.get("property") == "projected_dos":
            ctx["vasp_role"] = "complex"
            return self._run_projected_dos_from_source(ctx, role="complex")

        if ctx.get("property") == "bader_charge":
            ctx["vasp_role"] = "complex"
            return self._run_bader_from_source(ctx, role="complex")

        ctx = self.complex_chain.invoke(ctx)
        max_regenerations = int(ctx.get("vasp_structure_regen_max_attempts", 2) or 0)

        for attempt in range(1, max_regenerations + 1):
            if not self._is_structure_related_complex_failure(ctx):
                break
            print(
                f"[VASPAgent] structure-related complex failure detected; "
                f"regenerating complex geometry (attempt {attempt}/{max_regenerations})"
            )
            ctx = self._regenerate_complex_structure(ctx, attempt)
            ctx = self._run_complex_vasp_attempt(ctx, attempt)
            if self._vasp_finished_ok(ctx):
                break

        return self._maybe_run_frozen_interaction_calculations(ctx)

    
    
    
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        ctx = self._ensure_defaults(context)
        self._dump_step(ctx, "start")

        job_id = ctx.get("job_id", "")
        vasp_role = ctx.get("vasp_role", "")

        if vasp_role == "mof" or job_id.endswith("_mof"):
            return self._run_mof_job(ctx)
        if vasp_role == "guest" or job_id.endswith("_guest"):
            return self._run_guest_job(ctx)
        if vasp_role == "complex" or job_id.endswith("_complex"):
            return self._run_complex_job(ctx)

        prop = (ctx.get("property") or "").lower()
        if prop in {
            "density_of_states",
            "electronic_density_of_states",
            "dos",
            "geometry_optimization",
            "optimized_structure",
            "opt",
            "relax",
            "electronic_bandgap",
            "bandgap",
            "band_gap",
            "neb",
            "migration_barrier",
            "charge_density_difference",
            "vibrational_frequencies",
        }:
            ctx.setdefault("vasp_role", "mof")
            return self._run_mof_job(ctx)

        raise ValueError(f"[VASPAgent] Unknown job_id pattern: {job_id}")

    def resume(self, context: Dict[str, Any]) -> Dict[str, Any]:
        ctx = self._ensure_defaults(dict(context))
        ctx["resume_mode"] = True
        self._dump_step(ctx, "resume_start")

        vasp_dir = self._get_ctx_vasp_dir(ctx)
        has_marker = False
        if vasp_dir:
            run_dir = Path(vasp_dir)
            has_marker = any((run_dir / name).exists() for name in ("START", "DONE", "FAILED"))
        has_scheduler = bool(ctx.get("scheduler_job_id") or ctx.get("vasp_job_id"))
        was_submitted = bool(ctx.get("vasp_submitted"))

        if not (has_marker or has_scheduler or was_submitted or self._vasp_finished_ok(ctx)):
            raise RuntimeError(
                "VASPAgent.resume requires a saved context from or after VASP submission "
                "(scheduler job id, vasp_submitted flag, or START/DONE/FAILED marker required)."
            )

        if not self._vasp_finished_ok(ctx):
            ctx = timed_call(
                "vasp_resume_error",
                self.error.run,
                ctx,
                category="vasp_step",
                context=ctx,
                extra={"parent_agent": "VASPAgent"},
            )
            self._dump_step(ctx, "resume_error")

        if self._is_structure_related_complex_failure(ctx):
            max_regenerations = int(ctx.get("vasp_structure_regen_max_attempts", 2) or 0)
            for attempt in range(1, max_regenerations + 1):
                if not self._is_structure_related_complex_failure(ctx):
                    break
                ctx = self._regenerate_complex_structure(ctx, attempt)
                ctx = self._run_complex_vasp_attempt(ctx, attempt)
                if self._vasp_finished_ok(ctx):
                    break

        ctx = timed_call(
            "vasp_resume_output",
            self.output.run,
            ctx,
            category="vasp_step",
            context=ctx,
            extra={"parent_agent": "VASPAgent"},
        )
        self._dump_step(ctx, "resume_output")
        if (ctx.get("vasp_role") or "").lower() == "complex":
            ctx = self._maybe_run_frozen_interaction_calculations(ctx)
        return ctx

    def _run_mlip_complex_prescreen(
        self,
        complex_cif_paths,
        mlip_dir,
        device="cpu",
        top_n=1,
    ):
        from tool.utils import run_mlip_complex_candidates

        mlip_dir = Path(mlip_dir)
        mlip_dir.mkdir(parents=True, exist_ok=True)

        print(f"[MLIP] direct run start")
        print(f"[MLIP] num candidates = {len(complex_cif_paths)}")
        print(f"[MLIP] mlip_dir = {mlip_dir}")
        print(f"[MLIP] device = {device}")

        result = run_mlip_complex_candidates(
            complex_cif_paths=[str(p) for p in complex_cif_paths],
            okdir=str(mlip_dir),
            device=str(device),
            top_n=int(top_n),
        )

        result_json = mlip_dir / "_mlip_result.json"
        with open(result_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)

        print(f"[MLIP] direct run finished")
        print(f"[MLIP] result saved to {result_json}")

        return result
