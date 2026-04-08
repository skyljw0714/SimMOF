import os
import json
import shutil
import ase.io

from pathlib import Path
from typing import Dict, Any, Optional

from config import working_dir, LLM_DEFAULT
from core.pipeline import make_pipeline_chain

from structure.agent import VASPStructureAgent
from input.vasp_input import VASPInputAgent
from VASP.runner import VASPRunner
from error.vasp_error import VASPErrorAgent
from output.vasp_output import VASPOutputAgent
from VASP.bader_reuse import (
    is_valid_chgcar,
    make_charge_dir_from_source,
    run_bader,
    parse_acf,
)


class VASPAgent:

    def __init__(self, llm=None, debug_dump: bool = True):
        self.llm = llm or LLM_DEFAULT
        self.debug_dump = debug_dump

        self.structure = VASPStructureAgent()
        self.input = VASPInputAgent(llm=self.llm)
        self.runner = VASPRunner()
        self.error = VASPErrorAgent(llm=self.llm)
        self.output = VASPOutputAgent()

        
        
        
        self.mof_chain = make_pipeline_chain(
            steps=[
                ("mof_structure", self.structure.run_mof_only),
                ("mof_input", self.input.run),
                ("mof_submit", self.runner.run),
                ("mof_error", self.error.run),
                ("mof_output", self.output.run),
            ],
            dump_step=(lambda ctx, n, k: self._dump_step_lcel(ctx, prefix="mof", step_agent=n, step_order=k))
            if self.debug_dump
            else None,
        )

        self.guest_chain = make_pipeline_chain(
            steps=[
                ("guest_prepare_structure", self._guest_prepare_structure),
                ("guest_input", self.input.run),
                ("guest_submit", self.runner.run),
                ("guest_error", self.error.run),
                ("guest_output", self.output.run),
            ],
            dump_step=(lambda ctx, n, k: self._dump_step_lcel(ctx, prefix="guest", step_agent=n, step_order=k))
            if self.debug_dump
            else None,
        )

        self.complex_chain = make_pipeline_chain(
            steps=[
                ("complex_prepare_optimized_mof", self._complex_prepare_optimized_mof),
                ("complex_structure", self.structure.run_guest_and_complex_from_optimized),
                ("complex_prescreen", self._prescreen_complex_candidates_with_mlip),
                ("complex_input", self.input.run),
                ("complex_submit", self.runner.run),
                ("complex_error", self.error.run),
                ("complex_output", self.output.run),
            ],
            dump_step=(lambda ctx, n, k: self._dump_step_lcel(ctx, prefix="complex", step_agent=n, step_order=k))
            if self.debug_dump
            else None,
        )

    
    
    
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
            vasp_dir = self._get_ctx_vasp_dir(ctx)
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
        # Backward-compatible alias; vasp_system["dir"] is canonical.
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
        sys_info = self._get_ctx_vasp_system(ctx)
        if isinstance(sys_info, dict):
            return sys_info.get("dir")
        return None

    
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
    
    def _fetch_initial_mof_cif(self, mof: str, target_dir: str) -> str:
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        self.structure.get_mof(mof, target_dir)
        return os.path.join(target_dir, f"{mof}.cif")

    def _optimized_mof_contcar_path(self, plan_root: str) -> str:
        return os.path.join(plan_root, "vasp", "mof", "CONTCAR")

    def _find_binding_plan_name(self, ctx: Dict[str, Any]) -> str:
        ups = ctx.get("upstream_plans") or {}
        if not ups:
            raise RuntimeError("[VASPAgent] bader_charge requires upstream_plans (binding_energy results)")

        for pname, pres in ups.items():
            if not isinstance(pres, dict):
                continue
            for _, jctx in pres.items():
                if isinstance(jctx, dict) and jctx.get("property") == "binding_energy":
                    return pname

        if len(ups) == 1:
            return next(iter(ups.keys()))

        my_mof = ctx.get("mof")
        my_guest = ctx.get("guest")
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
            b0 = run_bader(source_vasp_dir)
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

            ctx = self.runner.run(ctx)
            ctx = self.error.run(ctx)

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

            b1 = run_bader(charge_dir)
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

        mof_cif = self._fetch_initial_mof_cif(mof, work_dir)
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

        ctx2 = self.input.run(ctx2)
        self._dump_step(ctx2, "dos_input")

        dos_dir = self._get_ctx_vasp_dir(ctx2)
        if dos_dir:
            for fn in ["CHGCAR", "WAVECAR"]:
                src = os.path.join(mof_vasp_dir, fn)
                dst = os.path.join(dos_dir, fn)
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)

        ctx2 = self.runner.run(ctx2)
        self._dump_step(ctx2, "dos_submit")

        ctx2 = self.error.run(ctx2)
        self._dump_step(ctx2, "dos_error")

        ctx2 = self.output.run(ctx2)
        self._dump_step(ctx2, "dos_output")

        ctx.setdefault("results", {})
        ctx["results"]["dos"] = ctx2.get("results", {}).get("dos", {})
        ctx["dos_vasp_dir"] = self._get_ctx_vasp_dir(ctx2)

        return ctx

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

        ctx2 = self.input.run(ctx2)
        self._dump_step(ctx2, "bandgap_input")

        bandgap_dir = self._get_ctx_vasp_dir(ctx2)
        if bandgap_dir:
            for fn in ["CHGCAR", "WAVECAR"]:
                src = os.path.join(mof_vasp_dir, fn)
                dst = os.path.join(bandgap_dir, fn)
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)

        ctx2 = self.runner.run(ctx2)
        self._dump_step(ctx2, "bandgap_submit")

        ctx2 = self.error.run(ctx2)
        self._dump_step(ctx2, "bandgap_error")

        ctx2 = self.output.run(ctx2)
        self._dump_step(ctx2, "bandgap_output")

        ctx.setdefault("results", {})
        ctx["results"]["bandgap"] = ctx2.get("results", {}).get("bandgap", ctx2.get("results", {}))
        ctx["bandgap_vasp_dir"] = self._get_ctx_vasp_dir(ctx2)

        return ctx

    
    
    
    def _run_mof_job(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        ctx["vasp_stage"] = "mof_opt"

        if ctx.get("property") == "bader_charge":
            ctx["vasp_role"] = "mof"
            return self._run_bader_from_source(ctx, role="mof")

        ctx = self.mof_chain.invoke(ctx)

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
        return self.guest_chain.invoke(ctx)

    def _run_complex_job(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        if ctx.get("property") == "bader_charge":
            ctx["vasp_role"] = "complex"
            return self._run_bader_from_source(ctx, role="complex")

        return self.complex_chain.invoke(ctx)

    
    
    
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        ctx = self._ensure_defaults(context)
        self._dump_step(ctx, "start")

        job_id = ctx.get("job_id", "")

        if job_id.endswith("_mof"):
            return self._run_mof_job(ctx)
        if job_id.endswith("_guest"):
            return self._run_guest_job(ctx)
        if job_id.endswith("_complex"):
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
            "neb",
            "migration_barrier",
            "charge_density_difference",
            "vibrational_frequencies",
        }:
            ctx.setdefault("vasp_role", "mof")
            return self._run_mof_job(ctx)

        raise ValueError(f"[VASPAgent] Unknown job_id pattern: {job_id}")

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