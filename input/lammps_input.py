import json
import os
import subprocess
import traceback
import io
from pathlib import Path
from typing import Any, Dict, Optional
from contextlib import redirect_stdout, redirect_stderr

from config import WORKING_DIR
from input.lammps.pipeline_lammps import generate_lammps_inputs

class LAMMPSInputAgent:
    def __init__(self, llm=None):
        from config import LLM_DEFAULT
        self.llm = llm or LLM_DEFAULT

    def _get_lammps_rag_hints(self, context: Dict[str, Any]) -> Dict[str, str]:
        import os
        disabled = (
            os.getenv("SIMMOF_DISABLE_LITERATURE_RAG", "").strip().lower() in {"1", "true", "yes", "on"}
            or os.getenv("SIMMOF_LAMMPS_FF_RAG", "1").strip().lower() in {"0", "false", "no", "off"}
        )
        if disabled:
            return {"ff_hints": "", "charge_hints": ""}

        cached = context.get("lammps_rag_hints")
        if isinstance(cached, dict):
            return {
                "ff_hints": (cached.get("ff_hints") or "").strip(),
                "charge_hints": (cached.get("charge_hints") or "").strip(),
            }

        out = {"ff_hints": "", "charge_hints": ""}
        try:
            from rag.agent import RagAgent
            rag_ctx = {
                "job_name": context.get("job_name") or "",
                "mof": context.get("mof") or "",
                "guest": context.get("guest") or "",
                "property": context.get("property") or "",
                "query_text": context.get("query_text") or context.get("QueryText") or "",
            }
            agent = RagAgent(agent_name="RagAgent")
            r = agent.run_for_lammps_ff(rag_ctx, top_files=5)
            out["ff_hints"] = (r.get("ff_hints") or "").strip()
            out["charge_hints"] = (r.get("charge_hints") or "").strip()
            print("[RAG] LAMMPS hints enabled" if (out["ff_hints"] or out["charge_hints"]) else "[RAG] no LAMMPS hints found")
        except Exception as e:
            print(f"[RAG] LAMMPS hints disabled due to error: {e}")

        context["lammps_rag_hints"] = out
        return out

    def _decide_charge_method_for_lammps(
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

        from langchain_core.messages import HumanMessage, SystemMessage

        mof        = context.get("mof", "")
        guest      = context.get("guest", "") or ""
        prop       = context.get("property", "")
        query_text = context.get("query_text", "") or context.get("QueryText", "")
        charge_hints = (context.get("lammps_rag_hints") or {}).get("charge_hints", "")

        system_msg = (
            "You decide whether and how to assign partial charges for a LAMMPS MD/MC simulation of a MOF.\n"
            "Return ONLY JSON: {\"method\": \"<choice>\", \"reason\": \"<one sentence>\"}\n"
            "\n"
            "Choices:\n"
            "  none   — no charges needed (nonpolar guest, van-der-Waals only)\n"
            "  cif    — use charges pre-assigned in the CIF/data file (DDEC, REPEAT, CHELPG, etc.)\n"
            "  eqeq   — compute charges with EQeq (fast, screening-quality fallback)\n"
            "  pacman — ML-predicted charges (approximate DDEC6, no DFT; suitable for screening)\n"
            "  ddec   — true DFT-based DDEC6 via VASP + CHARGEMOL (exact, slow; use when charge accuracy is critical)\n"
            "\n"
            "Decision guide:\n"
            "- Nonpolar guests (CH4, noble gases) or MOF-only mechanical/thermal properties: none\n"
            "- CIF already has charges (_atom_site_charge): cif (always prefer this)\n"
            "- Screening / high-throughput when approximate charges are needed: eqeq\n"
            "- Polar guests (CO2, H2O, SO2, NH3) without pre-computed charges: pacman by default\n"
            "- User explicitly requests DFT charges, CHARGEMOL, or publication-quality accuracy: ddec\n"
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
            f"CIF has pre-assigned charges: {cif_has_charges}\n"
        )
        if charge_hints:
            prompt += f"\nRAG_HINTS (charge methods used in similar studies):\n{charge_hints}\n"
        prompt += "\nWhich charge method should be used for this LAMMPS simulation?"

        try:
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
            print(f"[LAMMPSInputAgent] Charge method: {method} — {reason}")
            return method
        except Exception as e:
            print(f"[LAMMPSInputAgent] Charge method LLM failed: {e}")
            return "cif" if cif_has_charges else "none"

    def _run_generate_lammps_inputs(
        self,
        working_dir,
        mof_name,
        guest_name,
        prop,
        query_text,
        num_guest,
        job_name,
        simulation_input: Optional[Dict[str, Any]] = None,
        charge_method: str = "auto",
        context: Optional[Dict[str, Any]] = None,
    ):
        guest_name = "" if guest_name is None else str(guest_name)
        query_text = "" if query_text is None else str(query_text)

        if simulation_input is None:
            simulation_input = {"present": False, "snippets": []}

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        try:
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                generate_lammps_inputs(
                    working_dir=str(working_dir),
                    mof_name=str(mof_name),
                    guest_name=guest_name if guest_name != "" else None,
                    property_name=str(prop),
                    query_text=query_text,
                    simulation_input=simulation_input,
                    num_guest=int(num_guest),
                    job_name=str(job_name),
                    charge_method=charge_method,
                    context=context,
                )

            return subprocess.CompletedProcess(
                args=["generate_lammps_inputs"],
                returncode=0,
                stdout=stdout_buffer.getvalue(),
                stderr=stderr_buffer.getvalue(),
            )

        except Exception:
            traceback.print_exc(file=stderr_buffer)
            return subprocess.CompletedProcess(
                args=["generate_lammps_inputs"],
                returncode=1,
                stdout=stdout_buffer.getvalue(),
                stderr=stderr_buffer.getvalue(),
            )
    
    def _extract_guest_types_from_system_in(self, system_in_path):
        import re
        from pathlib import Path

        text = Path(system_in_path).read_text()
        m = re.search(r'^\s*group\s+guest\s+type\s+(.+)$', text, re.MULTILINE)
        if not m:
            return []

        return [int(x) for x in m.group(1).split()]

    def _infer_production_start_step_from_system_in(self, system_in_path):
        import re
        from pathlib import Path

        total = 0
        text = Path(system_in_path).read_text()

        for line in text.splitlines():
            s = line.strip()

            if re.match(r'^compute\s+msd_guest\b', s):
                break

            m = re.match(r'^run\s+(\d+)\b', s)
            if m:
                total += int(m.group(1))

        return total

    def _parse_masses_from_system_data(self, system_data_path):
        from pathlib import Path

        masses = {}
        lines = Path(system_data_path).read_text().splitlines()

        in_masses = False
        for line in lines:
            s = line.strip()

            if not s:
                continue

            if s.lower() == "masses":
                in_masses = True
                continue

            if in_masses:
                if s[0].isalpha():
                    break

                parts = s.split()
                if len(parts) >= 2:
                    try:
                        atype = int(parts[0])
                        mass = float(parts[1])
                        masses[atype] = mass
                    except ValueError:
                        pass

        return masses
    
    def _infer_dt_fs_from_system_in(self, system_in_path):
        import re
        from pathlib import Path

        text = Path(system_in_path).read_text()
        current_timestep = None

        for line in text.splitlines():
            s = line.strip()

            m = re.match(r'^timestep\s+([0-9Ee+.\-]+)\b', s)
            if m:
                current_timestep = float(m.group(1))

            if re.match(r'^compute\s+msd_guest\b', s):
                break

        if current_timestep is None:
            return 1.0

        return current_timestep

    def _inject_diffusivity_context(self, context):
        from pathlib import Path

        prop = str(context.get("property", "")).lower()
        if prop not in ["diffusivity", "diffusion", "self_diffusivity", "self_diffusion_coefficient"]:
            return context

        work_dir = Path(context["work_dir"])
        system_in_path = work_dir / "system.in"
        system_data_path = work_dir / "system.data"

        if not system_in_path.exists():
            raise RuntimeError(f"system.in not found: {system_in_path}")

        if not system_data_path.exists():
            raise RuntimeError(f"system.data not found: {system_data_path}")

        guest_types = self._extract_guest_types_from_system_in(system_in_path)
        production_start_step = self._infer_production_start_step_from_system_in(system_in_path)
        masses_by_type = self._parse_masses_from_system_data(system_data_path)
        dt_fs = self._infer_dt_fs_from_system_in(system_in_path)

        context["guest_types"] = guest_types
        context["production_start_step"] = production_start_step
        context["masses_by_type"] = masses_by_type
        context["dt_fs"] = dt_fs
        context.setdefault("fit_start_ps", 200.0)
        context.setdefault("fit_end_ps", None)

        return context

    def run(self, context):
        if context.get("lammps_status") == "needs_structure_from_user":
            context.setdefault("results", {})[
                "lammps_input_status"
            ] = "blocked_missing_structure"
            return context

        paths = context.get("paths") if isinstance(context.get("paths"), dict) else {}

        work_dir = context.get("work_dir")
        if work_dir is None:
            work_dir = paths.get("work_dir")

        if work_dir is None:
            plan_root = context.get("plan_root")
            if plan_root is None:
                plan_root = paths.get("plan_root")
            if plan_root:
                work_dir = str(Path(plan_root))
            else:
                work_dir = str(Path(WORKING_DIR) / context["job_name"])

        os.makedirs(work_dir, exist_ok=True)
        context["work_dir"] = work_dir

        print(f"LAMMPS input will be stored in: {work_dir}")

        sim_in = context.get("simulation_input")
        if sim_in is None:
            sim_in = {"present": False, "snippets": []}

        self._get_lammps_rag_hints(context)
        from input.lammps.pipeline_lammps import cif_has_atom_site_charge
        mof_cif = Path(work_dir) / f"{context['mof']}.cif"
        cif_has_q = cif_has_atom_site_charge(str(mof_cif)) if mof_cif.exists() else False
        charge_method = self._decide_charge_method_for_lammps(context, cif_has_charges=cif_has_q)
        context["charge_method"] = charge_method
        if charge_method in ("eqeq", "pacman", "ddec"):
            context["charge_method_required"] = charge_method

        result = self._run_generate_lammps_inputs(
            working_dir=work_dir,
            mof_name=context["mof"],
            guest_name=context.get("guest"),
            prop=context["property"],
            query_text=context.get("query_text", ""),
            num_guest=context.get("num_guest", 4 if "diffus" in str(context.get("property", "")).lower() else 1),
            job_name=context.get("job_name", ""),
            simulation_input=sim_in,
            charge_method=charge_method,
            context=context,
        )

        print("[LAMMPSInputAgent] generate_lammps_inputs")
        print("returncode =", result.returncode)
        if result.stdout:
            print("STDOUT:\n", result.stdout)
        if result.stderr:
            print("STDERR:\n", result.stderr)

        results = context.setdefault("results", {})
        results["lammps_input_status"] = "ok" if result.returncode == 0 else "failed"
        results["lammps_input_returncode"] = result.returncode
        results["lammps_input_stdout"] = result.stdout
        results["lammps_input_stderr"] = result.stderr

        if result.returncode != 0:
            raise RuntimeError("LAMMPS input generation failed; skipping submission.")

        context = self._inject_diffusivity_context(context)

        return context
