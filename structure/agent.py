import os
import ase.io
import glob
import shutil
import copy


from config import (
    get_csd_api_python_command,
    get_pormake_python_command,
    working_dir,
    LAMMPS_EXECUTABLE,
    LAMMPS_INTERFACE_EXECUTABLE,
)
import subprocess 
from packmol.run_packmol import run_packmol_from_cif
from pathlib import Path
from tool.utils import run_mofchecker

PACKMOL_DEFAULT_TOLERANCE = 2.0

class MOFNotFoundError(Exception):
    pass

def validate_mof(mof_path, save_dir):
    good_cifs = run_mofchecker(str(save_dir), okdir=None)
    good_set = {str(Path(p).resolve()) for p in good_cifs}

    if str(Path(mof_path).resolve()) not in good_set:
        raise ValueError(
            f"MOFChecker validation failed: {mof_path}\n"
            f"  → Check the [MOFChecker] FAILED output above for details and fix the CIF before retrying."
        )
    
    print(f"[MOFCHECKER] Validation passed: {Path(mof_path).name}")

class StructureAgent:
    def __init__(self):
        pass

    def _clean_cif_with_ase(self, mof_path):
        mof_path = str(mof_path)

        try:
            atoms = ase.io.read(mof_path)
            ase.io.write(mof_path, atoms, format="cif")
            self._patch_legacy_symmetry_tags(mof_path)
            print(f"[CIF] Cleaned MOF CIF written to: {mof_path}")
        except Exception as e:
            print(f"[CIF] Warning: failed to clean MOF CIF ({mof_path}): {e}")

    @staticmethod
    def _patch_legacy_symmetry_tags(mof_path):
        import re
        text = Path(mof_path).read_text(errors="replace")
        if "_symmetry_space_group_name_H-M" in text:
            return

        text = re.sub(
            r"(_space_group_IT_number\s+\d+)",
            r"\1\n_symmetry_space_group_name_H-M   'P 1'\n_symmetry_Int_Tables_number        1",
            text,
        )
        text = re.sub(
            r"(loop_\s*\n\s*_space_group_symop_operation_xyz\s*\n\s*'x, y, z'\s*\n)",
            r"\1loop_\n  _symmetry_equiv_pos_as_xyz\n  'x, y, z'\n",
            text,
        )
        Path(mof_path).write_text(text)

    def _after_fetch(self, mof_path, save_dir):
        self._clean_cif_with_ase(mof_path)

    def _resolve_job_work_dir(self, context, job_name, batch_root=None):
        seeded = context.get("work_dir")
        if seeded:
            return str(Path(seeded))
        if batch_root is not None:
            return os.path.join(str(batch_root), job_name)
        return os.path.join(working_dir, job_name)

    def _set_context_work_dir(self, context, save_dir):
        save_dir = str(Path(save_dir))
        os.makedirs(save_dir, exist_ok=True)
        context["work_dir"] = save_dir

        return save_dir

    
    def _prepare_optimize_job(self, cif_path, out_dir=None, queue="long", node="aa"):
        import re

        cif_path = Path(cif_path)
        stem = cif_path.stem
        work_dir = cif_path.parent / f"_opt_{stem}"
        work_dir.mkdir(parents=True, exist_ok=True)
        out_dir = Path(out_dir) if out_dir else cif_path.parent

        lammps_interface = shutil.which("lammps-interface") or LAMMPS_INTERFACE_EXECUTABLE

        r = subprocess.run(
            [lammps_interface, "--cutoff=6", str(cif_path.absolute())],
            capture_output=True, text=True, cwd=work_dir,
        )
        if r.returncode != 0:
            print(f"[optimize] lammps-interface failed for {stem}:\n{r.stderr}")
            return None

        in_file   = work_dir / f"in.{stem}"
        data_file = work_dir / f"data.{stem}"
        opt_data  = work_dir / f"{stem}.lammps-data"

        if not in_file.exists() or not data_file.exists():
            print(f"[optimize] lammps-interface did not produce expected files for {stem}")
            return None

        with open(in_file) as f:
            lines = f.readlines()

        patched = []
        for line in lines:
            if "read_data" in line:
                line = re.sub(r"read_data\s+\S+", f"read_data {data_file.absolute()}", line)
            if re.match(r"\s*log\s+", line):
                line = re.sub(r"log\s+\S+", f"log {work_dir / f'log.{stem}'}", line)
            patched.append(line)

        minimize_block = f"""
variable a loop 10
label loop

minimize 1.0e-5 1.0e-5 1000 10000
fix 1 all box/relax iso 0.0 vmax 0.001
minimize 1.0e-5 1.0e-5 1000 10000
unfix 1
thermo 1000

next a
jump SELF loop
run 0

compute 1 all pe
variable A equal c_1

thermo_style custom temp c_1
thermo 1
run 0

print "{stem}:$A" append {work_dir / "energy.txt"}

write_data           {opt_data}
"""
        with open(in_file, "w") as f:
            f.writelines(patched)
            f.write(minimize_block)

        qsub_file = work_dir / f"opt_{stem}.qsub"
        lammps_log = work_dir / f"out.{stem}.lammps"
        with open(qsub_file, "w") as f:
            f.write("#!/bin/sh\n")
            f.write("#PBS -r n\n")
            f.write(f"#PBS -q {queue}\n")
            f.write(f"#PBS -l nodes=1:ppn=1:{node}\n")
            f.write("#PBS -o /dev/null\n")
            f.write("#PBS -e /dev/null\n")
            f.write("export OMP_NUM_THREADS=1\n")
            f.write(f"cd {work_dir.absolute()}\n\n")
            f.write(f"{LAMMPS_EXECUTABLE} -in {in_file.absolute()} 1>{lammps_log} 2>&1\n")

        r = subprocess.run(["qas", str(qsub_file)], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[optimize] qas failed for {stem}:\n{r.stderr}")
            return None

        job_id = r.stdout.strip()
        print(f"[optimize] submitted {stem} → job {job_id}")
        return (cif_path, opt_data, out_dir, stem)

    def _poll_optimize_jobs(self, submissions, timeout=7200, poll_interval=10):
        import sys as _sys
        import time

        converter = Path(__file__).parent / "optimize" / "converter.py"
        results = {}
        pending = list(submissions)
        elapsed = 0

        while pending and elapsed < timeout:
            still_pending = []
            for (cif, opt_data, out_dir, stem) in pending:
                if opt_data.exists():
                    opt_cif = Path(out_dir) / f"{stem}_opt.cif"
                    r = subprocess.run(
                        [_sys.executable, str(converter), "-i", str(opt_data), "-o", str(opt_cif)],
                        capture_output=True, text=True,
                    )
                    if r.returncode == 0 and opt_cif.exists():
                        print(f"[optimize] {stem} → {opt_cif}")
                        results[cif] = opt_cif
                    else:
                        print(f"[optimize] converter failed for {stem}:\n{r.stderr}")
                        results[cif] = cif
                else:
                    still_pending.append((cif, opt_data, out_dir, stem))
            pending = still_pending
            if pending:
                time.sleep(poll_interval)
                elapsed += poll_interval

        for (cif, opt_data, out_dir, stem) in pending:
            print(f"[optimize] timeout: {stem}")
            results[cif] = cif

        return results

    def optimize_mof(self, cif_path, out_dir=None, queue="long", node="aa", poll_interval=10, timeout=7200):
        sub = self._prepare_optimize_job(cif_path, out_dir=out_dir, queue=queue, node=node)
        if sub is None:
            return None
        results = self._poll_optimize_jobs([sub], timeout=timeout, poll_interval=poll_interval)
        opt = results.get(Path(cif_path))
        return opt if opt != Path(cif_path) else None

    def make_random_hmof(
        self,
        n_mofs: int,
        save_dir,
        *,
        max_atoms: int = 1500,
        min_cell: float = 4.5,
        max_cell: float = 60.0,
        random_seed=None,
        optimize: bool = False,
        queue: str = "long",
        node: str = "aa",
        poll_interval: int = 10,
        timeout: int = 3600,
    ):
        import time

        script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data", "pormake", "_run_pormake.py",
        )
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        pormake_cmd = get_pormake_python_command() + [
            script, "random", str(n_mofs), str(save_dir),
            "--max_atoms", str(max_atoms),
            "--min_cell", str(min_cell),
            "--max_cell", str(max_cell),
        ]
        if random_seed is not None:
            pormake_cmd += ["--random_seed", str(random_seed)]

        pormake_qsub = save_dir / "run_pormake.qsub"
        pormake_log  = save_dir / "pormake.log"
        with open(pormake_qsub, "w") as f:
            f.write("#!/bin/sh\n")
            f.write("#PBS -r n\n")
            f.write(f"#PBS -q {queue}\n")
            f.write(f"#PBS -l nodes=1:ppn=1:{node}\n")
            f.write("#PBS -o /dev/null\n")
            f.write("#PBS -e /dev/null\n")
            f.write(f"cd {save_dir.absolute()}\n")
            f.write(" ".join(pormake_cmd) + f" 1>{pormake_log} 2>&1\n")

        r = subprocess.run(["qas", str(pormake_qsub)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"qas pormake failed:\n{r.stderr}")
        print(f"[pormake] submitted → job {r.stdout.strip()}")

        def _current_cifs():
            return {p for p in save_dir.glob("*.cif") if not p.name.endswith("_opt.cif")}

        if not optimize:
            elapsed = 0
            while elapsed < timeout:
                cifs = _current_cifs()
                if len(cifs) >= n_mofs:
                    return sorted(cifs)
                time.sleep(poll_interval)
                elapsed += poll_interval
            return sorted(_current_cifs())

        seen = set()
        cif_paths = []
        submissions = []
        elapsed = 0

        while elapsed < timeout:
            for cif in sorted(_current_cifs() - seen):
                seen.add(cif)
                cif_paths.append(cif)
                sub = self._prepare_optimize_job(cif, out_dir=save_dir, queue=queue, node=node)
                if sub:
                    submissions.append(sub)
            if len(seen) >= n_mofs:
                break
            time.sleep(poll_interval)
            elapsed += poll_interval

        if len(seen) < n_mofs:
            print(f"[pormake] timeout: {len(seen)}/{n_mofs} CIFs found")

        results = self._poll_optimize_jobs(submissions, timeout=timeout, poll_interval=poll_interval)
        return [results.get(cif, cif) for cif in cif_paths]

    def make_custom_hmof(
        self,
        topology_name: str,
        node_bbs: dict,
        out_path,
        *,
        edge_bbs: dict = None,
        optimize: bool = False,
    ):
        import json as _json

        script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data", "pormake", "_run_pormake.py",
        )
        out_path = Path(out_path)

        cmd = get_pormake_python_command() + [
            script, "custom",
            topology_name,
            _json.dumps({str(k): v for k, v in node_bbs.items()}),
            str(out_path),
        ]
        if edge_bbs:
            cmd += ["--edge_bbs", _json.dumps(edge_bbs)]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stderr.strip():
            print(result.stderr.strip())
        if result.returncode != 0:
            raise RuntimeError(f"pormake custom build failed:\n{result.stderr}")

        cif = None
        for line in result.stdout.splitlines():
            if line.startswith("CIF:"):
                cif = Path(line[4:])
                break
        cif = cif or out_path

        if optimize:
            opt = self.optimize_mof(cif, out_dir=cif.parent)
            return opt if opt else cif

        return cif

    def _resolve_guest_xyz(self, guest_name, save_dir, src_path=None):
        from .guest import GuestLoader, GuestNotFoundError

        out_xyz = Path(save_dir) / f"{guest_name}.xyz"

        if src_path:
            src = Path(src_path)
            if src.suffix == ".xyz":
                shutil.copy2(str(src), str(out_xyz))
            else:
                atoms = ase.io.read(str(src))
                ase.io.write(str(out_xyz), atoms, format="xyz")
            print(f"[GuestLoader] Using user-provided file: {src_path}")
            return str(out_xyz)

        try:
            g = GuestLoader(guest_name)
            g.get_guest(save_dir)
        except GuestNotFoundError:
            print(f"[StructureAgent] Guest '{guest_name}' not found on PubChem.")
            print("  - Enter a corrected guest name to retry search")
            print("  - Enter a file path (.xyz/.sdf) to use directly")
            print("  - Press Enter to abort")
            user_input = input(">> ").strip()
            if not user_input:
                raise GuestNotFoundError(f"'{guest_name}' not found and no file provided.")
            if any(user_input.endswith(ext) for ext in (".xyz", ".sdf", ".mol")):
                src = Path(user_input)
                if src.suffix == ".xyz":
                    shutil.copy2(str(src), str(out_xyz))
                else:
                    atoms = ase.io.read(str(src))
                    ase.io.write(str(out_xyz), atoms, format="xyz")
                print(f"[StructureAgent] Using user-provided file: {user_input}")
            else:
                print(f"[StructureAgent] Retrying with name: '{user_input}'")
                g = GuestLoader(user_input)
                g.get_guest(save_dir)
                new_xyz = Path(save_dir) / f"{user_input}.xyz"
                if new_xyz.exists() and user_input != guest_name:
                    shutil.copy2(str(new_xyz), str(out_xyz))

        return str(out_xyz)

    def preprocess_cif_dir(self, cif_dir, out_dir, keep_oxo=False):
        import shutil

        cif_dir = Path(cif_dir)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        cifs = sorted(cif_dir.glob("*.cif"))
        n_ok = 0
        n_fail = 0

        for cif in cifs:
            dst = out_dir / cif.name
            shutil.copy2(cif, dst)
            try:
                self._run_samosa(dst, keep_oxo=keep_oxo)
                self._after_fetch(dst, out_dir)
                n_ok += 1
            except Exception as e:
                print(f"[preprocess] failed {cif.name}: {e}")
                dst.unlink(missing_ok=True)
                n_fail += 1

        print(f"[preprocess] {n_ok}/{len(cifs)} CIFs preprocessed, {n_fail} failed/dropped")
        return str(out_dir)

    def get_guest(self, guest_name, save_dir, src_path=None):
        self._resolve_guest_xyz(guest_name, save_dir, src_path=src_path)
        return

    def _build_fetch_script(self, mof_name, save_dir):
        project_root = os.path.dirname(os.path.abspath(__file__))
        return f"""
import sys
sys.path.append('{project_root}')
from pathlib import Path
from structure.mof import MOFLoader
m = MOFLoader('{mof_name}')
m.get_structure(Path('{save_dir}'))
"""

    def _run_fetch_subprocess(self, mof_name, save_dir):
        script = self._build_fetch_script(mof_name, save_dir)
        result = subprocess.run(
                get_csd_api_python_command() + ["-c", script],
                capture_output=True,
                text=True
            )
        if result.returncode != 0:
            print(f"Error fetching {mof_name}")
            print(result.stderr)
            raise RuntimeError(f"Failed to fetch {mof_name}")

        if "No REFCODE found" in result.stdout:
            raise MOFNotFoundError(f"'{mof_name}' not found in CoREMOF or CSD.")

        print(f"Successfully fetched {mof_name} structure")

    def _run_samosa(self, mof_path, keep_oxo=False):
        samosa_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "SAMOSA", "run_single.py"
        )
        cmd = get_csd_api_python_command() + [samosa_script, str(mof_path)]
        if keep_oxo:
            cmd.append("--keep_oxo")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[SAMOSA] Warning: solvent removal failed for {mof_path}")
            print(result.stderr)
        else:
            if result.stdout.strip():
                print(result.stdout.strip())

    def _build_mof_path(self, mof_name, save_dir):
        return Path(save_dir) / f"{mof_name}.cif"

    def _get_mof_return_value(self, mof_path):
        return

    def _prompt_build_hmof(self, mof_name, save_dir):
        import json as _json

        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "pormake")
        print(f"[hMOF Builder] Topology dir : {os.path.join(data_dir, 'topologies')}")
        print(f"[hMOF Builder] BB names     : N<n> for nodes, E<n> for edges (e.g. N10, E41)")
        print()

        print("[hMOF Builder] Topology name (e.g. tbo, pcu, acs):")
        topology = input("  topology >> ").strip()
        if not topology:
            raise ValueError("Topology name is required.")

        print("[hMOF Builder] Node BBs as JSON {\"node_index\": \"bb_name\"}")
        print("  e.g. {\"0\": \"N10\", \"1\": \"N409\"}")
        node_bbs_str = input("  node_bbs >> ").strip()
        try:
            node_bbs = {int(k): v for k, v in _json.loads(node_bbs_str).items()}
        except Exception as e:
            raise ValueError(f"Invalid node_bbs JSON: {e}")

        print("[hMOF Builder] Edge BBs as JSON {\"i,j\": \"bb_name\"} (Enter to skip):")
        print("  e.g. {\"0,1\": \"E41\"}")
        edge_bbs_str = input("  edge_bbs >> ").strip()
        edge_bbs = None
        if edge_bbs_str:
            try:
                edge_bbs = _json.loads(edge_bbs_str)
            except Exception as e:
                raise ValueError(f"Invalid edge_bbs JSON: {e}")

        out_path = Path(save_dir) / f"{mof_name}.cif"
        print(f"[hMOF Builder] Building: topology={topology}, node_bbs={node_bbs}, edge_bbs={edge_bbs}")

        built_cif = self.make_custom_hmof(
            topology_name=topology,
            node_bbs=node_bbs,
            out_path=out_path,
            edge_bbs=edge_bbs,
        )

        built_cif = Path(built_cif)
        if built_cif.resolve() != out_path.resolve() and built_cif.exists():
            shutil.copy2(str(built_cif), str(out_path))
            print(f"[hMOF Builder] CIF saved to: {out_path}")

    def get_mof(
        self,
        mof_name,
        save_dir,
        src_cif_path=None,
        prompt_on_missing=True,
    ):
        if src_cif_path:
            dst = Path(save_dir) / f"{mof_name}.cif"
            shutil.copy2(src_cif_path, dst)
            print(f"[StructureAgent] Using user-provided CIF: {src_cif_path}")
            self._run_samosa(dst, keep_oxo=True)
        else:
            try:
                self._run_fetch_subprocess(mof_name, save_dir)
            except MOFNotFoundError:
                if not prompt_on_missing:
                    raise
                print(f"[StructureAgent] '{mof_name}' not found in CoREMOF or CSD.")
                print("  - Enter a corrected MOF name to retry search")
                print("  - Enter a CIF file path (.cif) to use directly")
                print("  - Type 'hmof' to build a custom hMOF via pormake")
                print("  - Press Enter to abort")
                user_input = input(">> ").strip()
                if not user_input:
                    raise MOFNotFoundError(f"'{mof_name}' not found and no CIF path provided.")
                if user_input.endswith(".cif"):
                    dst = Path(save_dir) / f"{mof_name}.cif"
                    shutil.copy2(user_input, dst)
                    print(f"[StructureAgent] Using user-provided CIF: {user_input}")
                    self._run_samosa(dst, keep_oxo=True)
                elif user_input.lower() == "hmof":
                    self._prompt_build_hmof(mof_name, save_dir)
                else:
                    print(f"[StructureAgent] Retrying with name: '{user_input}'")
                    mof_name = user_input
                    self._run_fetch_subprocess(mof_name, save_dir)
                    mof_path = self._build_mof_path(mof_name, save_dir)
                    self._run_samosa(mof_path)
            else:
                mof_path = self._build_mof_path(mof_name, save_dir)
                self._run_samosa(mof_path)

        mof_path = self._build_mof_path(mof_name, save_dir)
        self._after_fetch(mof_path, save_dir)
        validate_mof(mof_path, save_dir)

        return self._get_mof_return_value(mof_path)



class VASPStructureAgent(StructureAgent):
    def __init__(self, number_of_guest: int = 1, number_of_system: int = 5):
        self.number_of_guest = number_of_guest
        self.number_of_system = number_of_system

    def _attempt_automatic_cif_recovery(
        self,
        context,
        save_dir,
        initial_error,
    ):
        from error.structure_regeneration import request_structure_regeneration

        attempt = int(
            context.get("vasp_structure_recovery_attempts", 0) or 0
        ) + 1
        context["vasp_structure_recovery_attempts"] = attempt
        context["vasp_structure_recovery_attempted"] = True
        recovery_dir = (
            Path(save_dir)
            / "structure_regeneration"
            / f"attempt_{attempt:02d}"
        )
        recovery_dir.mkdir(parents=True, exist_ok=True)
        mof_name = str(context["mof"])
        fetched = recovery_dir / f"{mof_name}.cif"

        try:
            self._run_fetch_subprocess(mof_name, str(recovery_dir))
            if not fetched.is_file():
                raise FileNotFoundError(
                    f"structure source did not create {fetched}"
                )
            self._run_samosa(fetched)
            self._after_fetch(fetched, recovery_dir)
            validate_mof(fetched, recovery_dir)
        except Exception as recovery_error:
            recovery = {
                "status": "needs_structure_from_user",
                "action": "automatic_refetch_and_validate",
                "attempt": attempt,
                "mof": mof_name,
                "initial_error": (
                    f"{type(initial_error).__name__}: {initial_error}"
                ),
                "recovery_error": (
                    f"{type(recovery_error).__name__}: {recovery_error}"
                ),
                "recovery_dir": str(recovery_dir),
            }
            context["vasp_status"] = "needs_structure_from_user"
            context["vasp_state"] = "giveup"
            context.setdefault("results", {})[
                "vasp_structure_recovery"
            ] = recovery
            request_structure_regeneration(
                context,
                software="vasp",
                reason=(
                    "Automatic CIF refetch and validation failed: "
                    f"{recovery['recovery_error']}"
                ),
                action="needs_user_cif",
                status="blocked",
                metadata=recovery,
            )
            return None

        target = Path(save_dir) / fetched.name
        if fetched.resolve() != target.resolve():
            shutil.copy2(fetched, target)
        recovery = {
            "status": "recovered",
            "action": "automatic_refetch_and_validate",
            "attempt": attempt,
            "mof": mof_name,
            "initial_error": (
                f"{type(initial_error).__name__}: {initial_error}"
            ),
            "cif": str(target),
            "recovery_dir": str(recovery_dir),
        }
        context.setdefault("results", {})[
            "vasp_structure_recovery"
        ] = recovery
        context["vasp_status"] = "structure_recovered"
        return str(target)

    def _prepare_mof_with_recovery(self, context, save_dir):
        try:
            return self.get_mof(
                context["mof"],
                save_dir,
                src_cif_path=context.get("cif_path"),
                prompt_on_missing=False,
            )
        except Exception as initial_error:
            return self._attempt_automatic_cif_recovery(
                context,
                save_dir,
                initial_error,
            )

    def _after_fetch(self, mof_path, save_dir):
        mof_path = str(mof_path)

        try:
            atoms = ase.io.read(mof_path)
            ase.io.write(mof_path, atoms, format="cif")
            print(f"[CIF] Cleaned MOF CIF written to: {mof_path}")
        except Exception as e:
            print(f"[CIF] Warning: failed to clean MOF CIF ({mof_path}): {e}")

    def _get_mof_return_value(self, mof_path):
        return str(mof_path)

    def get_guest(self, guest_name, save_dir, mof_path=None, src_path=None):
        self._resolve_guest_xyz(guest_name, save_dir, src_path=src_path)
        guest_xyz_path = os.path.join(save_dir, f"{guest_name}.xyz")

        guest_cif_path = None

        
        if mof_path is not None:
            try:
                
                mof_atoms = ase.io.read(mof_path)
                mof_cell = mof_atoms.cell

                
                guest_atoms = ase.io.read(guest_xyz_path)
                guest_atoms.set_cell(mof_cell)


                guest_cif_path = os.path.join(save_dir, f"{guest_name}.cif")
                ase.io.write(guest_cif_path, guest_atoms, format="cif")
                print(f"[Guest] Guest CIF with MOF cell written to: {guest_cif_path}")
            except Exception as e:
                print(f"[Guest] Warning: failed to set MOF cell on guest: {e}")

        
        return guest_xyz_path, guest_cif_path

    
    
    
    def get_complex(self, mof_path, guest_xyz_path, save_dir):

        packmol_out_dir = os.path.join(save_dir, "packmol")
        os.makedirs(packmol_out_dir, exist_ok=True)

        print(f"Running Packmol for complex in: {packmol_out_dir}")

        
        run_packmol_from_cif(
            cif_file=mof_path,
            guest_xyz=guest_xyz_path,
            number_of_guest=self.number_of_guest,
            number_of_system=self.number_of_system,
            output_dir=packmol_out_dir,
            tolerance=PACKMOL_DEFAULT_TOLERANCE,
        )

        
        
        
        cif_name = os.path.splitext(os.path.basename(mof_path))[0]
        guest_name = os.path.splitext(os.path.basename(guest_xyz_path))[0]
        output_subdir = os.path.join(packmol_out_dir, f"{cif_name}_{guest_name}")

        complex_cif_paths = sorted(glob.glob(os.path.join(output_subdir, "*.cif")))

        if complex_cif_paths:
            print("Complex CIFs generated:")
            for p in complex_cif_paths:
                print("   -", p)
        else:
            print("No complex CIFs found in", output_subdir)

        return complex_cif_paths

    
    
    
    def run(self, context):

        mof_name   = context["mof"]
        guest_name = context["guest"]
        job_name   = context["job_name"]
        batch_root = context.get("batch_root")

        save_dir = self._resolve_job_work_dir(context, job_name, batch_root=batch_root)
        save_dir = self._set_context_work_dir(context, save_dir)

        print(f"Saving output to: {save_dir}")


        mof_path = self._prepare_mof_with_recovery(context, save_dir)
        if mof_path is None:
            return context
        context["mof_path"] = mof_path

        guest_xyz_path = None
        guest_cif_path = None

        
        if guest_name:
            guest_xyz_path, guest_cif_path = self.get_guest(
                guest_name, save_dir, mof_path=mof_path,
                src_path=context.get("guest_src_path"),
            )
            context["guest_path"] = guest_xyz_path      
            context["guest_cif_path"] = guest_cif_path  

        
        if guest_name and guest_xyz_path is not None:
            complex_cif_paths = self.get_complex(
                mof_path=mof_path,
                guest_xyz_path=guest_xyz_path,
                save_dir=save_dir,
            )
            
            context["complex_cif_paths"] = complex_cif_paths

        context["tool_mode"] = "mlip_binding"
        
        return context

    def run_mof_only(self, context: dict) -> dict:
        mof_name = context["mof"]
        job_name = context["job_name"]

        save_dir = self._resolve_job_work_dir(context, job_name)
        save_dir = self._set_context_work_dir(context, save_dir)

        mof_path = self._prepare_mof_with_recovery(context, save_dir)
        if mof_path is None:
            return context
        context["mof_path"] = mof_path

        return context

    def run_guest_and_complex_from_optimized(self, context: dict) -> dict:
        mof_path = context.get("mof_path")
        guest_name = context.get("guest")
        save_dir = context.get("work_dir")

        if not mof_path or not os.path.exists(mof_path):
            raise FileNotFoundError(f"[VASPStructureAgent] optimized mof_path missing: {mof_path}")
        if not guest_name:
            raise ValueError("[VASPStructureAgent] guest is missing in context")
        if not save_dir:
            raise ValueError("[VASPStructureAgent] work_dir missing in context")


        guest_xyz = os.path.join(save_dir, f"{guest_name}.xyz")
        guest_cif = os.path.join(save_dir, f"{guest_name}.cif")

        if os.path.exists(guest_xyz) and os.path.exists(guest_cif):
            guest_xyz_path = guest_xyz
            guest_cif_path = guest_cif
        else:
            guest_xyz_path, guest_cif_path = self.get_guest(
                guest_name, save_dir, mof_path=mof_path,
                src_path=context.get("guest_src_path"),
            )

        context["guest_path"] = guest_xyz_path
        context["guest_cif_path"] = guest_cif_path

        
        complex_cif_paths = self.get_complex(
            mof_path=mof_path,
            guest_xyz_path=guest_xyz_path,
            save_dir=save_dir,
        )
        context["complex_cif_paths"] = complex_cif_paths

        
        if complex_cif_paths:
            context["complex_cif_path"] = complex_cif_paths[0]
            context["complex_path"] = context["complex_cif_path"]
            complex_label = Path(context["complex_cif_path"]).stem
            context["vasp_label"] = complex_label
            context.setdefault("vasp_system", {})
            context["vasp_system"]["label"] = complex_label

        return context

class ZeoppStructureAgent(StructureAgent):

    def __init__(self):
        pass

    def run(self, context):

        mof_name   = context["mof"]
        guest_name = context["guest"]


        save_dir = self._resolve_job_work_dir(context, context["job_name"])
        save_dir = self._set_context_work_dir(context, save_dir)

        print(f"Saving output to: {save_dir}")


        self.get_mof(mof_name, save_dir, src_cif_path=context.get("cif_path"))
        context["mof_path"] = os.path.join(save_dir, f"{mof_name}.cif")

        return context

    
class LAMMPSStructureAgent(StructureAgent):
    def __init__(self):
        pass

    def _attempt_automatic_cif_recovery(
        self,
        context,
        save_dir,
        initial_error,
    ):
        from error.structure_regeneration import request_structure_regeneration

        attempt = int(
            context.get("lammps_structure_recovery_attempts", 0) or 0
        ) + 1
        context["lammps_structure_recovery_attempts"] = attempt
        context["lammps_structure_recovery_attempted"] = True
        recovery_dir = (
            Path(save_dir)
            / "structure_regeneration"
            / f"attempt_{attempt:02d}"
        )
        recovery_dir.mkdir(parents=True, exist_ok=True)
        mof_name = str(context["mof"])
        fetched = recovery_dir / f"{mof_name}.cif"

        try:
            self._run_fetch_subprocess(mof_name, str(recovery_dir))
            if not fetched.is_file():
                raise FileNotFoundError(
                    f"structure source did not create {fetched}"
                )
            self._run_samosa(fetched)
            self._after_fetch(fetched, recovery_dir)
            validate_mof(fetched, recovery_dir)
        except Exception as recovery_error:
            recovery = {
                "status": "needs_structure_from_user",
                "action": "automatic_refetch_and_validate",
                "attempt": attempt,
                "mof": mof_name,
                "initial_error": (
                    f"{type(initial_error).__name__}: {initial_error}"
                ),
                "recovery_error": (
                    f"{type(recovery_error).__name__}: {recovery_error}"
                ),
                "recovery_dir": str(recovery_dir),
            }
            context["lammps_status"] = "needs_structure_from_user"
            context["lammps_success"] = False
            context.setdefault("results", {})[
                "lammps_structure_recovery"
            ] = recovery
            request_structure_regeneration(
                context,
                software="lammps",
                reason=(
                    "Automatic CIF refetch and validation failed: "
                    f"{recovery['recovery_error']}"
                ),
                action="needs_user_cif",
                status="blocked",
                metadata=recovery,
            )
            return None

        target = Path(save_dir) / fetched.name
        if fetched.resolve() != target.resolve():
            shutil.copy2(fetched, target)
        recovery = {
            "status": "recovered",
            "action": "automatic_refetch_and_validate",
            "attempt": attempt,
            "mof": mof_name,
            "initial_error": (
                f"{type(initial_error).__name__}: {initial_error}"
            ),
            "cif": str(target),
            "recovery_dir": str(recovery_dir),
        }
        context.setdefault("results", {})[
            "lammps_structure_recovery"
        ] = recovery
        context["lammps_status"] = "structure_recovered"
        return target

    def run(self, context):

        mof_name   = context["mof"]
        guest_name = context["guest"]

        
        save_dir = self._resolve_job_work_dir(context, context["job_name"])
        save_dir = self._set_context_work_dir(context, save_dir)

        print(f"Saving output to: {save_dir}")


        try:
            self.get_mof(
                mof_name,
                save_dir,
                src_cif_path=context.get("cif_path"),
                prompt_on_missing=False,
            )
            mof_path = Path(save_dir) / f"{mof_name}.cif"
        except Exception as initial_error:
            mof_path = self._attempt_automatic_cif_recovery(
                context,
                save_dir,
                initial_error,
            )
            if mof_path is None:
                return context

        context["mof_path"] = str(mof_path)


        if guest_name:
            self.get_guest(guest_name, save_dir, src_path=context.get("guest_src_path"))
            context["guest_path"] = os.path.join(save_dir, f"{guest_name}.xyz")

        return context

class RASPAStructureAgent(StructureAgent):
    def __init__(self):
        pass

    def convert_cif_to_p1(self, cif_path, out_path=None, backup: bool = True):
        cif_path = Path(cif_path)
        out_path = Path(out_path) if out_path else cif_path
        if not cif_path.is_file():
            raise FileNotFoundError(f"[RASPAStructureAgent] CIF not found for P1 conversion: {cif_path}")

        backup_path = None
        if backup and out_path.exists():
            backup_path = out_path.with_name(f"{out_path.name}.simmof_before_p1")
            shutil.copy2(out_path, backup_path)

        atoms = ase.io.read(str(cif_path))
        ase.io.write(str(out_path), atoms, format="cif")
        self._patch_legacy_symmetry_tags(str(out_path))
        print(f"[RASPAStructureAgent] CIF converted to P1-style CIF: {out_path}")
        return {
            "status": "ok",
            "input_cif": str(cif_path),
            "output_cif": str(out_path),
            "backup": str(backup_path) if backup_path else None,
        }

    def _after_fetch(self, mof_path, save_dir):
        mof_path = str(mof_path)
        try:
            text = Path(mof_path).read_text(errors="replace")
            if "_atom_site_charge" not in text:
                self._clean_cif_with_ase(mof_path)
            else:
                import re as _re
                fixed = _re.sub(
                    r"('x, y, z'\s*\n)(\s*_atom_site_)",
                    r"\1\nloop_\n\2",
                    text,
                )
                if fixed != text:
                    Path(mof_path).write_text(fixed)
                    print(f"[RASPAStructureAgent] Fixed PACMAN CIF loop bug: {Path(mof_path).name}")
        except Exception as e:
            print(f"[RASPAStructureAgent] _after_fetch warning: {e}")

    def run(self, context):
        job_name = context["job_name"]
        guest_name = context.get("guest")

        
        base_dir = Path(self._resolve_job_work_dir(context, job_name))
        base_dir = Path(self._set_context_work_dir(context, base_dir))
        print(f"Saving output to: {base_dir}")

        
        screening_okdir = context.get("screening_okdir")
        if screening_okdir:
            okdir = Path(screening_okdir)
            cif_files = sorted(okdir.glob("*.cif"))
            if not cif_files:
                raise FileNotFoundError(f"[RASPAStructureAgent] No CIFs found in screening_okdir: {okdir}")

            batch = []
            batch_root = base_dir / "batch"
            batch_root.mkdir(parents=True, exist_ok=True)

            for cif in cif_files:
                stem = cif.stem  

                
                work_dir = batch_root / f"{stem}_raspa"
                work_dir.mkdir(parents=True, exist_ok=True)

                
                local_cif = work_dir / f"{stem}.cif"
                if not local_cif.exists():
                    shutil.copy2(cif, local_cif)

                subctx = copy.deepcopy(context)
                subctx["mof"] = stem
                subctx["mof_path"] = str(local_cif)
                subctx["work_dir"] = str(work_dir)

                
                subctx["job_name"] = f"{job_name}__{stem}"

                batch.append(subctx)

            self._set_context_work_dir(context, base_dir)
            context["batch"] = batch
            context["batch_size"] = len(batch)

            print(f"[RASPAStructureAgent] batch created: {len(batch)} MOFs from {okdir}")
            return context

        
        mof_name = context["mof"]

        
        self._set_context_work_dir(context, base_dir)

        self.get_mof(mof_name, str(base_dir), src_cif_path=context.get("cif_path"))
        context["mof_path"] = str(base_dir / f"{mof_name}.cif")

        return context
