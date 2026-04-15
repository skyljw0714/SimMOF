import json
import os
import csv
import shutil
import re
import math
import subprocess
import time

from ase.io import read
from pathlib import Path
from collections import OrderedDict
from ase.optimize import BFGS
from typing import Dict, Any, Optional
from config import LLM_DEFAULT, RASPA_DIR, RASPA_SIMULATE_BIN

RASPA_QSUB_QUEUE = "long"
RASPA_QSUB_RESOURCES = "nodes=1:ppn=8:aa"
SCREENING_DEFAULT_HENRY_TEMPERATURE_K = 298.0
RASPA_HENRY_NUMBER_OF_CYCLES = 5000
RASPA_HENRY_WIDOM_INSERTIONS = 10000
SCREENING_DEFAULT_TOP_N = 100

HARD_FAIL_FLAGS = [
    "has_atomic_overlaps",
    "has_overvalent_c",
    "has_overvalent_n",
    "has_overvalent_h",
    "has_undercoordinated_c",
    "has_undercoordinated_n",
    "has_undercoordinated_metal",
    "has_lone_atom",
    "has_lone_molecule",
]

REQUIRE_HAS_METAL = True
REQUIRE_HAS_CARBON = False
REQUIRE_HAS_HYDROGEN = False

def run_mofchecker(cif_dir: str, okdir: str):
    from mofchecker import MOFChecker

    cif_dir = Path(cif_dir)
    if not cif_dir.is_dir():
        raise ValueError(f"Not a directory: {cif_dir}")

    files = sorted(cif_dir.glob("*.cif"))

    good_cifs = []
    for path in files:
        record = OrderedDict()
        record["file"] = str(path)

        try:
            mc = MOFChecker.from_cif(path, primitive=True)
        except Exception:
            continue

        def safe_getattr(obj, name):
            try:
                val = getattr(obj, name)
                return bool(val) if isinstance(val, bool) else None
            except Exception:
                return None

        flags = {name: safe_getattr(mc, name) for name in [
            "has_metal",
            "has_carbon",
            "has_hydrogen",
            "has_atomic_overlaps",
            "has_overvalent_c",
            "has_overvalent_n",
            "has_overvalent_h",
            "has_undercoordinated_c",
            "has_undercoordinated_n",
            "has_undercoordinated_metal",
            "has_lone_atom",
            "has_lone_molecule",
        ]}

        reasons = []
        for k in HARD_FAIL_FLAGS:
            if flags.get(k) is True:
                reasons.append(k)
        if REQUIRE_HAS_METAL and flags.get("has_metal") is False:
            reasons.append("missing_metal")
        if REQUIRE_HAS_CARBON and flags.get("has_carbon") is False:
            reasons.append("missing_carbon")
        if REQUIRE_HAS_HYDROGEN and flags.get("has_hydrogen") is False:
            reasons.append("missing_hydrogen")

        if not reasons:
            good_cifs.append(str(path))

    if okdir:
        okdir = Path(okdir)
        okdir.mkdir(parents=True, exist_ok=True)
        for f in good_cifs:
            shutil.copy2(f, okdir / Path(f).name)

    return good_cifs


def run_omd(cif_dir: str, okdir: str):

    from omsdetector import MofCollection

    analysis_dir = os.path.join(cif_dir, "_analysis")
    num_batches = 1

    mof_coll = MofCollection.from_folder(
        collection_folder=cif_dir,
        analysis_folder=analysis_dir
    )
    mof_coll.analyse_mofs(num_batches=num_batches)

    oms_results_dir = os.path.join(analysis_dir, "oms_results")
    oms_list = []

    for mof_name in os.listdir(oms_results_dir):
        mof_dir = os.path.join(oms_results_dir, mof_name)
        json_file = os.path.join(mof_dir, f"{mof_name}.json")
        if os.path.isfile(json_file):
            with open(json_file) as f:
                data = json.load(f)
            if data.get("has_oms", False):
                oms_list.append(mof_name)

    if okdir and oms_list:
        os.makedirs(okdir, exist_ok=True)
        for mof_name in oms_list:
            cif_path = os.path.join(cif_dir, f"{mof_name}.cif")
            if os.path.exists(cif_path):
                shutil.copy2(cif_path, os.path.join(okdir, f"{mof_name}.cif"))
                
    if oms_list:
        print("MOFs with OMS detected:")
        for name in oms_list:
            print(" -", name)
    else:
        print("No MOFs with OMS detected.")

    return oms_list

def run_ase_atom_count(cif_dir: str, okdir: str, max_atoms: int):
    cif_dir = Path(cif_dir)
    okdir = Path(okdir)
    if not cif_dir.is_dir():
        raise ValueError(f"Not a directory: {cif_dir}")

    files = sorted(cif_dir.glob("*.cif"))
    good_cifs = []

    for path in files:
        try:
            atoms = read(path)
            if len(atoms) < max_atoms:
                good_cifs.append(str(path))
                okdir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, okdir / path.name)
        except Exception as e:
            print(f"Failed to read {path}: {e}")
            continue

    return good_cifs


def run_ase_atom_type(cif_dir: str, okdir: str, atom_type: str):
    cif_dir = Path(cif_dir)
    if not cif_dir.is_dir():
        raise ValueError(f"Not a directory: {cif_dir}")

    files = sorted(cif_dir.glob("*.cif"))

    good_cifs = []
    for path in files:
        try:
            atoms = read(path)
            elements = [atom.symbol for atom in atoms]
            if atom_type in elements:
                good_cifs.append(str(path))
                shutil.copy2(path, okdir / path.name)
        except Exception as e:
            print(f"Failed to read {path}: {e}")
            continue

    return good_cifs

def optimize_structure(atoms, device="cpu", fmax=0.02, max_steps=200):
    from mace.calculators import mace_mp

    calc = mace_mp(model="large", dispersion=True, default_dtype="float64", device=device)
    atoms.calc = calc

    dyn = BFGS(atoms, logfile=None)
    converged = dyn.run(fmax=fmax, steps=max_steps)

    if not converged:
        raise RuntimeError(f"MLIP optimization did not converge within {max_steps} steps")

    return atoms, atoms.get_potential_energy()


def run_mlip_geo(cif_dir: str, okdir: str, top_n: int = 100, device: str = "cpu"):
    cif_dir = Path(cif_dir)
    okdir = Path(okdir)
    okdir.mkdir(parents=True, exist_ok=True)
    output_csv = Path("mlip_geo_results.csv")

    cif_files = sorted(cif_dir.glob("*.cif"))
    total = len(cif_files)
    if total == 0:
        raise RuntimeError(f"No CIFs found in {cif_dir}")

    results = []
    for cif in cif_files:
        try:
            atoms = read(cif)
            atoms.set_pbc(True)
            opt_atoms, energy = optimize_structure(atoms, device=device)
            results.append({"MOF": cif.stem, "Energy(eV)": energy, "path": str(cif)})
            print(f"[MLIP_geo] {cif.name}: {energy:.4f} eV")
        except Exception as e:
            print(f"[MLIP_geo] Failed {cif.name}: {e}")

    results_sorted = sorted(results, key=lambda x: x["Energy(eV)"])
    keep_n = min(top_n, len(results_sorted))
    kept = results_sorted[:keep_n]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["MOF", "Energy(eV)", "path"])
        writer.writeheader()
        for row in results_sorted:
            writer.writerow(row)

    for row in kept:
        src = Path(row["path"])
        dst = okdir / src.name
        shutil.copy2(src, dst)

    print(f"[MLIP_geo] Total={total}, Kept={len(kept)}, CSV={output_csv}")
    return [row["path"] for row in kept]

def run_binding_energy(host_cif, host_guest_cif, guest_xyz, device="cpu"):
    host = read(host_cif)
    host_guest = read(host_guest_cif)
    guest = read(guest_xyz)

    host.set_pbc(True)
    host_guest.set_pbc(True)
    guest.set_pbc(False)

    host, e_host = optimize_structure(host, device)
    host_guest, e_host_guest = optimize_structure(host_guest, device)
    guest, e_guest = optimize_structure(guest, device)

    return e_host_guest - (e_host + e_guest)


def run_mlip_binding(input_dir, guest_xyz, okdir, device="cpu", top_n=100):
    input_dir = Path(input_dir)
    okdir = Path(okdir)
    okdir.mkdir(parents=True, exist_ok=True)

    cif_files = sorted(input_dir.glob("*.cif"))
    if len(cif_files) == 0:
        raise RuntimeError(f"No CIFs found in {input_dir}")

    results = []
    for cif in cif_files:
        try:
            host_guest_dir = input_dir / f"{cif.stem}_{Path(guest_xyz).stem}"
            host_guest_cif = host_guest_dir / f"{cif.stem}_{Path(guest_xyz).stem}_1.cif"

            if not host_guest_cif.exists():
                print(f"[MLIP_BE] Skipping {cif.name}, no {host_guest_cif.name}")
                continue

            be = run_binding_energy(cif, host_guest_cif, guest_xyz, device=device)
            results.append({"MOF": cif.stem, "BindingEnergy(eV)": be, "path": str(cif)})
            print(f"[MLIP_BE] {cif.name}: {be:.4f} eV")

        except Exception as e:
            print(f"[MLIP_BE] Failed {cif.name}: {e}")

    
    results_sorted = sorted(results, key=lambda x: x["BindingEnergy(eV)"])
    keep_n = min(top_n, len(results_sorted))
    kept = results_sorted[:keep_n]

    
    output_csv = input_dir / "mlip_binding_results.csv"
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["MOF", "BindingEnergy(eV)", "path"])
        writer.writeheader()
        for row in results_sorted:
            writer.writerow(row)

    
    for row in kept:
        src = Path(row["path"])
        dst = okdir / src.name
        shutil.copy2(src, dst)

    print(f"[MLIP_BE] Total={len(cif_files)}, Kept={len(kept)}, CSV={output_csv}")
    return [row["path"] for row in kept]

def run_mlip_complex_candidates(
    complex_cif_paths,
    okdir,
    device="cpu",
    top_n=1,
):
    okdir = Path(okdir)
    okdir.mkdir(parents=True, exist_ok=True)

    results = []

    for i, cif_path in enumerate(complex_cif_paths):
        cif_path = Path(cif_path)
        cand_dir = okdir / f"cand_{i:02d}"
        cand_dir.mkdir(parents=True, exist_ok=True)

        try:
            atoms = read(cif_path)
            atoms.set_pbc(True)

            opt_atoms, energy = optimize_structure(
                atoms,
                device=device,
                fmax=0.02,
                max_steps=200,
            )

            relaxed_cif = cand_dir / f"{cif_path.stem}_mlip_relaxed.cif"
            from ase.io import write
            write(relaxed_cif, opt_atoms, format="cif")

            results.append({
                "index": i,
                "input_cif": str(cif_path),
                "relaxed_cif": str(relaxed_cif),
                "energy_ev": float(energy),
                "status": "ok",
            })
            print(f"[MLIP_COMPLEX] {cif_path.name}: {energy:.6f} eV")

        except Exception as e:
            results.append({
                "index": i,
                "input_cif": str(cif_path),
                "relaxed_cif": None,
                "energy_ev": None,
                "status": f"failed: {e}",
            })
            print(f"[MLIP_COMPLEX] Failed {cif_path.name}: {e}")

    ok_results = [r for r in results if r["status"] == "ok" and r["energy_ev"] is not None]
    if not ok_results:
        raise RuntimeError("No successful MLIP-relaxed complex candidates")

    results_sorted = sorted(ok_results, key=lambda x: x["energy_ev"])
    kept = results_sorted[:min(top_n, len(results_sorted))]

    import json
    with open(okdir / "mlip_complex_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return {
        "all_results": results,
        "top_results": kept,
        "best_result": kept[0],
    }
    
def _cmp(val: float, op: str, thr: float) -> bool:
    if op == ">=": return val >= thr
    if op == "<=": return val <= thr
    if op == ">":  return val > thr
    if op == "<":  return val < thr
    if op == "==": return val == thr
    if op == "!=": return val != thr
    raise ValueError(f"Unknown op: {op}")


def run_zeopp(
    *,
    step: Dict[str, Any],
    goal: str,
    input_cif_dir: str,
    okdir: str,
    work_dir: Optional[str] = None,
    llm=None,
    base_context: Optional[Dict[str, Any]] = None,
    max_retries: int = 2,
) -> list:
    from input.zeopp_input import ZeoppInputAgent
    from Zeopp.runner import ZeoppRunner
    from output.zeopp_output import ZeoppOutputAgent
    from error.zeopp_error import ZeoppErrorAgent
    from tool.parsing import extract_conditions_with_llm

    base_context = base_context or {}

    input_cif_dir = str(input_cif_dir)
    okdir = str(okdir)

    Path(okdir).mkdir(parents=True, exist_ok=True)

    params = extract_conditions_with_llm(step, goal=goal, llm=llm)

    prop = params["property"]
    op = params["op"]
    thr = float(params["value"])

    zeopp_info_base = {
        "command": params["zeopp_command"],
        "probe_radius": params.get("probe_radius"),
        "num_samples": params.get("num_samples"),
    }

    zia = ZeoppInputAgent(llm=None)
    runner = ZeoppRunner()
    parser = ZeoppOutputAgent()
    error_agent = ZeoppErrorAgent(
        llm=llm,
        max_retries=max_retries,
        zeopp_runner=runner,
        zeopp_input_agent=zia,
    )

    passed = []

    
    exec_dir = input_cif_dir

    for cif_path in sorted(Path(input_cif_dir).glob("*.cif")):
        mof = cif_path.stem

        zeopp_info = dict(zeopp_info_base)
        zeopp_info["MOF"] = mof

        
        cmd = zia._get_zeopp_command(zeopp_info, cif_dir=exec_dir)

        ctx = {
            **base_context,
            "work_dir": exec_dir,         
            "mof": mof,
            "zeopp_info": zeopp_info,
            "zeopp_command": cmd,
            "results": {},
        }

        ctx = runner.run(ctx)
        ctx = error_agent.run(ctx)

        if ctx.get("results", {}).get("zeopp_status") != "ok":
            continue

        ctx = parser.run(ctx)
        raw = ctx.get("results", {}).get("zeopp", {}).get("raw", {})

        if prop not in raw:
            continue

        val = float(raw[prop])
        if _cmp(val, op, thr):
            passed.append(str(cif_path))
            shutil.copy2(str(cif_path), str(Path(okdir) / cif_path.name))

    return passed




RASPA_DIR = Path(RASPA_DIR)

HENRY_RE = re.compile(
    r"\s*\[[^\]]+\]\s*Average Henry coefficient:\s*([0-9Ee\+\-\.]+)\s*(?:\+/-|±)\s*([0-9Ee\+\-\.]+)\s*\[([^\]]+)\]"
)

def _parse_henry_from_text(text: str):
    for line in text.splitlines():
        if "Average Henry coefficient" in line and "[" in line:
            m = HENRY_RE.search(line)
            if m:
                return float(m.group(1)), float(m.group(2)), m.group(3)
    return None, None, None

def _submit_raspa_job(cif_file, molecule="hydrogen", output_dir="raspa_output",
                     number_of_cycles=RASPA_HENRY_NUMBER_OF_CYCLES, temperature=SCREENING_DEFAULT_HENRY_TEMPERATURE_K, n_widom=RASPA_HENRY_WIDOM_INSERTIONS):
    cif_file = Path(cif_file)
    framework_name = cif_file.stem
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    
    raspa_cif_dir = RASPA_DIR / "share/raspa/structures/cif"
    raspa_cif_dir.mkdir(parents=True, exist_ok=True)
    target_cif = raspa_cif_dir / f"{framework_name}.cif"
    shutil.copy2(cif_file, target_cif)

    
    input_file = output_dir / "simulation.input"
    with open(input_file, "w") as f:
        f.write(f"""
SimulationType                MonteCarlo
NumberOfCycles                {number_of_cycles}
NumberOfInitializationCycles  {number_of_cycles // 10}
PrintEvery                    1000

Forcefield                    UFF

Framework 0
FrameworkName                 {framework_name}
UnitCells                     1 1 1
ExternalTemperature           {temperature}

CalculationType               Widom
NumberOfWidomInsertions       {n_widom}

Component 0 MoleculeName      {molecule}
            MoleculeDefinition TraPPE
            WidomProbability   1.0
            CreateNumberOfMolecules 0
""")

    
    qsub_file = output_dir / "run_raspa.qsub"
    with open(qsub_file, "w") as sh:
        sh.write(f"""#!/bin/sh
#PBS -r n
#PBS -q {RASPA_QSUB_QUEUE}
#PBS -l {RASPA_QSUB_RESOURCES}
#PBS -e {output_dir}/pbs.err
#PBS -o {output_dir}/pbs.out

cd $PBS_O_WORKDIR

echo "START $(date)" > START

{RASPA_SIMULATE_BIN} > output 2>&1
rc=$?

if [ $rc -eq 0 ]; then
  echo "DONE $(date)" > DONE
else
  echo "FAILED rc=$rc $(date)" > FAILED
fi

exit $rc
""")

    subprocess.run(["qas", str(qsub_file)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(qsub_file)


def _wait_for_all_jobs(jobs, check_interval=60, max_wait_hours=1800):

    start_t = time.time()
    outdirs = [Path(j["outdir"]) for j in jobs]

    while True:
        done = 0
        failed = 0
        started = 0
        pending = 0

        for d in outdirs:
            if (d / "DONE").exists():
                done += 1
            elif (d / "FAILED").exists():
                failed += 1
            else:
                pending += 1
                if (d / "START").exists():
                    started += 1

        elapsed_m = int((time.time() - start_t) / 60)

        
        queued = pending - started

        print(f"[WAIT] done={done} failed={failed} running={started} queued={queued} | elapsed={elapsed_m}m")

        if pending == 0:
            print("[WAIT] All jobs finished (DONE/FAILED detected).")
            return

        if max_wait_hours and (time.time() - start_t) > max_wait_hours * 3600:
            print("[WAIT] Max wait exceeded -> proceed to parse whatever finished.")
            return

        time.sleep(check_interval)


def _parse_raspa_result(cif_file, output_dir):
    cif_file = Path(cif_file)
    data_dir = Path(output_dir) / "Output/System_0"
    res_files = list(data_dir.glob("*.data"))

    for res_file in res_files:
        with open(res_file, "r") as f:
            text = f.read()
        henry_value, henry_error, units = _parse_henry_from_text(text)
        if henry_value is not None:
            return {
                "file": str(cif_file),
                "henry_constant": henry_value,
                "henry_error": henry_error,
                "units": units
            }
    return {"file": str(cif_file), "henry_constant": None}


def run_raspa_henry(cif_dir: str,
                    parse_dir: str,
                    okdir: str,
                    molecule: str = "hydrogen",
                    temperature: float = SCREENING_DEFAULT_HENRY_TEMPERATURE_K,
                    number_of_cycles: int = RASPA_HENRY_NUMBER_OF_CYCLES,
                    n_widom: int = RASPA_HENRY_WIDOM_INSERTIONS,
                    top_n: int = SCREENING_DEFAULT_TOP_N,
                    check_interval: int = 60):

    cif_dir = Path(cif_dir)
    parse_dir = Path(parse_dir)
    okdir = Path(okdir)

    cif_files = sorted(cif_dir.glob("*.cif"))
    if not cif_files:
        raise FileNotFoundError(f"No CIFs in {cif_dir}")

    parse_dir.mkdir(parents=True, exist_ok=True)
    okdir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for cif in cif_files:
        outdir = parse_dir / f"{cif.stem}_raspa"
        outdir.mkdir(parents=True, exist_ok=True)
        _submit_raspa_job(cif, molecule=molecule, output_dir=outdir,
                         number_of_cycles=number_of_cycles,
                         temperature=temperature, n_widom=n_widom)
        jobs.append({"file": str(cif), "outdir": str(outdir)})

    print(f"[RASPA] Submitted {len(jobs)} jobs. Waiting...")
    _wait_for_all_jobs(jobs, check_interval=check_interval)

    results = []
    for job in jobs:
        outdir = Path(job["outdir"])
        if not (outdir / "DONE").exists():
            continue

        res = _parse_raspa_result(job["file"], job["outdir"])
        if res.get("henry_constant") is not None:
            results.append(res)

    results.sort(key=lambda x: x["henry_constant"], reverse=True)
    N = len(results)
    keep_n = min(top_n, N)
    top_results = results[:keep_n]

    output_csv = okdir / "henry_results.csv"
    with open(output_csv, "w", newline="", encoding="utf-8") as fw:
        writer = csv.writer(fw)
        writer.writerow(["file", "henry_constant", "henry_error", "units"])
        for r in results:
            writer.writerow([r["file"], r["henry_constant"], r["henry_error"], r.get("units")])

    for r in top_results:
        src = Path(r["file"])
        shutil.copy2(src, okdir / src.name)

    print(f"[RASPA] Total={N}, Kept={len(top_results)}, CSV={output_csv}, CIFs->{okdir}")

    return [r["file"] for r in top_results]





def run_mofsimplify(mofs, condition: str):
    print(f"[MOFSimplify] Filtering {len(mofs)} MOFs with condition: {condition}")
    
    return ["MOF1"]