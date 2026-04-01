import json
import os
import shutil
import subprocess
import textwrap
import time
from pathlib import Path

LAMMPS_INTERFACE_COMMAND = "lammps-interface"
LAMMPS_MOLTEMPLATE_COMMAND = "moltemplate.sh"
LAMMPS_PACKMOL_TOLERANCE = 2.5
LAMMPS_SUPERCELL_CUTOFF = 12.5
LAMMPS_CHARGED_PAIR_CUTOFF = 10.0
LAMMPS_KSPACE_ACCURACY = "1.0e-4"


def _run_command(cmd: str, cwd: str, shell: bool = True):
    print(f"\n>>> Running (cwd={cwd}): {cmd}")
    result = subprocess.run(cmd, shell=shell, capture_output=True, text=True, cwd=cwd)
    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)


def _pick_snippet(simulation_input: dict, software: str) -> str:
    if not simulation_input:
        return ""
    for s in (simulation_input.get("snippets") or []):
        if (s.get("software") == software) and (s.get("text") or "").strip():
            return s["text"].strip()
    return ""


def _read_xyz_rows(xyz_path: str):
    lines = Path(xyz_path).read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"Invalid xyz file: {xyz_path}")

    n = int(lines[0].strip())
    comment = lines[1] if len(lines) > 1 else ""

    rows = []
    for line in lines[2:2 + n]:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Bad xyz atom line in {xyz_path}: {line}")
        sym = parts[0]
        x, y, z = map(float, parts[1:4])
        rows.append((sym, x, y, z))

    if len(rows) != n:
        raise ValueError(
            f"XYZ atom count mismatch in {xyz_path}: header={n}, parsed={len(rows)}"
        )

    return comment, rows


def _write_xyz_rows(xyz_path: str, comment: str, rows):
    out = [str(len(rows)), comment]
    for sym, x, y, z in rows:
        out.append(f"{sym:4s} {x: .8f} {y: .8f} {z: .8f}")
    Path(xyz_path).write_text("\n".join(out) + "\n")


def _expand_packed_xyz_for_linear_com(packmol_xyz: str, output_xyz: str, mof_atom_count: int, num_guest: int, center_label: str = "COM"):
    comment, rows = _read_xyz_rows(packmol_xyz)

    guest_atoms_per_mol_in_packmol = 2
    expected_total = mof_atom_count + num_guest * guest_atoms_per_mol_in_packmol

    if len(rows) != expected_total:
        raise ValueError(
            f"Packed xyz atom count mismatch: got {len(rows)}, "
            f"expected {expected_total} = mof({mof_atom_count}) + num_guest({num_guest})*2"
        )

    mof_rows = rows[:mof_atom_count]
    guest_rows = rows[mof_atom_count:]

    expanded_guest_rows = []
    for i in range(num_guest):
        sym1, x1, y1, z1 = guest_rows[2 * i]
        sym2, x2, y2, z2 = guest_rows[2 * i + 1]

        xm = 0.5 * (x1 + x2)
        ym = 0.5 * (y1 + y2)
        zm = 0.5 * (z1 + z2)

        expanded_guest_rows.append((sym1, x1, y1, z1))
        expanded_guest_rows.append((center_label, xm, ym, zm))
        expanded_guest_rows.append((sym2, x2, y2, z2))

    all_rows = mof_rows + expanded_guest_rows
    _write_xyz_rows(output_xyz, comment, all_rows)

    print(
        f"[linear-expand] mof_atom_count={mof_atom_count}, "
        f"num_guest={num_guest}, total_in={len(rows)}, total_out={len(all_rows)}"
    )


def cif_has_atom_site_charge(cif_path: str) -> bool:
    text = Path(cif_path).read_text(errors="ignore").lower()
    return "_atom_site_charge" in text


def generate_lammps_inputs(
    working_dir: str,
    mof_name: str,
    guest_name: str,
    property_name: str,
    query_text: str = "",
    num_guest: int = 5,
    job_name: str = "",
    simulation_input: dict = None,
):
    from ase.io import read, write

    from config import AUTO_RESEARCH_PYTHON, LAMMPS_MOLTEMPLATE_SCRIPT, TRAPPE_PAR_FILE, TRAPPE_TOP_FILE
    from packmol.run_packmol import run_packmol_from_cif

    from .input_gen import (
        clean_cif_with_ase,
        compute_supercell_size,
        deduplicate_system_in_init,
        detect_charged_system,
        extract_hybrid_style_keys,
        generate_system_in,
        llm_option_from_query,
        patch_pair_kspace_after_read_data,
        update_settings_with_style,
        write_system_lt,
    )
    from .input_trappe import generate_lt
    from .parser import make_group_commands, match_trappe_abbreviation

    working_dir = str(working_dir)
    os.makedirs(working_dir, exist_ok=True)

    print(f"[LAMMPS pipeline] working_dir = {working_dir}")
    print(f"[LAMMPS pipeline] mof = {mof_name}, guest = {guest_name}, property = {property_name}")

    prop = property_name.lower()
    is_te = ("thermal_expansion" in prop) or ("thermal expansion" in prop)
    is_young = ("young" in prop) or ("elastic" in prop) or ("modulus" in prop)
    is_mof_only = is_te or is_young

    unit_cif_file = str(Path(working_dir) / f"{mof_name}.cif")
    supercell_cif_file = None

    if not is_mof_only:
        original_guest_xyz = os.path.join(working_dir, f"{guest_name}.xyz")

    print("[CIF] Checking CIF...")
    if cif_has_atom_site_charge(unit_cif_file):
        print("[CIF] Detected _atom_site_charge; skipping ASE cleaning to preserve DDEC charges.")
    else:
        print("[CIF] Cleaning CIF with ASE...")
        clean_cif_with_ase(unit_cif_file, unit_cif_file)

    if not is_mof_only:
        molecule_name = match_trappe_abbreviation(guest_name)
        print("[TRAPPE] mapped guest:", guest_name, "→", molecule_name)

        TOP_TRAPPE = str(TRAPPE_TOP_FILE)
        from input.lammps.trappe_ua_convert import convert_allatom_xyz_to_trappe_ua_xyz, needs_ua_conversion

        guest_xyz_for_packmol = os.path.join(working_dir, f"{molecule_name}.xyz")
        guest_xyz_for_lt = os.path.join(working_dir, f"{molecule_name}.xyz")
        original_guest_xyz = os.path.join(working_dir, f"{guest_name}.xyz")

        if os.path.abspath(original_guest_xyz) == os.path.abspath(guest_xyz_for_packmol):
            print("[TRAPPE] guest xyz already TRAPPE-compatible filename, skip.")
        else:
            if needs_ua_conversion(molecule_name):
                xyz_elem = os.path.join(working_dir, f"{molecule_name}.xyz")
                xyz_site = os.path.join(working_dir, f"{molecule_name}.site.xyz")

                convert_allatom_xyz_to_trappe_ua_xyz(
                    original_xyz=original_guest_xyz,
                    top_file=TOP_TRAPPE,
                    resi_name=molecule_name,
                    out_xyz=xyz_elem,
                    out_xyz_site=xyz_site,
                    cc_cutoff=1.85,
                )

                guest_xyz_for_packmol = xyz_elem
                guest_xyz_for_lt = xyz_site

                print(f"[TRAPPE-UA] wrote packmol xyz: {xyz_elem}")
                print(f"[TRAPPE-UA] wrote LT xyz    : {xyz_site}")
            else:
                shutil.copyfile(original_guest_xyz, guest_xyz_for_packmol)
                print(f"[TRAPPE] copied {original_guest_xyz} → {guest_xyz_for_packmol}")

        print("[TraPPE] Generating guest LT file...")
        generate_lt(
            molecule=molecule_name,
            xyz_file=guest_xyz_for_lt,
            top_file=str(TRAPPE_TOP_FILE),
            par_file=str(TRAPPE_PAR_FILE),
            output_file=str(Path(working_dir) / f"{molecule_name}.lt"),
        )

    print("[lammps-interface] inferring options from query...")
    lammps_interface_option = llm_option_from_query(query_text)
    print("[lammps-interface options]", lammps_interface_option)
    print(f"[lammps-interface] unit-cell CIF = {unit_cif_file}")

    _run_command(f"{LAMMPS_INTERFACE_COMMAND} {lammps_interface_option} {unit_cif_file}", cwd=working_dir)

    data_file = f"data.{mof_name}"
    in_file = f"in.{mof_name}"
    lt_file = f"{mof_name}.lt"

    _run_command(
        f"python3 {LAMMPS_MOLTEMPLATE_SCRIPT} -name mof {data_file} {in_file} > {lt_file}",
        cwd=working_dir,
    )

    print("[Supercell] Computing supercell size...")
    supercell = compute_supercell_size(unit_cif_file, cutoff=LAMMPS_SUPERCELL_CUTOFF)
    atoms = read(unit_cif_file)
    atoms_super = atoms.repeat(supercell)

    supercell_cif_file = str(Path(working_dir) / f"{mof_name}_supercell.cif")
    write(supercell_cif_file, atoms_super)
    print(f"[Supercell] wrote supercell CIF → {supercell_cif_file}")

    system_xyz = os.path.join(working_dir, "system.xyz")

    if not is_mof_only:
        print("[Packmol] Running packmol...")
        molecule_num = num_guest
        run_packmol_from_cif(
            cif_file=supercell_cif_file,
            guest_xyz=guest_xyz_for_packmol,
            number_of_guest=molecule_num,
            number_of_system=1,
            tolerance=LAMMPS_PACKMOL_TOLERANCE,
            output_dir=os.path.join(working_dir, "packmol"),
        )

        print("[system.lt] Writing system.lt...")
        write_system_lt(
            cif_path=supercell_cif_file,
            mof_lt_name=mof_name,
            guest_lt_name=molecule_name,
            guest_count=molecule_num,
            output_file=str(Path(working_dir) / "system.lt"),
        )

        supercell_stem = Path(supercell_cif_file).stem
        packmol_xyz = os.path.join(
            working_dir,
            "packmol",
            f"{supercell_stem}_{molecule_name}",
            f"{supercell_stem}_{molecule_name}_1.xyz",
        )

        print(f"[Packmol] expecting packed xyz: {packmol_xyz}")

        if molecule_name in ["N2", "O2", "CO"]:
            mof_atom_count = len(read(supercell_cif_file))
            _expand_packed_xyz_for_linear_com(
                packmol_xyz=packmol_xyz,
                output_xyz=system_xyz,
                mof_atom_count=mof_atom_count,
                num_guest=molecule_num,
                center_label="COM",
            )
            print(f"[Packmol] Expanded {molecule_name} packed xyz with COM sites → {system_xyz}")
        else:
            shutil.copyfile(packmol_xyz, system_xyz)
            print(f"[Packmol] Copied {packmol_xyz} → {system_xyz}")
    else:
        print("[MOF-only] MOF-only mode: skipping guest/TRAPPE/Packmol.")
        print("[system.lt] Writing MOF-only system.lt...")

        write_system_lt(
            cif_path=supercell_cif_file,
            mof_lt_name=mof_name,
            guest_lt_name=None,
            guest_count=0,
            output_file=str(Path(working_dir) / "system.lt"),
        )

        print("[MOF-only] Writing system.xyz from supercell CIF (MOF only)...")
        atoms = read(supercell_cif_file)
        write(system_xyz, atoms)
        print(f"[MOF-only] Wrote {system_xyz}")

    print("[Moltemplate] Building system from system.lt + system.xyz ...")
    _run_command(f"{LAMMPS_MOLTEMPLATE_COMMAND} -xyz {system_xyz} system.lt", cwd=working_dir)

    init_path = str(Path(working_dir) / "system.in.init")
    settings_path = str(Path(working_dir) / "system.in.settings")

    deduplicate_system_in_init(init_path, init_path)

    hybrid_keys = extract_hybrid_style_keys(init_path)

    if not is_mof_only:
        update_settings_with_style(
            lt_path=str(Path(working_dir) / f"{molecule_name}.lt"),
            settings_path=settings_path,
            hybrid_keys=hybrid_keys,
            output_path=settings_path,
        )

    update_settings_with_style(
        lt_path=str(Path(working_dir) / f"{mof_name}.lt"),
        settings_path=settings_path,
        hybrid_keys=hybrid_keys,
        output_path=settings_path,
    )

    run_example = Path(working_dir) / "run.in.EXAMPLE"
    system_in = Path(working_dir) / "system.in"

    if run_example.exists():
        run_example.replace(system_in)
        print("[Run] 'run.in.EXAMPLE' renamed to 'system.in'")
    else:
        print("[Run] 'run.in.EXAMPLE' not found")

    charged = detect_charged_system(Path(working_dir) / "system.data")
    patch_pair_kspace_after_read_data(
        Path(working_dir) / "system.in",
        charged,
        cutoff=LAMMPS_CHARGED_PAIR_CUTOFF,
        acc=LAMMPS_KSPACE_ACCURACY,
    )

    print("[Group] Generating group commands...")

    mof_lt_path = str(Path(working_dir) / f"{mof_name}.lt")

    if is_mof_only:
        group_definitions = ""
    else:
        guest_lt_path = str(Path(working_dir) / f"{molecule_name}.lt")
        mof_atoms, guest_atoms, group_cmds = make_group_commands(mof_lt_path, guest_lt_path)
        group_definitions = "\n".join(group_cmds)

    print("[Run Section] Generating run commands into system.in ...")

    if query_text:
        simulation_description = f"[JOB_NAME={job_name}] {query_text}"
    else:
        if is_te:
            simulation_description = f"Simulate thermal expansion of {mof_name} (MOF-only, NPT temperature scan)."
        elif is_young:
            simulation_description = f"Compute Young's modulus of {mof_name} (MOF-only)."
        else:
            simulation_description = f"Compute {property_name} of {guest_name} in {mof_name}"

    mode = "standard"
    example_text = _pick_snippet(simulation_input, "LAMMPS")
    if example_text:
        mode = "reproduce"

    rag_summaries = ""

    if mode != "reproduce":
        try:
            project_root = str(Path(__file__).resolve().parents[2])
            rag_ctx = {
                "job_name": job_name,
                "mof": mof_name,
                "guest": guest_name or "",
                "property": property_name,
                "query_text": query_text or "",
            }

            rag_script = f"""
import sys, json
sys.path.append({json.dumps(project_root)})
from rag.agent import RagAgent

agent = RagAgent(agent_name="RagAgent")
out = agent.run_for_system_in({json.dumps(rag_ctx, ensure_ascii=False)}, top_files=5)
print(json.dumps({{"rag_summaries": out.get("rag_summaries","")}}, ensure_ascii=False))
""".strip()

            auto_py = str(AUTO_RESEARCH_PYTHON)
            result = subprocess.run(
                [auto_py, "-c", rag_script],
                capture_output=True,
                text=True,
                cwd=working_dir,
            )

            if result.returncode == 0 and result.stdout.strip():
                obj = json.loads(result.stdout.strip())
                rag_summaries = (obj.get("rag_summaries") or "").strip()
                if rag_summaries:
                    print("[RAG] system.in hints enabled (auto-research)")
                else:
                    print("[RAG] no relevant hints found (auto-research)")
            else:
                print("[RAG] auto-research run failed")
                if result.stdout:
                    print("STDOUT:\n", result.stdout)
                if result.stderr:
                    print("STDERR:\n", result.stderr)
        except Exception as e:
            print(f"[RAG] disabled due to error: {e}")

    generate_system_in(
        property=property_name,
        simulation_description=simulation_description,
        group_definition=group_definitions,
        output_file=str(Path(working_dir) / "system.in"),
        example_text=example_text,
        rag_summaries=rag_summaries,
    )
