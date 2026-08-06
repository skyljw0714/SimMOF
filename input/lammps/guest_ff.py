
import os
import sys
from pathlib import Path
from core.llm_logging import log_llm_decision, set_llm_context

GUEST_FF_FAMILIES = ("TraPPE", "GAFF", "OPLSAA", "CHARMM")
DEFAULT_GUEST_FF_FAMILY = GUEST_FF_FAMILIES[0]


def llm_guest_ff_from_query(
    guest_name: str,
    query_text: str,
    rag_hints: str = "",
) -> str:
    import json
    from langchain.schema import SystemMessage, HumanMessage
    from config import LLM_DEFAULT, ask_user_confirmation

    system = (
        "You select a LAMMPS guest molecule force field family.\n"
        "Return ONLY JSON like {\"guest_ff\": \"TraPPE\"}.\n\n"
        "Decision procedure:\n"
        "1) If the user query explicitly requests an implemented FF family, use it.\n"
        "2) Otherwise, inspect the RAG_HINTS and choose the family best supported by "
        "retrieved literature for the target guest, MOF, and simulation task.\n"
        "3) If RAG_HINTS are absent or not actionable, choose the implemented family whose "
        "scope best matches the guest chemistry based on the descriptions below. Treat this "
        "as a conservative internal-library choice, not as a literature-supported claim.\n\n"
        "Implemented family descriptions:\n"
        "- TraPPE: Transferable Potentials for Phase Equilibria family of force fields, "
        "developed for transferable molecular models and phase-equilibrium/adsorption-related simulations.\n"
        "- GAFF: General AMBER Force Field for organic molecules, designed to be compatible "
        "with the AMBER force-field family.\n"
        "- OPLSAA: all-atom Optimized Potentials for Liquid Simulations force-field family, "
        "developed for conformational energetics and liquid-phase properties of organic molecules.\n"
        "- CHARMM: CHARMM General Force Field/CGenFF family for small organic and drug-like "
        "molecules, with CHARMM-compatible atom typing and parameter assignment.\n"
        f"Allowed families: {list(GUEST_FF_FAMILIES)}\n"
        "No extra text."
    )

    user_msg = (
        f"Guest molecule: {guest_name}\n"
        f"USER_QUERY: {query_text}\n\n"
        f"RAG_HINTS (from literature):\n{rag_hints if rag_hints else '(None)'}\n\n"
        f"Output ONLY JSON: {{\"guest_ff\": \"<one of {list(GUEST_FF_FAMILIES)}>\"}}"
    )

    try:
        set_llm_context("LAMMPSInputAgent", "guest_ff_selection")
        resp = LLM_DEFAULT.invoke([
            SystemMessage(content=system),
            HumanMessage(content=user_msg),
        ])
        raw = resp.content.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.splitlines()[1:-1]).strip()
        result = json.loads(raw).get("guest_ff", DEFAULT_GUEST_FF_FAMILY)
        if result not in GUEST_FF_FAMILIES:
            result = DEFAULT_GUEST_FF_FAMILY
        print(f"[guest_ff] LLM selected: {result}")

        def _reinvoke(instruction: str) -> str:
            revised_user = user_msg + f"\n\nUser instruction: {instruction}\nRevise your guest FF selection accordingly."
            set_llm_context("LAMMPSInputAgent", "guest_ff_selection_revision")
            r = LLM_DEFAULT.invoke([
                SystemMessage(content=system),
                HumanMessage(content=revised_user),
            ])
            raw2 = r.content.strip()
            if raw2.startswith("```"):
                raw2 = "\n".join(raw2.splitlines()[1:-1]).strip()
            try:
                r2 = json.loads(raw2).get("guest_ff", result)
                return r2 if r2 in GUEST_FF_FAMILIES else result
            except Exception:
                return result

        action, revised = ask_user_confirmation(
            "LAMMPSGuestFF",
            f"Proposed guest FF: {result}",
            reinvoke_fn=_reinvoke,
            required=True,
        )
        if action == "apply" and revised != f"Proposed guest FF: {result}":
            if revised in GUEST_FF_FAMILIES:
                print(f"[guest_ff] Updated per user instruction: {revised}")
                result = revised
        try:
            log_llm_decision("LAMMPSInputAgent", "guest_ff_selection",
                             {"guest": guest_name, "guest_ff": result})
        except Exception:
            pass
        return result
    except Exception as e:
        print(f"[guest_ff] LLM selection failed ({e}), defaulting to {DEFAULT_GUEST_FF_FAMILY}")
        return DEFAULT_GUEST_FF_FAMILY


def generate_guest_lt(
    molecule_name: str,
    guest_xyz: str,
    output_lt: str,
    ff_family: str,
    workdir: str = None,
    top_file: str = None,
    par_file: str = None,
    resi_name: str = None,
    rtf_file: str = None,
    prm_file: str = None,
) -> str:
    ff_family = ff_family.upper().replace("-", "").replace("_", "")
    if ff_family in ("TRAPPE", "TRAPPEAU", "TRAPPEUA"):
        ff_family = "TraPPE"
    elif ff_family == "GAFF2":
        ff_family = "GAFF"
    elif ff_family in ("OPLS", "OPLSAA", "OPLSAA"):
        ff_family = "OPLSAA"
    elif ff_family in ("CHARMM", "CGENFF"):
        ff_family = "CHARMM"

    os.makedirs(os.path.dirname(os.path.abspath(output_lt)) or ".", exist_ok=True)

    if ff_family == "TraPPE":
        if top_file is None or par_file is None:
            raise ValueError("top_file and par_file required for TraPPE")
        from input.lammps.input_trappe import generate_lt as trappe_generate_lt

        trappe_generate_lt(
            molecule=molecule_name,
            xyz_file=guest_xyz,
            top_file=top_file,
            par_file=par_file,
            output_file=output_lt,
        )

    elif ff_family == "GAFF":
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from LAMMPS.Forcefields.GAFF.gaff_lt_autogen import generate_lt as gaff_generate_lt

        gaff_generate_lt(
            molecule=molecule_name,
            xyz_file=guest_xyz,
            output_file=output_lt,
            workdir=workdir,
        )

    elif ff_family == "OPLSAA":
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from LAMMPS.Forcefields.OPLSAA.oplsaa import generate_lt as opls_generate_lt

        opls_generate_lt(
            molecule=molecule_name,
            xyz_file=guest_xyz,
            output_file=output_lt,
            workdir=workdir,
        )

    elif ff_family == "CHARMM":
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from LAMMPS.Forcefields.CHARMM.charmm_lt_autogen import generate_lt as charmm_generate_lt

        kwargs = {}
        if rtf_file:
            kwargs["rtf_file"] = rtf_file
        if prm_file:
            kwargs["prm_file"] = prm_file

        charmm_generate_lt(
            molecule=resi_name or molecule_name,
            xyz_file=guest_xyz,
            output_file=output_lt,
            **kwargs,
        )

    else:
        raise ValueError(f"Unknown guest FF family: {ff_family}. Choose from {GUEST_FF_FAMILIES}")

    return output_lt
