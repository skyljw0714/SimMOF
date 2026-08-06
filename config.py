from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, List

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
LEGACY_PROJECT_ROOT = Path("/path/to/SimMOF")
LEGACY_ZEOPP_ROOT = Path("/path/to/SimMOF/Zeopp")
LEGACY_RASPA_ROOT = Path("/path/to/RASPA/Research/simulations")
LEGACY_CSD_MODIFIED_ROOT = Path("/path/to/SimMOF/CSD-modified/CSD-modified")
DEFAULT_VASP_POTENTIAL_DIR = Path("/path/to/vasp/PseudoPotential/potpaw_PBE.54")
DEFAULT_VASP_EXECUTABLE = "/path/to/vasp/vasp_std"
DEFAULT_LAMMPS_EXECUTABLE = "/path/to/lammps/bin/lmp_mpi"
DEFAULT_MOLTEMPLATE_SCRIPT = Path("/path/to/moltemplate/moltemplate/ltemplify.py")
DEFAULT_MOLTEMPLATE_SH = Path("/path/to/moltemplate/moltemplate/scripts/moltemplate.sh")
DEFAULT_PACKMOL_EXECUTABLE = Path("/path/to/packmol/packmol")
DEFAULT_EQEQ_DIR = Path("/path/to/EQeq")

for env_file in (PROJECT_ROOT / ".env", PROJECT_ROOT / "config.env"):
    if env_file.exists():
        load_dotenv(env_file, override=False)


def _first_existing(*candidates: Path | str | None) -> Path:
    normalized: List[Path] = []
    for candidate in candidates:
        if candidate is None:
            continue
        normalized.append(Path(candidate).expanduser())
    for path in normalized:
        if path.exists():
            return path
    if normalized:
        return normalized[0]
    return PROJECT_ROOT


def _path_from_env(name: str, default: Path | str) -> Path:
    raw = os.getenv(name)
    if raw:
        return Path(raw).expanduser()
    return Path(default).expanduser()


def _require_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Export it in the environment or provide it via .env/config.env."
        )
    return api_key


class _LazyChatModel:
    def __init__(self, model: str):
        self.model = model
        self._instance = None

    def _get_instance(self):
        if self._instance is None:
            from langchain_openai import ChatOpenAI

            self._instance = ChatOpenAI(model=self.model, api_key=_require_openai_api_key(), temperature=1)
        return self._instance

    def invoke(self, messages, **kwargs):
        resp = self._get_instance().invoke(messages, **kwargs)
        try:
            from core.llm_logging import log_llm_call
            usage = getattr(resp, "usage_metadata", None) or {}
            log_llm_call(
                model=self.model,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            )
        except Exception:
            pass
        return resp

    def __getattr__(self, name: str):
        return getattr(self._get_instance(), name)

    def __repr__(self) -> str:
        return f"_LazyChatModel(model={self.model!r})"


TOKENIZERS_PARALLELISM = os.getenv("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TOKENIZERS_PARALLELISM", TOKENIZERS_PARALLELISM)

OPENAI_MODEL_DEFAULT = os.getenv("SIMMOF_OPENAI_MODEL_DEFAULT", "gpt-5.5")


@lru_cache(maxsize=1)
def get_openai_client() -> Any:
    from openai import OpenAI

    return OpenAI(api_key=_require_openai_api_key())


def _make_chat_model(model: str) -> _LazyChatModel:
    return _LazyChatModel(model)


LLM_DEFAULT = _make_chat_model(OPENAI_MODEL_DEFAULT)
LLM_STRICT = LLM_DEFAULT

AGENT_LLM_MAP = {
    "ZeoppInputAgent": LLM_STRICT,
    "LAMMPSErrorAgent": LLM_STRICT,
    "ResponseAgent": LLM_STRICT,
    "QueryAgent": LLM_STRICT,
    "LAMMPSInputAgent": LLM_STRICT,
    "WorkingAgent": LLM_STRICT,
    "ScreeningAgent": LLM_STRICT,
}

WORKING_DIR = _path_from_env("SIMMOF_WORKING_DIR", PROJECT_ROOT / "working_dir")
SCREENING_WORK_ROOT = _path_from_env("SIMMOF_SCREENING_WORK_ROOT", WORKING_DIR / "screening")
SCREENING_CIF_ROOT = _path_from_env("SIMMOF_SCREENING_CIF_ROOT", SCREENING_WORK_ROOT / "cifs")
BATCH_WORK_ROOT = _path_from_env("SIMMOF_BATCH_WORK_ROOT", WORKING_DIR / "batch")

ZEO_DIR = _path_from_env("SIMMOF_ZEO_DIR", _first_existing(PROJECT_ROOT / "Zeopp", LEGACY_ZEOPP_ROOT))
ZEOPP_BIN = _path_from_env("SIMMOF_ZEOPP_BIN", ZEO_DIR / "network")

RASPA_ROOT = _path_from_env("SIMMOF_RASPA_DIR", _first_existing(LEGACY_RASPA_ROOT, PROJECT_ROOT / "RASPA"))
RASPA_SIMULATE_BIN = _path_from_env("SIMMOF_RASPA_SIMULATE_BIN", RASPA_ROOT / "bin" / "simulate")

LAMMPS_FORCEFIELD_ROOT = PROJECT_ROOT / "LAMMPS" / "Forcefields"
TRAPPE_DIR = _path_from_env("SIMMOF_TRAPPE_DIR", LAMMPS_FORCEFIELD_ROOT / "TraPPE")
TRAPPE_TOP_FILE = _path_from_env("SIMMOF_TRAPPE_TOP_FILE", TRAPPE_DIR / "top_trappe.inp")
TRAPPE_PAR_FILE = _path_from_env("SIMMOF_TRAPPE_PAR_FILE", TRAPPE_DIR / "par_trappe.inp")
TRAPPE_DICT_FILE = _path_from_env("SIMMOF_TRAPPE_DICT_FILE", TRAPPE_DIR / "trappe_dict.json")

PACKMOL_EXECUTABLE = _path_from_env(
    "SIMMOF_PACKMOL_EXECUTABLE",
    _first_existing(DEFAULT_PACKMOL_EXECUTABLE, PROJECT_ROOT / "packmol"),
)
PACKMOL_OUTPUT_DIR = _path_from_env("SIMMOF_PACKMOL_OUTPUT_DIR", WORKING_DIR / "packmol")

EQEQ_DIR = _path_from_env(
    "SIMMOF_EQEQ_DIR",
    _first_existing(PROJECT_ROOT / "charge" / "EQeq", DEFAULT_EQEQ_DIR),
)

PACMAN_AVAILABLE: bool = True
try:
    from PACMANCharge.pmcharge import predict as _pacman_predict  # noqa: F401
except ImportError:
    PACMAN_AVAILABLE = False

DEFAULT_CHARGEMOL_BIN = Path("/path/to/chargemol/chargemol")
DEFAULT_CHARGEMOL_ATOMIC_DENSITIES_DIR = Path("/path/to/chargemol/atomic_densities")

CHARGEMOL_BIN = _path_from_env("SIMMOF_CHARGEMOL_BIN", DEFAULT_CHARGEMOL_BIN)
CHARGEMOL_ATOMIC_DENSITIES_DIR = _path_from_env(
    "SIMMOF_CHARGEMOL_ATOMIC_DENSITIES_DIR", DEFAULT_CHARGEMOL_ATOMIC_DENSITIES_DIR
)

LAMMPS_EXECUTABLE = os.getenv("SIMMOF_LAMMPS_EXECUTABLE", DEFAULT_LAMMPS_EXECUTABLE)
LAMMPS_INTERFACE_EXECUTABLE = os.getenv(
    "SIMMOF_LAMMPS_INTERFACE_BIN", "/path/to/anaconda3/envs/simmof/bin/lammps-interface"
)
LAMMPS_MOLTEMPLATE_SCRIPT = _path_from_env("SIMMOF_MOLTEMPLATE_SCRIPT", DEFAULT_MOLTEMPLATE_SCRIPT)
LAMMPS_MOLTEMPLATE_SH = _path_from_env("SIMMOF_MOLTEMPLATE_SH", DEFAULT_MOLTEMPLATE_SH)
VASP_POTENTIAL_DIR_PATH = _path_from_env("SIMMOF_VASP_POTENTIAL_DIR", DEFAULT_VASP_POTENTIAL_DIR)
VASP_EXECUTABLE = os.getenv("SIMMOF_VASP_EXECUTABLE", DEFAULT_VASP_EXECUTABLE)

PORMAKE_CONDA_ENV_NAME = os.getenv("SIMMOF_PORMAKE_CONDA_ENV_NAME", "pormake")
PORMAKE_CONDA_ENV_PREFIX = Path(
    os.getenv(
        "SIMMOF_PORMAKE_CONDA_ENV_PREFIX",
        f"/path/to/anaconda3/envs/{os.getenv('SIMMOF_PORMAKE_CONDA_ENV_NAME', 'pormake')}",
    )
)
PORMAKE_PYTHON = os.getenv(
    "SIMMOF_PORMAKE_PYTHON",
    str(PORMAKE_CONDA_ENV_PREFIX / "bin" / "python"),
)

CSD_API_CONDA_ENV_NAME = os.getenv("SIMMOF_CSD_API_CONDA_ENV_NAME", "csd_api")
CSD_API_CONDA_ENV_PREFIX = Path(
    os.getenv(
        "SIMMOF_CSD_API_CONDA_ENV_PREFIX",
        f"/path/to/anaconda3/envs/{CSD_API_CONDA_ENV_NAME}",
    )
)
CSD_API_PYTHON = os.getenv(
    "SIMMOF_CSD_API_PYTHON",
    str(CSD_API_CONDA_ENV_PREFIX / "bin" / "python"),
)

MOFSIMPLIFY_CONDA_ENV_NAME = os.getenv("SIMMOF_MOFSIMPLIFY_CONDA_ENV_NAME", "mofsimplify")
MOFSIMPLIFY_CONDA_ENV_PREFIX = Path(
    os.getenv(
        "SIMMOF_MOFSIMPLIFY_CONDA_ENV_PREFIX",
        f"/path/to/anaconda3/envs/{MOFSIMPLIFY_CONDA_ENV_NAME}",
    )
)
MOFSIMPLIFY_PYTHON = os.getenv(
    "SIMMOF_MOFSIMPLIFY_PYTHON",
    str(MOFSIMPLIFY_CONDA_ENV_PREFIX / "bin" / "python"),
)

MOFID_CONDA_ENV_NAME = os.getenv("SIMMOF_MOFID_CONDA_ENV_NAME", "mofid")
MOFID_CONDA_ENV_PREFIX = Path(
    os.getenv(
        "SIMMOF_MOFID_CONDA_ENV_PREFIX",
        f"/path/to/anaconda3/envs/{MOFID_CONDA_ENV_NAME}",
    )
)
MOFID_PYTHON = os.getenv(
    "SIMMOF_MOFID_PYTHON",
    str(MOFID_CONDA_ENV_PREFIX / "bin" / "python"),
)

RAG_STORE_DIR = _path_from_env(
    "SIMMOF_RAG_STORE_DIR",
    PROJECT_ROOT / "rag" / "vector_db_fulltext" / "sentence-transformers_all-MiniLM-L6-v2",
)
RAG_SI_STORE_DIR = _path_from_env(
    "SIMMOF_RAG_SI_STORE_DIR",
    PROJECT_ROOT / "rag" / "vector_db_SI" / "sentence-transformers_all-MiniLM-L6-v2",
)
RAG_CORPUS_DIR = _path_from_env("SIMMOF_RAG_CORPUS_DIR", PROJECT_ROOT / "rag" / "parsed_fulltext")
RAG_EMBED_MODEL_NAME = os.getenv("SIMMOF_RAG_EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

CSD_MODIFIED_ROOT = _path_from_env(
    "SIMMOF_CSD_MODIFIED_ROOT",
    _first_existing(PROJECT_ROOT / "CSD-modified" / "CSD-modified", LEGACY_PROJECT_ROOT / "CSD-modified" / "CSD-modified", LEGACY_CSD_MODIFIED_ROOT),
)
COREMOF_DATA_CSV = _path_from_env(
    "SIMMOF_COREMOF_DATA_CSV",
    CSD_MODIFIED_ROOT / "CR_data_CSD_modified_20250227.csv",
)
COREMOF_PHASE_DIRS = {
    "ASR": _path_from_env("SIMMOF_COREMOF_ASR_DIR", CSD_MODIFIED_ROOT / "cifs" / "CR" / "ASR"),
    "FSR": _path_from_env("SIMMOF_COREMOF_FSR_DIR", CSD_MODIFIED_ROOT / "cifs" / "CR" / "FSR"),
    "Ion": _path_from_env("SIMMOF_COREMOF_ION_DIR", CSD_MODIFIED_ROOT / "cifs" / "CR" / "Ion"),
}


def get_pormake_python_command() -> List[str]:
    if PORMAKE_PYTHON:
        return [PORMAKE_PYTHON]
    command = ["conda", "run"]
    if PORMAKE_CONDA_ENV_PREFIX:
        command.extend(["-p", str(PORMAKE_CONDA_ENV_PREFIX)])
    else:
        command.extend(["-n", PORMAKE_CONDA_ENV_NAME])
    command.append("python")
    return command


def get_csd_api_python_command() -> List[str]:
    if CSD_API_PYTHON:
        return [CSD_API_PYTHON]
    command = ["conda", "run"]
    if CSD_API_CONDA_ENV_PREFIX:
        command.extend(["-p", str(CSD_API_CONDA_ENV_PREFIX)])
    else:
        command.extend(["-n", CSD_API_CONDA_ENV_NAME])
    command.append("python")
    return command


working_dir = str(WORKING_DIR)
zeo_dir = str(ZEO_DIR)
RASPA_DIR = str(RASPA_ROOT)

INTERACTION_MODE: str = os.getenv("SIMMOF_INTERACTION_MODE", "autonomous").lower()


def ask_user_confirmation(label: str, proposed_text: str, reinvoke_fn=None,
                          required: bool = False):
    if INTERACTION_MODE != "interactive":
        return "apply", proposed_text

    user_in = input("\n[y] Apply / [n] Skip / [or type your instruction]: ").strip()
    if user_in.lower() == "y":
        return "apply", proposed_text
    if user_in.lower() in ("n", ""):
        if not required:
            print(f"[{label}] Skipped by user.")
            return "skip", ""
        user_in = input(f"[{label}] Cannot skip — please type your instruction: ").strip()
        if not user_in:
            print(f"[{label}] No instruction given. Using original proposal.")
            return "apply", proposed_text
    if reinvoke_fn is None:
        print(f"[{label}] Custom instruction not supported here. Skipped.")
        return "skip", ""
    print(f"[{label}] Re-invoking with your instruction...")
    revised = reinvoke_fn(user_in)
    print(f"\n[{label}] Revised:\n{revised}")
    return "apply", revised
