from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, List

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
LEGACY_PROJECT_ROOT = Path("/home/users/skyljw0714/SimMOF")
LEGACY_ZEOPP_ROOT = Path("/home/users/skyljw0714/SimMOF/Zeopp")
LEGACY_RASPA_ROOT = Path("/home/users/skyljw0714/RASPA/Research/simulations")
LEGACY_CSD_MODIFIED_ROOT = Path("/home/users/skyljw0714/SimMOF/CSD-modified/CSD-modified")
DEFAULT_VASP_POTENTIAL_DIR = Path("/opt/vasp/PseudoPotential/potpaw_PBE.54")
DEFAULT_VASP_EXECUTABLE = "/opt/vasp/5.4.1/vasp_std"
DEFAULT_LAMMPS_EXECUTABLE = "/opt/lammps/200303/bin/lmp_mpi"
DEFAULT_MOLTEMPLATE_SCRIPT = Path("/home/users/skyljw0714/bin/moltemplate/moltemplate/ltemplify.py")
DEFAULT_PACKMOL_EXECUTABLE = Path("/home/users/skyljw0714/packmol-21.1.0/packmol")
DEFAULT_AUTO_RESEARCH_PYTHON = Path("/home/users/skyljw0714/anaconda3/envs/auto-research/bin/python")

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

            self._instance = ChatOpenAI(model=self.model, api_key=_require_openai_api_key())
        return self._instance

    def __getattr__(self, name: str):
        return getattr(self._get_instance(), name)

    def __repr__(self) -> str:
        return f"_LazyChatModel(model={self.model!r})"


TOKENIZERS_PARALLELISM = os.getenv("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TOKENIZERS_PARALLELISM", TOKENIZERS_PARALLELISM)

OPENAI_MODEL_STRICT = os.getenv("SIMMOF_OPENAI_MODEL_STRICT", "gpt-5")
OPENAI_MODEL_DEFAULT = os.getenv("SIMMOF_OPENAI_MODEL_DEFAULT", "gpt-5")
OPENAI_MODEL_FAST = os.getenv("SIMMOF_OPENAI_MODEL_FAST", "gpt-5-mini")
OPENAI_MODEL_PARSER = os.getenv("SIMMOF_OPENAI_MODEL_PARSER", "gpt-4.1-mini")
OPENAI_MODEL_LAMMPS = os.getenv("SIMMOF_OPENAI_MODEL_LAMMPS", OPENAI_MODEL_DEFAULT)


@lru_cache(maxsize=1)
def get_openai_client() -> Any:
    from openai import OpenAI

    return OpenAI(api_key=_require_openai_api_key())


def _make_chat_model(model: str) -> _LazyChatModel:
    return _LazyChatModel(model)


LLM_STRICT = _make_chat_model(OPENAI_MODEL_STRICT)
LLM_DEFAULT = _make_chat_model(OPENAI_MODEL_DEFAULT)
LLM_FAST = _make_chat_model(OPENAI_MODEL_FAST)

AGENT_LLM_MAP = {
    "ZeoppInputAgent": LLM_STRICT,
    "LAMMPSErrorAgent": LLM_STRICT,
    "ResponseAgent": LLM_STRICT,
    "QueryAgent": LLM_STRICT,
    "LAMMPSInputAgent": LLM_STRICT,
    "WorkingAent": LLM_STRICT,
    "ScreeningAgent": LLM_STRICT,
}

WORKING_DIR = _path_from_env(
    "SIMMOF_WORKING_DIR",
    _first_existing(PROJECT_ROOT / "working_dir", LEGACY_PROJECT_ROOT / "working_dir", PROJECT_ROOT / "working"),
)
SCREENING_WORK_ROOT = _path_from_env("SIMMOF_SCREENING_WORK_ROOT", WORKING_DIR / "screening")
SCREENING_CIF_ROOT = _path_from_env("SIMMOF_SCREENING_CIF_ROOT", SCREENING_WORK_ROOT / "cifs")

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

LAMMPS_EXECUTABLE = os.getenv("SIMMOF_LAMMPS_EXECUTABLE", DEFAULT_LAMMPS_EXECUTABLE)
LAMMPS_MOLTEMPLATE_SCRIPT = _path_from_env("SIMMOF_MOLTEMPLATE_SCRIPT", DEFAULT_MOLTEMPLATE_SCRIPT)
VASP_POTENTIAL_DIR_PATH = _path_from_env("SIMMOF_VASP_POTENTIAL_DIR", DEFAULT_VASP_POTENTIAL_DIR)
VASP_EXECUTABLE = os.getenv("SIMMOF_VASP_EXECUTABLE", DEFAULT_VASP_EXECUTABLE)

CSD_API_CONDA_ENV_NAME = os.getenv("SIMMOF_CSD_API_CONDA_ENV_NAME", "csd_api")
CSD_API_CONDA_ENV_PREFIX = Path(
    os.getenv(
        "SIMMOF_CSD_API_CONDA_ENV_PREFIX",
        f"/home/users/skyljw0714/anaconda3/envs/{CSD_API_CONDA_ENV_NAME}",
    )
)
CSD_API_PYTHON = os.getenv(
    "SIMMOF_CSD_API_PYTHON",
    str(CSD_API_CONDA_ENV_PREFIX / "bin" / "python"),
)
AUTO_RESEARCH_PYTHON = _path_from_env(
    "SIMMOF_AUTO_RESEARCH_PYTHON",
    _first_existing(DEFAULT_AUTO_RESEARCH_PYTHON, Path(os.sys.executable)),
)

RAG_STORE_DIR = _path_from_env(
    "SIMMOF_RAG_STORE_DIR",
    PROJECT_ROOT / "rag" / "vector_db_fulltext" / "sentence-transformers_all-MiniLM-L6-v2",
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

REMOTE_MLIP_GPU_HOST = os.getenv("SIMMOF_REMOTE_MLIP_GPU_HOST", "143.248.130.69")
REMOTE_MLIP_GPU_USER = os.getenv("SIMMOF_REMOTE_MLIP_GPU_USER", "taeun8991")
REMOTE_MLIP_GPU_PORT = os.getenv("SIMMOF_REMOTE_MLIP_GPU_PORT", "7722")
REMOTE_MLIP_REMOTE_DIR = _path_from_env("SIMMOF_REMOTE_MLIP_REMOTE_DIR", "/home/users/taeun8991/mlip_test")
REMOTE_MLIP_LOCAL_OUTPUT_DIR = _path_from_env("SIMMOF_REMOTE_MLIP_LOCAL_OUTPUT_DIR", SCREENING_WORK_ROOT)
REMOTE_MLIP_CONDA_INIT = os.getenv(
    "SIMMOF_REMOTE_MLIP_CONDA_INIT",
    "source /opt/anaconda3/2023.09/etc/profile.d/conda.sh && conda activate mace",
)
REMOTE_MLIP_DEVICE = os.getenv("SIMMOF_REMOTE_MLIP_DEVICE", "cuda:0")


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
