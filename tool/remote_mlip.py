import subprocess
from pathlib import Path
from packmol.run_packmol import run_packmol_from_cif
from config import (
    REMOTE_MLIP_CONDA_INIT,
    REMOTE_MLIP_DEVICE,
    REMOTE_MLIP_GPU_HOST,
    REMOTE_MLIP_GPU_PORT,
    REMOTE_MLIP_GPU_USER,
    REMOTE_MLIP_LOCAL_OUTPUT_DIR,
    REMOTE_MLIP_REMOTE_DIR,
)

GPU_HOST = REMOTE_MLIP_GPU_HOST
GPU_USER = REMOTE_MLIP_GPU_USER
GPU_PORT = REMOTE_MLIP_GPU_PORT
REMOTE_DIR = str(REMOTE_MLIP_REMOTE_DIR)

def run_remote_mlip_be(
    host_cif,
    complex_dir,
    guest_xyz,
    okdir,
    top_n=100,
    local_output_dir=str(REMOTE_MLIP_LOCAL_OUTPUT_DIR),
):
    host_cif = Path(host_cif)
    complex_dir = Path(complex_dir)
    guest_xyz = Path(guest_xyz)
    local_output_dir = Path(local_output_dir)

    
    subprocess.run([
        "ssh", "-p", GPU_PORT, f"{GPU_USER}@{GPU_HOST}",
        f"mkdir -p {REMOTE_DIR}"
    ], check=True)

    
    subprocess.run([
        "scp", "-P", GPU_PORT, "-r",
        str(host_cif),
        str(complex_dir),
        str(guest_xyz),
        f"{GPU_USER}@{GPU_HOST}:{REMOTE_DIR}/",
    ], check=True)

    
    cmd = f"""
    {REMOTE_MLIP_CONDA_INIT} && \
    cd {REMOTE_DIR} && \
    CUDA_VISIBLE_DEVICES=0 python -c "from utils import run_mlip_binding; \
run_mlip_binding(
    '{REMOTE_DIR}/{host_cif.name}',
    '{REMOTE_DIR}/{complex_dir.name}',
    '{REMOTE_DIR}/{guest_xyz.name}',
    '{REMOTE_DIR}/{okdir}',
    device='{REMOTE_MLIP_DEVICE}',
    top_n={top_n}
)"
    """

    subprocess.run([
        "ssh", "-p", GPU_PORT, f"{GPU_USER}@{GPU_HOST}", cmd
    ], check=True)

    
    local_output_dir.mkdir(parents=True, exist_ok=True)

    
    remote_csv = f"{REMOTE_DIR}/{complex_dir.name}/mlip_binding_results.csv"
    subprocess.run([
        "scp", "-P", GPU_PORT,
        f"{GPU_USER}@{GPU_HOST}:{remote_csv}",
        f"{local_output_dir}/",
    ], check=True)

    
    subprocess.run([
        "scp", "-P", GPU_PORT, "-r",
        f"{GPU_USER}@{GPU_HOST}:{REMOTE_DIR}/{okdir}",
        f"{local_output_dir}/",
    ], check=True)

    csv_path = local_output_dir / "mlip_binding_results.csv"
    okdir_path = local_output_dir / okdir

    return csv_path, okdir_path


if __name__ == "__main__":
    run_remote_mlip_geo("input_cifs", "ok_cifs")
