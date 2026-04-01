import os
import ase.io
import glob
import shutil
import copy


from .guest import GuestLoader
from config import get_csd_api_python_command, working_dir
import subprocess 
from packmol.run_packmol import run_packmol_from_cif
from pathlib import Path

PACKMOL_DEFAULT_TOLERANCE = 2.0


class StructureAgent:
    def __init__(self):
        pass

    
    def get_guest(self, guest_name, save_dir):
        g = GuestLoader(guest_name)
        g.get_guest(save_dir)
        return 


    def get_mof(self, mof_name, save_dir):
        project_root = os.path.dirname(os.path.abspath(__file__))
        script = f"""
import sys
sys.path.append('{project_root}')
from pathlib import Path
from structure.mof import MOFLoader
m = MOFLoader('{mof_name}')
m.get_structure(Path('{save_dir}'))
"""
        result = subprocess.run(
                get_csd_api_python_command() + ["-c", script],
                capture_output=True,
                text=True
            )
        if result.returncode == 0:
            print(f"Successfully fetched {mof_name} structure")
        else:
            print(f"Error fetching {mof_name}")
            print(result.stderr)
        return 



class VASPStructureAgent:
    def __init__(self, number_of_guest: int = 1, number_of_system: int = 1):
        self.number_of_guest = number_of_guest
        self.number_of_system = number_of_system

    
    
    
    def get_mof(self, mof_name, save_dir):
        project_root = os.path.dirname(os.path.abspath(__file__))

        script = f"""
import sys
sys.path.append('{project_root}')
from pathlib import Path
from structure.mof import MOFLoader
m = MOFLoader('{mof_name}')
m.get_structure(Path('{save_dir}'))
"""
        result = subprocess.run(
            get_csd_api_python_command() + ["-c", script],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print(f"Successfully fetched {mof_name} structure")
        else:
            print(f"Error fetching {mof_name}")
            print(result.stderr)

        mof_path = os.path.join(save_dir, f"{mof_name}.cif")

        try:
            atoms = ase.io.read(mof_path)
            ase.io.write(mof_path, atoms, format="cif")
            print(f"[CIF] Cleaned MOF CIF written to: {mof_path}")
        except Exception as e:
            print(f"[CIF] Warning: failed to clean MOF CIF ({mof_path}): {e}")

        return mof_path

    
    
    
    def get_guest(self, guest_name, save_dir, mof_path=None):
        
        g = GuestLoader(guest_name)
        g.get_guest(save_dir)  
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

        if batch_root is not None:
            save_dir = os.path.join(batch_root, job_name)
        else:
            save_dir = os.path.join(working_dir, job_name)

        os.makedirs(save_dir, exist_ok=True)

        print(f"Saving output to: {save_dir}")

        context["work_dir"] = save_dir

        
        mof_path = self.get_mof(mof_name, save_dir)
        context["mof_path"] = mof_path

        guest_xyz_path = None
        guest_cif_path = None

        
        if guest_name:
            guest_xyz_path, guest_cif_path = self.get_guest(
                guest_name, save_dir, mof_path=mof_path
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

        save_dir = str(Path(working_dir) / job_name)
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        context["work_dir"] = save_dir

        mof_path = self.get_mof(mof_name, save_dir)
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
                guest_name, save_dir, mof_path=mof_path
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
            context["complex_path"] = complex_cif_paths[0]
            complex_label = Path(context["complex_path"]).stem
            context["vasp_label"] = complex_label
            context.setdefault("vasp_system", {})
            context["vasp_system"]["label"] = complex_label

        return context

class ZeoppStructureAgent():
    
    def __init__(self):
        pass
    
    def get_guest(self, guest_name, save_dir):
        g = GuestLoader(guest_name)
        g.get_guest(save_dir)
        return 

    def get_mof(self, mof_name, save_dir):
        project_root = os.path.dirname(os.path.abspath(__file__))
        script = f"""
import sys
sys.path.append('{project_root}')
from pathlib import Path
from structure.mof import MOFLoader
m = MOFLoader('{mof_name}')
m.get_structure(Path('{save_dir}'))
"""
        result = subprocess.run(
            get_csd_api_python_command() + ["-c", script],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"Successfully fetched {mof_name} structure")
        else:
            print(f"Error fetching {mof_name}")
            print(result.stderr)
        return

    def run(self, context):

        mof_name   = context["mof"]
        guest_name = context["guest"]

        
        save_dir = os.path.join(working_dir, context["job_name"])
        os.makedirs(save_dir, exist_ok=True)

        print(f"Saving output to: {save_dir}")

        context["work_dir"] = save_dir

        
        self.get_mof(mof_name, save_dir)
        context["mof_path"] = os.path.join(save_dir, f"{mof_name}.cif")

        return context

    
class LAMMPSStructureAgent():
    def __init__(self):
        pass

    def get_guest(self, guest_name, save_dir):
        g = GuestLoader(guest_name)
        g.get_guest(save_dir)
        return 

    def get_mof(self, mof_name, save_dir):
        project_root = os.path.dirname(os.path.abspath(__file__))
        script = f"""
import sys
sys.path.append('{project_root}')
from pathlib import Path
from structure.mof import MOFLoader
m = MOFLoader('{mof_name}')
m.get_structure(Path('{save_dir}'))
"""
        result = subprocess.run(
            get_csd_api_python_command() + ["-c", script],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"Successfully fetched {mof_name} structure")
        else:
            print(f"Error fetching {mof_name}")
            print(result.stderr)
        return

    def run(self, context):

        mof_name   = context["mof"]
        guest_name = context["guest"]

        
        save_dir = os.path.join(working_dir, context["job_name"])
        os.makedirs(save_dir, exist_ok=True)

        print(f"Saving output to: {save_dir}")

        context["work_dir"] = save_dir

        
        self.get_mof(mof_name, save_dir)
        context["mof_path"] = os.path.join(save_dir, f"{mof_name}.cif")

        
        if guest_name:
            self.get_guest(guest_name, save_dir)
            context["guest_path"] = os.path.join(save_dir, f"{guest_name}.xyz")

        return context
    
class RASPAStructureAgent:
    def __init__(self):
        pass

    def get_mof(self, mof_name, save_dir):
        project_root = os.path.dirname(os.path.abspath(__file__))
        script = f"""
import sys
sys.path.append('{project_root}')
from pathlib import Path
from structure.mof import MOFLoader
m = MOFLoader('{mof_name}')
m.get_structure(Path('{save_dir}'))
"""
        result = subprocess.run(
            get_csd_api_python_command() + ["-c", script],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"Successfully fetched {mof_name} structure")
        else:
            print(f"Error fetching {mof_name}")
            print(result.stderr)

    def run(self, context):
        job_name = context["job_name"]
        guest_name = context.get("guest")

        
        base_dir = Path(working_dir) / job_name
        base_dir.mkdir(parents=True, exist_ok=True)
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

            context["work_dir"] = str(base_dir)
            context["batch"] = batch
            context["batch_size"] = len(batch)

            print(f"[RASPAStructureAgent] batch created: {len(batch)} MOFs from {okdir}")
            return context

        
        mof_name = context["mof"]

        
        context["work_dir"] = str(base_dir)

        self.get_mof(mof_name, str(base_dir))
        context["mof_path"] = str(base_dir / f"{mof_name}.cif")

        return context