import os
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
from ase.io import write
from pathlib import Path

class GuestLoader:
    def __init__(self, name):
        self.name = name 
        self.atoms = None
    
    def get_guest(self, save_dir):
        output_dir = Path(save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        atoms = self.get_atoms(save_dir=output_dir)
        output_path = output_dir / f"{self.name}.xyz"
        write(output_path, atoms)
        return


    def get_atoms(self, save_dir=None):
        if self.atoms is not None and save_dir is None:
            return self.atoms

        scratch_dir = Path(save_dir) if save_dir is not None else Path.cwd()
        scratch_dir.mkdir(parents=True, exist_ok=True)

        self._download_sdf_from_pubchem(scratch_dir)
        atoms = self._optimize_atoms(scratch_dir)

        if save_dir is None:
            self.atoms = atoms

        return atoms

    def _download_sdf_from_pubchem(self, save_dir):
        
        exceptions = []
        filename = str(Path(save_dir) / f"{self.name}.sdf")

        try:
            if self.name in exceptions:
                compounds = pcp.get_compounds(self.name, 'formula')
            else:
                compounds = pcp.get_compounds(self.name, 'name')

            if not compounds:
                raise ValueError(f"No compound found for name: {self.name}")
            if len(compounds) > 1:
                print("Warning: Multiple compounds found. Using the first one.")

            cid = compounds[0].cid
            try:
                pcp.download('SDF', filename, cid, 'cid',
                            overwrite=True, record_type='3d')
                print(f"[INFO] Downloaded 3D SDF: {filename}")
            except Exception as e3d:
                
                print(f"[WARN] 3D SDF not available for {self.name} (cid={cid}): {e3d}")
                pcp.download('SDF', filename, cid, 'cid', overwrite=True)
                print(f"[INFO] Downloaded fallback SDF (2D/default): {filename}")

        except Exception as e:
            raise RuntimeError(f"Failed to download SDF from PubChem: {e}")

    def _optimize_atoms(self, save_dir):
        sdf_path = str(Path(save_dir) / f"{self.name}.sdf")

        
        mol = Chem.MolFromMolFile(sdf_path, removeHs=False)
        if mol is None:
            raise ValueError("Failed to load molecule from SDF.")

        
        if mol.GetNumConformers() == 0:
            print("[INFO] Embedding 3D conformer...")
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.UFFOptimizeMolecule(mol)

        
        conf = mol.GetConformer()
        atoms = []
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            atoms.append((atom.GetSymbol(), [pos.x, pos.y, pos.z]))
        
        ase_atoms = Atoms(symbols=[a[0] for a in atoms],
                          positions=[a[1] for a in atoms])
        
        return ase_atoms

