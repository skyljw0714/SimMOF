from config import working_dir
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
        self.atoms = self.get_atoms()
    
    def get_guest(self, save_dir):
        output_path = Path(save_dir) / f"{self.name}.xyz"
        write(output_path, self.atoms)
        return


    def get_atoms(self):
        self._download_sdf_from_pubchem()
        atoms = self._optimize_atoms()
        return atoms

    def _download_sdf_from_pubchem(self):
        
        exceptions = []
        filename = os.path.join(working_dir, f"{self.name}.sdf")

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

    def _optimize_atoms(self):
        sdf_path = os.path.join(working_dir, f"{self.name}.sdf")
        xyz_path = os.path.join(working_dir, f"{self.name}.xyz")

        
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

