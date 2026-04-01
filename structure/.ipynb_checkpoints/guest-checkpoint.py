from config import working_dir
import os
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
import ase.io

class GuestLoader:
    def __init__(self, name):
        self.name = name 
        self.make_structure_file()
        self.structure = self.get_structure()

    def get_structure(self):
        xyz_path = os.path.join(working_dir, f"{self.name}.xyz")
        return ase.io.read(xyz_path)

    def make_structure_file(self):
        self._download_sdf_from_pubchem()
        self._optimize_and_convert_to_xyz()

    def _download_sdf_from_pubchem(self):
        exceptions = ['CO', 'HF']  # 화학식으로 검색이 더 안정적인 경우
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
            pcp.download('SDF', filename, cid, 'cid', overwrite=True, record_type='3d')
            print(f"[INFO] Downloaded: {filename}")

        except Exception as e:
            raise RuntimeError(f"Failed to download SDF from PubChem: {e}")

    def _optimize_and_convert_to_xyz(self):
        sdf_path = os.path.join(working_dir, f"{self.name}.sdf")
        xyz_path = os.path.join(working_dir, f"{self.name}.xyz")

        # RDKit로 읽고 3D 좌표 생성
        mol = Chem.MolFromMolFile(sdf_path, removeHs=False)
        if mol is None:
            raise ValueError("Failed to load molecule from SDF.")

        # 3D 최적화 (필요한 경우)
        if mol.GetNumConformers() == 0:
            print("[INFO] Embedding 3D conformer...")
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.UFFOptimizeMolecule(mol)

        # XYZ 포맷으로 저장
        conf = mol.GetConformer()
        atoms = []
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            atoms.append((atom.GetSymbol(), [pos.x, pos.y, pos.z]))
        
        ase_atoms = Atoms(symbols=[a[0] for a in atoms],
                          positions=[a[1] for a in atoms])
        ase.io.write(xyz_path, ase_atoms)
        print(f"[INFO] Saved optimized structure to {xyz_path}")
