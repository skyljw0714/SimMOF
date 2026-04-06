import re
import math
import pandas as pd     
import os

from pathlib import Path

from config import COREMOF_DATA_CSV, COREMOF_PHASE_DIRS

print(1)

try:
    import ccdc
    from ccdc.search import TextNumericSearch
    from ccdc.io import EntryReader, CrystalWriter


except ModuleNotFoundError:
    
    TextNumericSearch = None
    EntryReader = None
    CrystalWriter = None
    print("Warning: 'ccdc' not available in current environment. Use csd_api for structure loading.")


CoREMOF_2024_abspath = Path(COREMOF_DATA_CSV)

CoREMOF_2024 = pd.read_csv(CoREMOF_2024_abspath)
CoREMOF_2024_filename = CoREMOF_2024['refcode'].values.tolist()


entry_reader = EntryReader('CSD') if EntryReader is not None else None


MOF_dict = {
    
    'Cu-BTC': 'HKUST-1',
    'IRMOF-1': 'MOF-5',

}

class MOFLoader:
    def __init__(self, name, doi = None):
        self.name = name 
        self.doi = doi        
        self.get_refcode()
        
    def get_refcode(self):
        print('### MOF Name : {} ###'.format(self.name))
        print('### Find REFCODE ... ###')
        self.refcode = None
        refcode = self._get_refcode_from_name(self.name)  
        if refcode:
            print('### name is REFCODE ###')
            if len(refcode) > 1:
                print('### More than one structure found for this REFCODE ###')
            self.refcode = refcode
            return
        
        refcode = self._get_refcode_from_synonym(self.name) 
        if refcode:
            print('### name is synonym of REFCODE ###')
            if len(refcode) > 1:
                print('### More than one structure found for this name ###')
            self.refcode = refcode
            return
        
        refcode = self._get_refcode_from_doi(self.doi) if self.doi else None 
        if refcode:
            print('### DOI is uploaded on CSD ###')
            if len(refcode) > 1:
                print('### More than one structure found for this DOI ###')
            self.refcode = refcode
            return

        if not self.refcode and self.name in MOF_dict.keys():
            print(f'### {self.name} == {MOF_dict[self.name]} ###')
            self.refcode = self._get_refcode_from_synonym(MOF_dict[self.name])
            return
        
    def _get_refcode_from_name(self, name):
        search = TextNumericSearch()
        search.add_all_identifiers(name.upper(), mode='exact')
        search_result = search.search()
        if search_result:
            print(f'### {name} is REFCODE ###')
            return [structure.identifier for structure in search_result]
          
    def _get_refcode_from_synonym(self, name):
        search = TextNumericSearch()
        search.add_synonym(name, mode='anywhere')
        search_result = search.search()
        exact_match = [structure.identifier for structure in search_result if is_exact_match(structure.entry.synonyms[0], name)]
        if exact_match:
            return exact_match
        
        else:
            print('### No exact match found ###')
            print('### Found structures with synonym name ###')
            return [structure.identifier for structure in search_result]
        
    def _get_refcode_from_doi(self, doi):
        print('### Find REFCODE from DOI... ###')
        search = TextNumericSearch()
        search.add_doi(self.doi, mode = 'anywhere')
        search_result= search.search()
        if search_result:
            return [structure.identifier for structure in search_result]
        
    
    def get_structure(self, save_path):
        if not self.refcode:
            print('### No REFCODE found ###')
            return
        
        expanded = set()
        for rc in self.refcode:
            expanded.update(list_coremof_variants_from_base(rc, CoREMOF_2024_filename))
        coremof_structure = sorted(expanded)

        if coremof_structure:
            if len(coremof_structure) == 1:
                print('###Found structure from CoREMOF###')
                self.cif_string = get_cif_from_mofdb(coremof_structure[0])
                write_cif_from_mofdb(save_path / Path(f'./{self.name}.cif'), self.cif_string)
                self.cif_path = Path(save_path / Path(f'./{self.name}.cif'))
                return 
            
            if len(coremof_structure) > 1:
                print('###More than one structure found for this REFCODE###')
                self.cif_string = find_min_volume_cif(coremof_structure)
                write_cif_from_mofdb(save_path / Path(f'./{self.name}.cif'), self.cif_string)
                self.cif_path = save_path / Path(f'./{self.name}.cif')
                return 
                
        if len(self.refcode) == 1:
            print('### Structure Not in Core MOF. Found structure from CSD ###')
            search = TextNumericSearch()
            search.add_all_identifiers(self.refcode[0].upper(), mode='exact')
            search_result = search.search()
            structure = search_result[0]

            crystal = structure.crystal
            crystal.molecule = crystal.molecule.heaviest_component

            with CrystalWriter(save_path / Path(f'./{self.name}.cif')) as writer:
                writer.write(crystal)

            self.cif_path = save_path / Path(f'./{self.name}.cif')
            return

        if len(self.refcode) > 1:
            print('### Structure Not in Core MOF. Found structure from CSD ###')
            print('### More than one structure found for this name. there can be an error. ###')
            print('### Download the first structure found ###')
            search = TextNumericSearch()
            search.add_all_identifiers(self.refcode[0].upper(), mode='exact')
            search_result = search.search()
            structure = search_result[0]

            crystal = structure.crystal
            crystal.molecule = crystal.molecule.heaviest_component

            with CrystalWriter(save_path / Path(f'./{self.name}.cif')) as writer:
                writer.write(crystal)

            self.cif_path = save_path / Path(f'./{self.name}.cif')
            return
   

def extract_cell_parameters(cif_data):
    patterns = {
        'a': r"_cell_length_a\s+([0-9.]+)",
        'b': r"_cell_length_b\s+([0-9.]+)",
        'c': r"_cell_length_c\s+([0-9.]+)",
        'alpha': r"_cell_angle_alpha\s+([0-9.]+)",
        'beta': r"_cell_angle_beta\s+([0-9.]+)",
        'gamma': r"_cell_angle_gamma\s+([0-9.]+)"
    }
    parameters = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, cif_data)
        if match:
            parameters[key] = float(match.group(1))
        else:
            parameters[key] = None

    return parameters


def calculate_volume_from_parameters(cell_parameters):
    alpha_rad = math.radians(cell_parameters['alpha'])
    beta_rad = math.radians(cell_parameters['beta'])
    gamma_rad = math.radians(cell_parameters['gamma'])

    volume = cell_parameters['a'] * cell_parameters['b'] * cell_parameters['c'] * math.sqrt(
        1 - math.cos(alpha_rad)**2 - math.cos(beta_rad)**2 - math.cos(gamma_rad)**2
        + 2 * math.cos(alpha_rad) * math.cos(beta_rad) * math.cos(gamma_rad)
    )
    return volume


def find_min_volume_cif(coremof_perfect_match):
    min_vol = None
    min_cif = None
    for i in coremof_perfect_match:
        cif_string = get_cif_from_mofdb(i)  
        if not cif_string:
            continue  

        cell_parameters = extract_cell_parameters(cif_string)
        vol = calculate_volume_from_parameters(cell_parameters)

        if min_vol is None or vol < min_vol:
            min_vol = vol
            min_cif = cif_string
    return min_cif

def find_file_by_name(name, file_list):
    for file in file_list:
        if name in file:
            return file
    return None

def list_coremof_variants_from_base(base_refcode: str, filenames: list[str]) -> list[str]:
    base = re.sub(r'[^A-Za-z0-9]', '', base_refcode).upper()
    pat = re.compile(rf"^({re.escape(base)}(?:\d{{2}})?)_(ASR|FSR|ion)_pacman$", re.IGNORECASE)
    variants = set()
    for fn in filenames:
        m = pat.match(str(fn))
        if m:
            variants.add(m.group(1))
    return sorted(variants)


def get_refcode_mofdb(name):
    file = find_file_by_name(name, CoREMOF_2024_filename)
    if file:
        return file
    else:
        raise ValueError('refcode not in CoREMOF_2024')


PHASE_DIRS = {key: Path(value) for key, value in COREMOF_PHASE_DIRS.items()}

COL_FILENAME = "refcode"
COL_COREID   = "coreid"


def _pick_candidate_row_for_refcode(base_refcode: str):
    base = base_refcode.upper()
    wanted = [
        f"{base}_ASR_pacman",
        f"{base}_FSR_pacman",
        f"{base}_ion_pacman",
    ]

    hits = CoREMOF_2024[CoREMOF_2024[COL_FILENAME].isin(wanted)]

    if len(hits) == 0:
        raise LookupError(f"No row in CSV for refcode '{base_refcode}' (wanted={wanted})")

    if len(hits) == 1:
        return hits.iloc[0]

    hits_sorted = hits.copy()
    def _phase_rank(fn: str) -> int:
        if fn.endswith("_FSR_pacman"): return 0
        if fn.endswith("_ASR_pacman"): return 1
        if fn.endswith("_ion_pacman"): return 2
        return 3
    hits_sorted["__rank"] = hits_sorted[COL_FILENAME].apply(_phase_rank)
    hits_sorted = hits_sorted.sort_values(["__rank", COL_FILENAME])
    return hits_sorted.iloc[0]

def _phase_from_filename(fn: str) -> str:
    if fn.endswith("_FSR_pacman"): return "FSR"
    if fn.endswith("_ASR_pacman"): return "ASR"
    if fn.endswith("_ion_pacman"): return "Ion"
    raise ValueError(f"Cannot infer phase from filename='{fn}'")

def get_cif_from_mofdb(refcode: str) -> str:
    row = _pick_candidate_row_for_refcode(refcode)
    filename = str(row[COL_FILENAME])
    coreid   = str(row[COL_COREID])
    phase    = _phase_from_filename(filename)

    cif_dir = PHASE_DIRS[phase]
    cif_path = cif_dir / f"{coreid}.cif"
    if not cif_path.exists():
        alt = list(cif_dir.glob(f"{coreid}.*"))
        if alt:
            cif_path = alt[0]
        else:
            raise FileNotFoundError(f"CIF not found: {cif_path}")

    return cif_path.read_text(encoding="utf-8", errors="ignore")
    
def write_cif_from_mofdb(path, cif):
    with open('{}'.format(path), 'w') as f:
        f.write(cif)

def normalize_string(s):
    return re.sub(r"[-\s]", "", s).lower()

def is_exact_match(name, keyword):
    normalized_keyword = normalize_string(keyword)
    normalized_name = normalize_string(name)
    
    if normalized_name == normalized_keyword:
        return True
    else:
        return False 
 
