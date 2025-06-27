import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from config.env
load_dotenv('config.env')

# Get API key from environment variables
api_key = os.getenv('OPENAI_API_KEY')

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in config.env file.")

# Create shared ChatOpenAI instance
chat_model = ChatOpenAI(model_name="gpt-4", temperature=0.0)

working_dir = "/home/users/skyljw0714/MOFScientist/working_dir"
zeo_dir = "/home/users/skyljw0714/MOFScientist/ZeoPP"


# RASPA input format template
RASPA_FORMAT = """
# RASPA Simulation Input File
# Simulation type: {simulation_type}
# MOF: {mof}
# Guest: {guest}

SimulationType                {simulation_type}
NumberOfCycles                10000
NumberOfInitializationCycles  1000
PrintEvery                    1000
PrintPropertiesEvery          1000

Forcefield                    {forcefield}

Framework 0
FrameworkName {mof}
UnitCells {unit_cells}
ExternalTemperature {temperature}
ExternalPressure {pressure}

Component 0 MoleculeName {guest}
    MoleculeDefinition {guest}
    CreateNumberOfMolecules 0
"""

# Example method section from paper
METHOD_SECTION = """
The molecular simulations were performed using the RASPA software package. 
The Universal Force Field (UFF) was used to describe the framework atoms, 
while the guest molecules were modeled using the TraPPE force field. 
The simulation box consisted of 2×2×2 unit cells of the MOF structure. 
The system was equilibrated for 10,000 cycles followed by production runs 
of 50,000 cycles. The temperature was maintained at 298 K using the 
Berendsen thermostat, and the pressure was set to 1 bar for adsorption 
isotherm calculations. The cutoff distance for van der Waals interactions 
was set to 12.8 Å, and long-range electrostatic interactions were 
calculated using the Ewald summation method.
""" 