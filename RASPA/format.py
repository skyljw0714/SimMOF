RASPA_FORMAT = """
SimulationType                MonteCarlo
NumberOfCycles                5000
NumberOfInitializationCycles  1000
PrintEvery                    1000
RestartFile                   no
UseChargesFromCIFFile        yes

Forcefield 		 UFF
CutOffVDW 		 12.8 #Default

ExternalTemperature 	 298 #Default

Framework 0
FrameworkName                 IRMOF-1   # Name of the MOF structure
UnitCells                     1 1 1     # Number of unit cells in each direction

Component 0 MoleculeName             methane #Name of the guest molecule
            MoleculeDefinition       TraPPE #Default
            TranslationProbability   0.5 #Default
            ReinsertionProbability   0.5 #Default
            SwapProbability          1.0 #Default
            CreateNumberOfMolecules  0 #Default
            
"""