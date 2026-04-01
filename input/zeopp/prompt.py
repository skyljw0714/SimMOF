
ZEOPP_EXAMPLES = """

Examples:
1. User query: "Calculate the accessible surface area of UiO-66 for a nitrogen probe."
   Output (JSON):
   {
     "MOF": "UiO-66",
     "simulation_type": "surface area",
     "command": "-ha -sa",
     "probe_radius": 1.8,
     "num_samples": 50000
   }

2. User query: "What is the largest pore diameter in MOF-5?"
   Output (JSON):
   {
     "MOF": "MOF-5",
     "simulation_type": "pore diameter",
     "command": "-ha -res",
     "probe_radius": null,
     "num_samples": null
   }

3. User query: "Find the accessible volume of ZIF-8 for CO2."
   Output (JSON):
   {
     "MOF": "ZIF-8",
     "simulation_type": "accessible volume",
     "command": "-ha -vol",
     "probe_radius": 1.65 ,
     "num_samples": 50000
   }
"""


ZEOPP_DESCRIPTION = """

Zeo++ is a command-line software for analyzing the structure of porous materials such as MOFs.  
It can calculate properties like pore diameters, channel dimensionality, accessible surface area, accessible volume, probe-occupiable volume, and pore size distribution using various commands.

- **Pore diameters:**  
  Calculates the largest included sphere, largest free sphere, and largest included sphere along the free sphere path.  
  command : -ha -res
  probe_radius : null
  num_samples : null
  you must not use `probe_radius` and `num_samples` if you use `-ha -res`

- **Accessible surface area:**  
  Calculates the surface area accessible to a spherical probe. 
  command : -ha -sa
  probe_radius : default is 1.2(H2), depands on the guest molecule
  num_samples : default is 50000

- **Accessible volume: Pore volume**  
  Calculates the volume accessible to the center of a spherical probe. 
  command : -ha -vol
  probe_radius : default is 1.2(H2), depands on the guest molecule
  num_samples : default is 50000

**Notes:**  
- `chan_radius` and `probe_radius` are usually set to the same value (recommended).
- The commonly used probe radius for H₂ is 1.2 Å and for N₂ is 1.8 Å when measuring properties such as pore volume (PV) or surface area (SA).
- You must use the default value for `probe_radius` and `num_samples` if no specific value is provided.
- When calculating accessible properties for a specific gas, the probe radius should correspond to the half of the kinetic diameter
- `num_samples` is the number of Monte Carlo samples (per atom or per unit cell, default is 50000).

"""