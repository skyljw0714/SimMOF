# SimMOF

  ## Overview

  SimMOF is a Python-based workflow framework for metal-organic framework (MOF) simulations and analysis on HPC clusters.

  It integrates multiple simulation engines and supporting tools in a single workflow layer:

  - LAMMPS for classical MD-style simulations
  - VASP for DFT workflows
  - Zeo++ for pore and geometric analysis
  - RASPA for adsorption and Monte Carlo simulations
  - Packmol for structure packing / host-guest setup
  - AI-based modules for query parsing, planning, retrieval, analysis, and response generation

  The framework is designed around Python agents and workflow stages that prepare structures, generate simulator inputs, run jobs, handle errors, parse outputs, and
  summarize results.

  ———

  ## Project Structure
```text
  SimMOF/
  ├── main.py
  ├── input/        # Simulator-specific input generation
  ├── LAMMPS/       # LAMMPS workflow logic and runner
  ├── VASP/         # VASP workflow logic and runner
  ├── Zeopp/        # Zeo++ workflow logic and runner
  ├── RASPA/        # RASPA workflow logic and runner
  ├── packmol/      # Packmol-based structure generation
  ├── structure/    # MOF / guest loading and structure preparation
  ├── tool/         # Shared helper tools and screening utilities
  ├── rag/          # Retrieval and vector-store related modules/data
  ├── analysis/     # Result aggregation and scientific interpretation
  ├── output/       # Output parsing for each simulator
  └── error/        # Error handling and retry logic
```

  Practical role of each directory:

  - input/: builds simulator input files from parsed workflow context
  - LAMMPS/, VASP/, Zeopp/, RASPA/: simulator-specific agents, runners, and helpers
  - packmol/: prepares packed structures for host-guest workflows
  - structure/: retrieves or prepares MOF and guest structures
  - tool/: shared utilities used across workflows
  - rag/: retrieval-augmented generation components and local knowledge assets
  - analysis/: summarizes outputs into interpretable results
  - output/: parses raw simulator output files
  - error/: detects failures, patches inputs when possible, and retries jobs

  ———

  ## Requirements

  ### Python

  - Python 3.9
  - conda environment manager
  - The recommended environment is provided as `environment.yml`
  - Recommended main environment name: `simmof`

  ### Python packages

  The required Python packages are managed through `environment.yml`.

  Key packages used in this project include:

  - pymatgen
  - torch
  - langchain
  - langchain-openai
  - openai
  - python-dotenv
  - rdkit
  - openbabel
  - ase
  - numpy
  - pandas
  - mofchecker

  ### External software

  The following external software must be installed and available on the cluster:

  - LAMMPS
  - Moltemplate
  - VASP
  - Zeo++
  - RASPA
  - Packmol
  - CCDC / CSD Python API

  ### CCDC / CSD Python API

  `structure/agent.py` depends on the CCDC Python API, so a working CCDC installation is required for structure-agent functionality.

  Download the CCDC software from the official CSDS Downloads page:

  https://www.ccdc.cam.ac.uk/support-and-resources/csdsdownloads/

  Notes:
  - You need a valid CCDC customer number and activation key to obtain the latest CSD Suite installers.
  - The CSD Python API is installed with the CSD installation, and CCDC also states that it can be installed manually using provided conda packages.
  - Make sure the CCDC Python API is available in the Python environment used to run SimMOF.
  - If CCDC is not installed, structure-agent features that rely on `ccdc` will not work.

  ### Moltemplate installation

  SimMOF's LAMMPS input-generation workflow uses Moltemplate utilities such as
  `ltemplify.py` and `moltemplate.sh`, so Moltemplate must be installed separately.

  Download Moltemplate from the official website:

  https://www.moltemplate.org/download.html

  After installation, set the Moltemplate script paths in your configuration or
  environment variables as needed for your local system.

  Example checks:

      /path/to/moltemplate.sh -h
      python /path/to/ltemplify.py -h

  ———

  ## Installation

  Create the environment from the provided YAML file:

      conda env create -f environment.yml
      conda activate simmof

  If you want to create the environment under a different name for testing:

      conda env create -n simmof-test -f environment.yml
      conda activate simmof-test

  After installation, verify that the main modules import correctly before running workflows.

  Example checks:

      python -c "import langchain_openai; print('langchain_openai ok')"
      python -c "import mofchecker; print('mofchecker ok')"
      python -c "from structure.agent import *; print('structure.agent ok')"
      python -c "from input.lammps.pipeline_lammps import generate_lammps_inputs; print('pipeline ok')"
      
  ———

  ## Environment Variables

  SimMOF uses a .env file for API keys, model settings, working directories, and executable paths.

  Typical values include:

  OPENAI_API_KEY=your_api_key

  SIMMOF_WORKING_DIR=/path/to/working_dir

  SIMMOF_LAMMPS_EXECUTABLE=/path/to/lammps
  SIMMOF_VASP_EXECUTABLE=/path/to/vasp_std
  SIMMOF_VASP_POTENTIAL_DIR=/path/to/potpaw
  SIMMOF_ZEOPP_BIN=/path/to/network
  SIMMOF_RASPA_SIMULATE_BIN=/path/to/simulate
  SIMMOF_PACKMOL_EXECUTABLE=/path/to/packmol

  Use cluster-local absolute paths for all external executables.

  ———

  ## Usage

  Run the main workflow entrypoint:

  conda activate simmof
  python main.py

  Typical execution flow:

  1. parse the query or requested task
  2. build a workflow
  3. prepare structures and simulator inputs
  4. submit or run jobs
  5. parse outputs
  6. summarize and analyze results

  Basic example:

  python main.py

  If needed, set the working directory root before running:

  export SIMMOF_WORKING_DIR=/scratch/$USER/simmof_work
  python main.py

  ———

  ## Notes

  - SimMOF is intended for cluster use, not only local interactive runs.
  - .env should be treated as the primary place to configure executable paths and API keys.
  - VASP is not bundled; you must provide a valid licensed installation and pseudopotentials.
  - The exact job submission behavior depends on your cluster environment and scheduler setup.
  - Before running production workflows, verify:
      - executable paths
      - pseudopotential locations
      - working directory permissions
      - API key availability
  - AI-generated planning or input fixes should be reviewed before publication-grade simulations.
  - Keep large workflow outputs on scratch or project storage rather than home directories.
