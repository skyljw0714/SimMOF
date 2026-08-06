# SAMOSA (modified for SimMOF)

This directory contains a modified copy of SAMOSA (Structural Activation via
Metal Oxidation State Analysis), a solvent-removal protocol that generates
activated crystal structures from experimental crystallographic information.

Original project: https://github.com/uowoolab/SAMOSA
Citation: https://doi.org/10.1021/acs.jcim.4c01897
License: GNU General Public License v3.0 (see `LICENSE` in this directory)

Changes made for SimMOF integration:
- `run_single.py` added — runs SAMOSA solvent removal on a single CIF file,
  invoked directly by `structure/agent.py`.
- `main.py` and `modules/*.py` modified to support this integration.

This code remains licensed under GPLv3, unchanged from upstream.
