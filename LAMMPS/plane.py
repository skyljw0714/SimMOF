#!/bin/env python

# Molecular Simulation Laboratory
# Last modified : 2021/3/31

#################################################################
# Module Import
#################################################################
import argparse
import sys

from ase.geometry import cellpar_to_cell
import numpy as np

#################################################################
# Main Code Line
#################################################################
parser = argparse.ArgumentParser(description="Code for obtaining planes for packmol inputs")
parser.add_argument("-m", "--margin", action='store', dest="margin", type=float, help="set margin", default=0.0)
parser.add_argument("-t", "--type", action='store', dest="type", type=str, help="either cell or cellpar: default=cellpar", choices=["cell", "cellpar"], default="cellpar")
parser.add_argument("inputs", action='store', help="enter cell info", nargs='*', type=float)
args = parser.parse_args()


class ShowPlane():
    def __init__(self):
        input_len = len(args.inputs)
        if args.type == "cellpar":
            if input_len != 6:
                sys.exit("Inappropriate number of inputs")
            self.planes_from_cellpar(args.inputs, args.margin)
        else:
            if input_len != 9:
                sys.exit("Inappropriate number of inputs")
            args.inputs = np.reshape(args.inputs, (3, 3))
            self.planes_from_cell(args.inputs, args.margin)

    def planes_from_cellpar(self, cellpar, marge):
        cell = cellpar_to_cell(cellpar)
        self.planes_from_cell(cell, marge)

    def planes_from_cell(self, cell, marge):
        # Cell lengths.
        L = [np.linalg.norm(cell[i]) for i in range(3)]
        # Normalized cell vectors.
        e = [cell[i]/L[i] for i in range(3)]
        # Permutation of cells.
        ijk = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]

        for i, j, k in ijk:
            # Normal vector for plane.
            n = np.cross(e[i], e[j])
            n /= np.linalg.norm(n)

            ekn = np.dot(e[k], n)

            # over (x - margin*e_k).n = 0.
            # below = (x - (L_k-margin)*e_k).n = 0.
            print("over plane {} {} {} {}".format(*n, marge*ekn))
            print("below plane {} {} {} {}".format(*n, (L[k]-marge) * ekn))


ShowPlane()

# End of Code
# K̲A̲I̲S̲T̲ M̲O̲L̲S̲I̲M̲
