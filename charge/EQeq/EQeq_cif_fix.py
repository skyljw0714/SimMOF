#!/bin/env python
import os
import sys
import math

'''
Raymond Chong, Apr 2018
'''


def main():


    ## get EQeq.cif file input name

    if len(sys.argv) == 2:
        out_name = sys.argv[1]

    else:
	    print("Please provide an input argument ---> './script EQeq_output.cif'")
	    return

    with open(out_name,'r') as f:
        lines = [i[:-1].split() for i in f.readlines()]
     

    mol_name = out_name.split(".")[0]


    print("\n\n\nWRITING NEW CIF for visualization..... \n\n")

    ## open output files
    f_out = open(mol_name + '_EQeq.cif', 'w')


    atom_index = 1
    ## search for final coordinates flag
    for i, line in enumerate(lines):
	## Carry over the headers
        if i < 21:
	        for j in range(0,len(line)):
	            f_out.write(str(line[j])+"\t")
                
                if  j == len(line)-1:
                    f_out.write("\n")
                

        ## Fix atom site labels
        elif len(line) > 1:
            f_out.write(str(line[1])+str(atom_index)+"\t")
            f_out.write(str(line[1])+"\t")
            f_out.write(str(line[2])+"\t")
            f_out.write(str(line[3])+"\t")
            f_out.write(str(line[4])+"\t")
            f_out.write(str(line[5])+"\t")
            f_out.write("\n")
            atom_index += 1

    f_out.write("_end\n")

    f.close()
    f_out.close()

## run script from command line





if __name__  == "__main__":
    main()
