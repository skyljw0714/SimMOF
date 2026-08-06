strides = 1000
max_n_mofs = 50000

rng1 = range(0, max_n_mofs, strides)
rng2 = range(strides, max_n_mofs+1, strides)

for start, end in zip(rng1, rng2):
    with open("make-mof-{}-{}.qsub".format(start, end), "w") as f:
        f.write("#!/bin/bash\n")
        f.write("cd $PBS_O_WORKDIR\n")
        f.write("python make_mofs.py {} {}".format(start, end))
