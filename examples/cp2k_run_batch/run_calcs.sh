#! /bin/bash

for i in $(find ./production/ -mindepth 1 -type d); do
    
    cd $i
    cp2k -i in.cp2k

    # within a HPC environment it my be useful to parallelize over stoichiometries.
    # ie. using SLURM and a seperate submit script for each subdirectory.

    cd -
done

