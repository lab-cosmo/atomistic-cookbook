#!/bin/bash

#SBATCH --job-name=HC_dimer
#SBATCH --output=job_%x.out
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=1G
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --partition=Slim

set -e

spack load quantum-espresso 

# For running locally we set number to fixed value
if [[ -z "${SLURM_NTASKS_PER_NODE}" ]]; then
    export SLURM_NTASKS_PER_NODE=4
fi

mpirun -n ${SLURM_NTASKS_PER_NODE} pw.x -in pw.in | tee pw.out
