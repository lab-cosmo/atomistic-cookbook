#!/bin/bash
#SBATCH --job-name=water_bpnn
#SBATCH --mail-type=FAIL

#SBATCH --output=log.out
#SBATCH --ntasks-per-node=72
#SBATCH --mem=480GB
#SBATCH --time=72:00:00
#SBATCH --nodes=1
##SBATCH --partition=slim
##SBATCH --qos=serial
#SBATCH --get-user-env

set -e

module load intel
module load cmake/3.23.1
#module load git/2.35.2
module load intel-oneapi-mkl
module load intel-oneapi-mpi

module load fftw/3.3.10-mpi-openmp
module load libxc/5.1.7

source /scratch/kellner/cp2k-2024.1/tools/toolchain/install/setup

set -e

for i in $(find . -mindepth 1 -type d); do
    cd $i
    nice -n 72 mpirun /scratch/kellner/cp2k-2024.1/exe/local/cp2k.popt -i in.cp2k | tee cp2k.log
    cd -
done
