#!/bin/bash

#SBATCH --job-name=wave_p3_cuda
#SBATCH --partition=edu_a40
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=00:10:00
#SBATCH --output=wave_p3_cuda_%j.out
#SBATCH --error=wave_p3_cuda_%j.err

set -e

module purge
module load gcc
module load openmpi

# CUDA 12.6
export PATH=/share/apps/hpc_sdk/Linux_x86_64/25.1/cuda/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/hpc_sdk/Linux_x86_64/25.1/cuda/lib64:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=close
export OMP_PLACES=cores

cd ~/Test

make clean
make part3

make run-part3 \
    MPI_NP=$SLURM_NTASKS \
    THREADS=$SLURM_CPUS_PER_TASK