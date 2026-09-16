#!/bin/bash
#SBATCH -J ass_3_v100
#SBATCH --partition=edu_v100
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1G
#SBATCH --output=output_v100_%j.txt

## Cross-hardware sanity check for the phase-breakdown finding (write/fclose
## dominates, not GPU compute): otherwise identical to job_assignment3.sh,
## just pointed at edu_v100 instead of edu_a40 (2 nodes, 32 cores, 4xV100
## each, per hpcpolito_guide Table 1.5 -- confirmed live via
## `sinfo -o "%P %D %G"`). --cpus-per-task=4 still fits fine inside a
## 32-core V100 node. Not meant to replace the A40 results anywhere in the
## report -- just to confirm the I/O-bound conclusion isn't an A40-specific
## artifact.

## --- Error check (exit immediately in case of an error) ---
set -e

## --- Load SLURM modules ---
module purge
module load gcc
module load openmpi

## --- CUDA toolkit (HPC SDK bundle) ---
export CUDA_HOME=/share/apps/hpc_sdk/Linux_x86_64/25.1/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

## --- Nsight Systems (separate from the CUDA toolkit in the SDK layout).
## Nsight Compute is intentionally not used here: this cluster returns
## ERR_NVGPUCTRPERM (GPU performance-counter access is admin-only), which
## is a driver-level restriction no user-side flag can work around.
NSYS=/share/apps/hpc_sdk/Linux_x86_64/25.1/profilers/Nsight_Systems/bin/nsys

## --- OpenMP configuration ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=close
export OMP_PLACES=cores
echo "OpenMP threads per rank: $OMP_NUM_THREADS"
echo "MPI ranks: $SLURM_NTASKS"

# --- Code compilation ---
echo "Compiling..."
mpicc -O3 -fopenmp -c assignment_3.c -o assignment_3.o
nvcc -O3 -c wave_color_cuda.cu -o wave_color_cuda.o
mpicc -O3 -fopenmp assignment_3.o wave_color_cuda.o -o assignment_3 \
    -L$CUDA_HOME/lib64 -lcudart -lstdc++

## --- Program run ---
echo "Running the program"
rm -rf ./sim1_ppm ./sim2_ppm ./sim3_ppm
mpirun -np $SLURM_NTASKS ./assignment_3 --size 400 --steps 400
echo "Program finished. Frames saved in sim1_ppm, sim2_ppm, sim3_ppm"

# --- profile ---
echo "Profiling..."
WORKDIR="$PWD"
RESULTS="$PWD/nsight_results_v100"

[ -d "$RESULTS" ] && rm -rf "$RESULTS"
mkdir -p "$RESULTS"

mpirun -np $SLURM_NTASKS "$NSYS" profile \
    --trace=cuda,openmp,osrt --backtrace=lbr \
    --output="$RESULTS/timeline_rank%q{OMPI_COMM_WORLD_RANK}" \
    -- ./assignment_3 --size 400 --steps 400

cd "$WORKDIR"
zip -r results_v100.zip nsight_results_v100

echo "Profile results saved in: $WORKDIR/results_v100.zip"
