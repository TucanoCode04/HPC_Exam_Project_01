#!/bin/bash
#SBATCH -J ass_3_cuda
#SBATCH --partition=edu_a40
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1G
#SBATCH --output=output_%j.txt

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

## --- Nsight profilers (separate from the CUDA toolkit in the SDK layout) ---
NSYS=/share/apps/hpc_sdk/Linux_x86_64/25.1/profilers/Nsight_Systems/bin/nsys
NCU=/share/apps/hpc_sdk/Linux_x86_64/25.1/profilers/Nsight_Compute/ncu

## --- OpenMP configuration ---
# Use the CPUs actually allocated by the scheduler for this task,
# not a fixed value disconnected from the allocation.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=close
export OMP_PLACES=cores
echo "OpenMP threads per rank: $OMP_NUM_THREADS"
echo "MPI ranks: $SLURM_NTASKS"

# --- Code compilation ---
# No Makefile: nvcc drives the whole build. It compiles assignment_3.c
# through its host compiler (gcc), compiles wave_color_cuda.cu itself,
# and links both against the CUDA runtime and MPI in one pass.
echo "Compiling..."
MPI_CFLAGS=$(mpicc --showme:compile)
MPI_LDFLAGS=$(mpicc --showme:link)

nvcc -O3 -Xcompiler -fopenmp $MPI_CFLAGS -c assignment_3.c -o assignment_3.o
nvcc -O3 -c wave_color_cuda.cu -o wave_color_cuda.o
nvcc -O3 -Xcompiler -fopenmp assignment_3.o wave_color_cuda.o -o assignment_3 $MPI_LDFLAGS

## --- Program run ---
echo "Running the program"
rm -rf ./sim1_ppm ./sim2_ppm ./sim3_ppm
mpirun -np $SLURM_NTASKS ./assignment_3
echo "Program finished. Frames saved in sim1_ppm, sim2_ppm, sim3_ppm"

# --- profile ---
echo "Profiling..."
WORKDIR="$PWD"
RESULTS="$PWD/nsight_results"

[ -d "$RESULTS" ] && rm -rf "$RESULTS"
mkdir -p "$RESULTS"

# --- Nsight Systems: whole-application timeline (CPU/GPU overlap, memcpy,
#     kernel launches, OpenMP/MPI activity). One report per rank.
mpirun -np $SLURM_NTASKS "$NSYS" profile \
    --trace=cuda,openmp,osrt \
    --output="$RESULTS/timeline_rank%q{OMPI_COMM_WORLD_RANK}" \
    -- ./assignment_3

# --- Nsight Compute: detailed metrics for the colorize_kernel itself
#     (occupancy, memory throughput, ...). Only rank 0, and only the first
#     few kernel launches, since full instrumentation per launch is costly.
"$NCU" --set basic --launch-count 5 \
    --export "$RESULTS/kernel_metrics" \
    -- ./assignment_3 --steps 20

cd "$WORKDIR"
zip -r results.zip nsight_results

echo "Profile results saved in: $WORKDIR/results.zip"
