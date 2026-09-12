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
# No Makefile: two compilers, three steps. mpicc already knows its own
# MPI include/lib paths and understands its own -Wl,-rpath linker flags,
# so it compiles and links everything except the CUDA source. nvcc only
# ever sees wave_color_cuda.cu; the two .o files are linked together by
# mpicc, which just needs to be told where libcudart lives.
echo "Compiling..."
mpicc -O3 -fopenmp -c assignment_3.c -o assignment_3.o
nvcc -O3 -c wave_color_cuda.cu -o wave_color_cuda.o
mpicc -O3 -fopenmp assignment_3.o wave_color_cuda.o -o assignment_3 \
    -L$CUDA_HOME/lib64 -lcudart -lstdc++

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
#     (occupancy, memory throughput, ...). Only the first few kernel
#     launches per rank, since full instrumentation per launch is costly.
# Note: unlike nsys, ncu takes the target executable as the first
# positional argument directly, with no "--" separator before it.
mpirun -np $SLURM_NTASKS "$NCU" --set basic --launch-count 5 \
    --export "$RESULTS/kernel_metrics_rank%q{OMPI_COMM_WORLD_RANK}" \
    ./assignment_3 --steps 20

cd "$WORKDIR"
zip -r results.zip nsight_results

echo "Profile results saved in: $WORKDIR/results.zip"
