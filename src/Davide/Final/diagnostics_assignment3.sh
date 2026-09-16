#!/bin/bash
#SBATCH -J ass_3_diag
#SBATCH --partition=edu_a40
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1G
#SBATCH --output=diagnostics_%j.txt

## --- Two follow-up questions this job is designed to answer ---
## (1) The nsys OS Runtime Summary shows epoll_wait (~52%) and poll (~25%)
##     dominating wall time. The report guesses this is rank idling on MPI's
##     progress engine, but the original --trace=cuda,openmp,osrt run never
##     asked nsys to actually capture MPI events, so that guess was never
##     checked. This job adds "mpi" to --trace so mpi_event_sum can show
##     real MPI_Allreduce/MPI_Wait time, directly comparable to epoll/poll.
## (2) Section 3.7 attributes the flat OpenMP thread-scaling curve to the
##     5-point stencil being memory-bandwidth-bound, but an equally
##     plausible cause was never ruled out: compute_next forks a fresh
##     OpenMP thread team on every one of 300 calls/rank, and that
##     fork/join overhead alone could explain the same flat curve. VTune
##     Hotspots directly reports time in gomp_team_barrier_wait_end
##     (fork/join+barrier cost) vs. the stencil computation itself -- the
##     same method Assignment 1 already used. Run once at 4 threads (the
##     production setting) and once at 32 threads (the thread-sweep's top
##     end): if fork/join overhead is a real contributor, its share should
##     grow measurably from 4->32 threads even though total wall time does
##     not.

set -e

module purge
module load gcc
module load openmpi

export CUDA_HOME=/share/apps/hpc_sdk/Linux_x86_64/25.1/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

NSYS=/share/apps/hpc_sdk/Linux_x86_64/25.1/profilers/Nsight_Systems/bin/nsys
VTUNE=/share/apps/intel/oneapi/vtune/2025.0/bin64/vtune

export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "Compiling..."
mpicc -O3 -fopenmp -c assignment_3.c -o assignment_3.o
nvcc -O3 -c wave_color_cuda.cu -o wave_color_cuda.o
mpicc -O3 -fopenmp assignment_3.o wave_color_cuda.o -o assignment_3 \
    -L$CUDA_HOME/lib64 -lcudart -lstdc++

RESULTS="$PWD/diagnostics_results"
[ -d "$RESULTS" ] && rm -rf "$RESULTS"
mkdir -p "$RESULTS"

## --- (1) Nsight Systems with MPI tracing added ---
echo "=== nsys with MPI tracing (production params: --size 400 --steps 400) ==="
export OMP_NUM_THREADS=4
rm -rf ./sim1_ppm ./sim2_ppm ./sim3_ppm
mpirun -np $SLURM_NTASKS "$NSYS" profile \
    --trace=cuda,openmp,osrt,mpi --backtrace=lbr \
    --mpi-impl=openmpi \
    --output="$RESULTS/nsys_mpi_rank%q{OMPI_COMM_WORLD_RANK}" \
    -- ./assignment_3 --size 400 --steps 400
rm -rf ./sim1_ppm ./sim2_ppm ./sim3_ppm

## --- (2) VTune Hotspots + Threading, at 4 threads and at 32 threads ---
## One rank (rank 0, sim1) is enough to answer the question -- this isn't a
## timing sweep, just "what is compute_next's time actually made of."
for THREADS in 4 32; do
    echo "=== VTune hotspots+threading, M=4096, ${THREADS} threads ==="
    export OMP_NUM_THREADS=$THREADS
    rm -rf ./sim1_ppm ./sim2_ppm ./sim3_ppm

    "$VTUNE" -collect hotspots \
        -result-dir "$RESULTS/vtune_hotspots_t${THREADS}" -data-limit=0 \
        -- mpirun -np 1 ./assignment_3 --size 4096 --steps 50

    "$VTUNE" -collect threading \
        -result-dir "$RESULTS/vtune_threading_t${THREADS}" -data-limit=0 \
        -- mpirun -np 1 ./assignment_3 --size 4096 --steps 50

    "$VTUNE" -report summary -r "$RESULTS/vtune_hotspots_t${THREADS}" \
        > "$RESULTS/vtune_hotspots_t${THREADS}_summary.txt"
    "$VTUNE" -report summary -r "$RESULTS/vtune_threading_t${THREADS}" \
        > "$RESULTS/vtune_threading_t${THREADS}_summary.txt"
done
rm -rf ./sim1_ppm ./sim2_ppm ./sim3_ppm

cd "$PWD"
zip -r diagnostics_results.zip diagnostics_results
echo "Diagnostics saved in: $PWD/diagnostics_results.zip"
