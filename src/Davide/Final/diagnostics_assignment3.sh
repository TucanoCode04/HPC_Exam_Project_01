#!/bin/bash
#SBATCH -J ass_3_diag
#SBATCH --partition=edu_a40
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1G
#SBATCH --output=diagnostics_%j.txt

## --- Two follow-up questions this job is designed to answer ---
## (1) Section 3.7 attributes the flat OpenMP thread-scaling curve to the
##     5-point stencil being memory-bandwidth-bound, but an equally
##     plausible cause was never ruled out: compute_next forks a fresh
##     OpenMP thread team on every one of 300 calls/rank, and that
##     fork/join overhead alone could explain the same flat curve. VTune
##     Hotspots directly reports time in gomp_team_barrier_wait_end
##     (fork/join+barrier cost) vs. the stencil computation itself -- the
##     same method Assignment 1 already used, and already proven to work
##     on this cluster. Run once at 4 threads (production setting) and
##     once at 32 threads (thread-sweep's top end): if fork/join overhead
##     is a real contributor, its share should grow measurably from
##     4->32 threads even though total wall time does not. THIS RUNS
##     FIRST -- it's the safe, proven part, and must not be starved by
##     the riskier part below.
## (2) The nsys OS Runtime Summary shows epoll_wait/poll dominating wall
##     time, guessed (not confirmed) to be MPI progress-engine idling.
##     Adding "mpi" to --trace would let mpi_event_sum check this
##     directly -- BUT a first attempt at this (previous job 1941014)
##     hung: only 2 of 3 ranks ever printed their "finished" line, so
##     rank 3 almost certainly deadlocked inside nsys's MPI
##     instrumentation at the final MPI_Allreduce, and the whole 30-min
##     job budget burned waiting on it before VTune ever ran. Wrapped in
##     `timeout` this time so a repeat hang costs 3 minutes, not the
##     whole job -- and it now runs LAST, after the part we actually
##     need is already safely on disk.
##
## Resource note: previously requested 96 cores (copied from
## sweep_assignment3.sh out of habit) though nothing here needs more than
## 32 at once (one VTune solo rank) or 3x4=12 (the MPI-trace step) --
## that's why it sat queued so long. 3x12=36 is enough with headroom and
## should schedule far faster.

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

## --- (1) VTune Hotspots + Threading, at 4 threads and at 32 threads ---
## Runs FIRST: proven safe on this cluster (assignments 1 and 2 both used
## it successfully), and is the higher-value of the two questions. One
## rank (np=1, sim1 alone) is enough -- this isn't a timing sweep, just
## "what is compute_next's time actually made of."
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

    echo "=== VTune t${THREADS} done, results on disk ==="
done
rm -rf ./sim1_ppm ./sim2_ppm ./sim3_ppm

## --- (2) Nsight Systems with MPI tracing added -- bounded attempt ---
## Runs LAST, capped at 3 minutes. If it hangs again like job 1941014
## did, `timeout` kills it and the script continues to the zip step
## regardless -- we keep whatever VTune already produced either way.
echo "=== nsys with MPI tracing (bounded to 180s; may fail, that's OK) ==="
export OMP_NUM_THREADS=4
rm -rf ./sim1_ppm ./sim2_ppm ./sim3_ppm
timeout 180 mpirun -np $SLURM_NTASKS "$NSYS" profile \
    --trace=cuda,openmp,osrt,mpi --backtrace=lbr \
    --mpi-impl=openmpi \
    --output="$RESULTS/nsys_mpi_rank%q{OMPI_COMM_WORLD_RANK}" \
    -- ./assignment_3 --size 400 --steps 400 \
    && echo "=== nsys MPI trace completed ===" \
    || echo "=== nsys MPI trace timed out/failed (>180s) -- skipping, VTune results are unaffected ==="
rm -rf ./sim1_ppm ./sim2_ppm ./sim3_ppm

cd "$PWD"
zip -r diagnostics_results.zip diagnostics_results
echo "Diagnostics saved in: $PWD/diagnostics_results.zip"
