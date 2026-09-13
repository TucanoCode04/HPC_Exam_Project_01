#!/bin/bash
#SBATCH -J ass_3_sweep
#SBATCH --partition=edu_a40
#SBATCH --time=00:55:00
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1G
#SBATCH --output=sweep_%j.txt

## edu_a40 nodes have 96 LOGICAL cores each (confirmed via
## `sinfo -p edu_a40 -N -o "%N %c %X %Y %Z"`: 2 sockets * 24 cores *
## 2 threads/core = 96 -- the hpcpolito_guide's "48" is physical cores
## only, same SMT pattern as edu_sapphire). --ntasks=3 * --cpus-per-task=32
## = 96 exactly fills one node, and 32 is also the highest thread count
## the thread sweep below tests for exactly that reason (96 cores / 3 ranks).

set -e

## --- Load SLURM modules ---
module purge
module load gcc
module load openmpi

## --- CUDA toolkit (HPC SDK bundle) ---
export CUDA_HOME=/share/apps/hpc_sdk/Linux_x86_64/25.1/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export OMP_PROC_BIND=close
export OMP_PLACES=cores

# --- Build (same three steps as job_assignment3.sh) ---
echo "Compiling..."
mpicc -O3 -fopenmp -c assignment_3.c -o assignment_3.o
nvcc -O3 -c wave_color_cuda.cu -o wave_color_cuda.o
mpicc -O3 -fopenmp assignment_3.o wave_color_cuda.o -o assignment_3 \
    -L$CUDA_HOME/lib64 -lcudart -lstdc++

# --- Reduced step count for the sweep: enough to get a stable timing
#     average, short enough that dozens of runs finish in one job.
SWEEP_STEPS=50

SIZE_CSV="size_sweep.csv"
THREADS_CSV="thread_sweep.csv"
BLOCK_CSV="blocksize_sweep.csv"
CONTENTION_CSV="contention_sweep.csv"

echo "matrix_size,rank,scenario,elapsed_s,color_time_s,compute_time_s,omp_threads" > "$SIZE_CSV"
echo "omp_threads,rank,scenario,elapsed_s,color_time_s,compute_time_s,reported_threads" > "$THREADS_CSV"
echo "block_size,rank,scenario,elapsed_s,color_time_s,compute_time_s,omp_threads" > "$BLOCK_CSV"
echo "mode,rank,scenario,elapsed_s,color_time_s,compute_time_s,omp_threads" > "$CONTENTION_CSV"

# Pulls the numbers out of one "Rank N: simulation '...' finished in ..."
# line and prints them comma-separated, matching write_ppm_frame's log
# format in assignment_3.c exactly.
parse_line() {
    local line="$1"
    local rank scenario elapsed color compute threads
    rank=$(grep -oP '(?<=^Rank )\d+' <<< "$line")
    scenario=$(grep -oP "(?<=simulation ')[^']+" <<< "$line")
    elapsed=$(grep -oP '(?<=finished in )[0-9.]+' <<< "$line")
    color=$(grep -oP '(?<=colorize\+write )[0-9.]+' <<< "$line")
    compute=$(grep -oP '(?<=compute )[0-9.]+' <<< "$line")
    threads=$(grep -oP '[0-9]+(?= OpenMP threads)' <<< "$line")
    echo "$rank,$scenario,$elapsed,$color,$compute,$threads"
}

# Runs one sweep point under a given MPI process count and appends one CSV
# row per rank, prefixed with the value of the parameter being swept.
run_point_np() {
    local csv="$1"
    local np="$2"
    local sweep_value="$3"
    shift 3
    local output
    output=$(mpirun -np "$np" ./assignment_3 "$@")
    echo "$output"
    while IFS= read -r line; do
        [[ "$line" == Rank\ * ]] || continue
        echo "$sweep_value,$(parse_line "$line")" >> "$csv"
    done <<< "$output"
}

# Same as above, always at the job's full rank count ($SLURM_NTASKS = 3).
run_point() {
    local csv="$1"
    local sweep_value="$2"
    shift 2
    run_point_np "$csv" "$SLURM_NTASKS" "$sweep_value" "$@"
}

## --- Sweep 1: matrix size, fixed thread count ---
# 4096 is one doubling past the previous max (2048), not a jump to
# assignment_1-style M~15000-20000: PPM is ASCII P3 (up to ~12 bytes/pixel
# as text) vs. assignment_1's binary P5 PGM (1 byte/pixel), so frame I/O
# here is far heavier per pixel. A 4096x4096 frame is already tens of MB;
# going another 4x further would push single sweep points into minutes
# and tens of GB of scratch usage for little extra insight.
echo "=== Size sweep ==="
for M in 128 256 512 1024 2048 4096; do
    run_point "$SIZE_CSV" "$M" --size "$M" --steps "$SWEEP_STEPS" --threads 4
done

## --- Sweep 2: OpenMP thread count, fixed matrix size ---
# Fixed size bumped to 4096 (was 1024): at 1024, compute_next only costs
# ~0.1s regardless of thread count, so thread-team overhead swamped any
# real scaling signal. 4096 gives OpenMP enough actual work to show one.
# Range extended to 32 threads (was capped at 16) now that edu_a40 is
# confirmed to have 96 logical cores, not 48 -- see header comment.
echo "=== Thread sweep ==="
for T in 1 2 4 8 16 32; do
    run_point "$THREADS_CSV" "$T" --size 4096 --steps "$SWEEP_STEPS" --threads "$T"
done

## --- Sweep 3: CUDA block size, fixed matrix size and thread count ---
echo "=== Block size sweep ==="
for B in 32 64 128 256 512 1024; do
    run_point "$BLOCK_CSV" "$B" --size 4096 --steps "$SWEEP_STEPS" \
        --threads 4 --block-size "$B"
done

## --- Sweep 4: GPU contention -- solo (1 rank, exclusive GPU) vs shared
## (3 concurrent ranks on one GPU), identical workload otherwise. Isolates
## how much of the production numbers is genuine cost vs. contention for
## the one physical GPU all 3 ranks share. "solo" only produces a sim1 row
## (rank 0's scenario); compare it against sim1's row in "shared".
echo "=== GPU contention sweep ==="
run_point_np "$CONTENTION_CSV" 1 "solo" --size 2048 --steps "$SWEEP_STEPS" --threads 4
run_point_np "$CONTENTION_CSV" 3 "shared" --size 2048 --steps "$SWEEP_STEPS" --threads 4

# The frames written during the sweep are disposable (only the timing
# printout matters here) -- clean them up so they don't get mistaken
# for the real production output of job_assignment3.sh.
rm -rf ./sim1_ppm ./sim2_ppm ./sim3_ppm

echo "Sweep complete. CSVs: $SIZE_CSV, $THREADS_CSV, $BLOCK_CSV, $CONTENTION_CSV"
