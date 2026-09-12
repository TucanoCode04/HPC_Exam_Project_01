#!/bin/bash
#SBATCH -J ass_3_sweep
#SBATCH --partition=edu_a40
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1G
#SBATCH --output=sweep_%j.txt

## edu_a40 nodes have 48 physical cores each (hpcpolito_guide, Table 1.5).
## --ntasks=3 * --cpus-per-task=16 = 48 exactly fills one node, and 16 is
## also the highest thread count the thread sweep below tests for exactly
## that reason (48 cores / 3 ranks).

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

echo "matrix_size,rank,scenario,elapsed_s,color_time_s,compute_time_s,omp_threads" > "$SIZE_CSV"
echo "omp_threads,rank,scenario,elapsed_s,color_time_s,compute_time_s" > "$THREADS_CSV"
echo "block_size,rank,scenario,elapsed_s,color_time_s,compute_time_s" > "$BLOCK_CSV"

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

# Runs one sweep point and appends one CSV row per rank, prefixed with
# the value of the parameter being swept.
run_point() {
    local csv="$1"
    local sweep_value="$2"
    shift 2
    local output
    output=$(mpirun -np $SLURM_NTASKS ./assignment_3 "$@")
    echo "$output"
    while IFS= read -r line; do
        [[ "$line" == Rank\ * ]] || continue
        echo "$sweep_value,$(parse_line "$line")" >> "$csv"
    done <<< "$output"
}

## --- Sweep 1: matrix size, fixed thread count ---
echo "=== Size sweep ==="
for M in 128 256 512 1024 2048; do
    run_point "$SIZE_CSV" "$M" --size "$M" --steps "$SWEEP_STEPS" --threads 4
done

## --- Sweep 2: OpenMP thread count, fixed matrix size ---
echo "=== Thread sweep ==="
for T in 1 2 4 8 16; do
    run_point "$THREADS_CSV" "$T" --size 1024 --steps "$SWEEP_STEPS" --threads "$T"
done

## --- Sweep 3: CUDA block size, fixed matrix size and thread count ---
echo "=== Block size sweep ==="
for B in 32 64 128 256 512 1024; do
    run_point "$BLOCK_CSV" "$B" --size 1024 --steps "$SWEEP_STEPS" \
        --threads 4 --block-size "$B"
done

# The frames written during the sweep are disposable (only the timing
# printout matters here) -- clean them up so they don't get mistaken
# for the real production output of job_assignment3.sh.
rm -rf ./sim1_ppm ./sim2_ppm ./sim3_ppm

echo "Sweep complete. CSVs: $SIZE_CSV, $THREADS_CSV, $BLOCK_CSV"
