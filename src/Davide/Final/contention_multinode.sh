#!/bin/bash
#SBATCH -J ass_3_multinode
#SBATCH --partition=edu_a40
#SBATCH --time=00:15:00
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1G
#SBATCH --output=multinode_%j.txt

## Extends contention_sweep.csv with a "multinode" mode: 3 ranks, 3
## separate nodes, one dedicated GPU per node (--gres=gpu:1 is a PER-NODE
## request, so --nodes=3 --gres=gpu:1 = 3 GPUs total, one per rank).
## MPI's cross-node capability is actually exercised here, unlike every
## other run so far. Build happens once on the batch/lead node; the
## resulting binary is visible from the other allocated nodes via the
## shared home/scratch filesystem, so no per-node build step is needed.

set -e

module purge
module load gcc
module load openmpi

export CUDA_HOME=/share/apps/hpc_sdk/Linux_x86_64/25.1/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "Compiling..."
mpicc -O3 -fopenmp -c assignment_3.c -o assignment_3.o
nvcc -O3 -c wave_color_cuda.cu -o wave_color_cuda.o
mpicc -O3 -fopenmp assignment_3.o wave_color_cuda.o -o assignment_3 \
    -L$CUDA_HOME/lib64 -lcudart -lstdc++

CSV="contention_sweep.csv"
PHASE_HEADER="h2d_s,kernel_s,d2h_s,format_s,write_s"
[ -f "$CSV" ] || echo "mode,rank,scenario,elapsed_s,color_time_s,compute_time_s,omp_threads,$PHASE_HEADER" > "$CSV"

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

parse_phase_line() {
    local line="$1"
    local h2d kernel d2h format write
    h2d=$(grep -oP '(?<=h2d )[0-9.]+' <<< "$line")
    kernel=$(grep -oP '(?<=kernel )[0-9.]+' <<< "$line")
    d2h=$(grep -oP '(?<=d2h )[0-9.]+' <<< "$line")
    format=$(grep -oP '(?<=format )[0-9.]+' <<< "$line")
    write=$(grep -oP '(?<=write )[0-9.]+' <<< "$line")
    echo "$h2d,$kernel,$d2h,$format,$write"
}

echo "=== Multi-node (3 ranks, 3 nodes, 1 dedicated GPU each) ==="
rm -rf ./sim1_ppm ./sim2_ppm ./sim3_ppm
output=$(mpirun -np 3 ./assignment_3 --size 2048 --steps 50 --threads 4)
echo "$output"
lines=()
mapfile -t lines <<< "$output"
n=${#lines[@]}
for ((i = 0; i < n; i++)); do
    line="${lines[$i]}"
    [[ "$line" == Rank\ * ]] || continue
    base=$(parse_line "$line")
    phase="0,0,0,0,0"
    if (( i + 1 < n )) && [[ "${lines[$((i + 1))]}" == *"phase breakdown"* ]]; then
        phase=$(parse_phase_line "${lines[$((i + 1))]}")
    fi
    echo "multinode,$base,$phase" >> "$CSV"
done
rm -rf ./sim1_ppm ./sim2_ppm ./sim3_ppm

echo "Appended 'multinode' rows to $CSV"
