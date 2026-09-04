#!/bin/bash
#SBATCH -J ass_1
#SBATCH --partition=edu_sapphire
#SBATCH --time=0:05:00
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --mail-user=leonardo.riola.13@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=output_%j.txt        

## --- Error check (exit immediately in case of an error) ---
set -e

## --- Load SLURM module ---
module purge
module load gcc

## --- OpenMP configuration ---
# Usa il numero di CPU effettivamente allocate dallo scheduler,
# invece di un valore fisso scollegato dall'allocazione.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "Threads: $OMP_NUM_THREADS"

# --- Code compilation ---
gcc -O2 -g -fopenmp -fno-omit-frame-pointer assign_1v3.c -o assign_1v3

## --- Program run ---
echo "Running the program"
mkdir -p ./sim          # crea la cartella se non esiste
rm -f ./sim/*            # rimuove i risultati precedenti (sicuro anche se vuota)
./assign_1v3
echo "Program finished. Frame saved in directory ./sim"

# --- profile ---
echo "Profiling..."
WORKDIR="$PWD"
RESULTS="$PWD/vtune_results"

# Se esiste una versione precedente dei risultati, la elimina.
[ -d "$RESULTS" ] && rm -rf "$RESULTS"
mkdir -p "$RESULTS"

# VTune Path
VTUNE="/share/apps/intel/oneapi/vtune/2025.0/bin64/vtune"

# --- Profiling execution ---
# "--" separa le opzioni di vtune dall'eseguibile target: buona pratica,
# evita ambiguità se in futuro assign_1 dovesse ricevere argomenti.
"$VTUNE" -collect hotspots  -result-dir "$RESULTS/hotspots"  -- ./assign_1v3
"$VTUNE" -collect threading -result-dir "$RESULTS/threading" -- ./assign_1v3

# Compress all the results into a zip file (path relativi, non assoluti).
cd "$WORKDIR"
zip -r results.zip vtune_results

echo "Profile results saved in: $WORKDIR/results.zip"