#!/bin/bash
#SBATCH -J wave-mpi-omp
#SBATCH --time=0:10:00
#SBATCH --partition=cpu_sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=500M
#SBATCH --output=output_%j.txt

## Interrompe lo script in caso di errore
set -e

## Caricamento moduli (se presenti nel sistema/cluster)
if command -v module &> /dev/null; then
    module purge
    module load openmpi/4.1.8_gcc11
    # module load ffmpeg   # Decommenta se ffmpeg sul cluster richiede un modulo dedicato
fi

## Configurazione thread OpenMP per ciascun processo MPI
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

## Creazione cartelle di output e pulizia vecchi file
mkdir -p sim1 sim2 sim3
rm -f sim1/*.pgm sim2/*.pgm sim3/*.pgm video_sim1.mp4 video_sim2.mp4 video_sim3.mp4

## Compilazione codice Ibrido (MPI + OpenMP)
echo "Compilazione del codice ibrido MPI + OpenMP..."
mpicc -O3 -fopenmp main.c -o es02_exe -lm

## Esecuzione simulazione MPI (3 processi)
echo "Esecuzione di 3 processi MPI con $OMP_NUM_THREADS thread OpenMP ciascuno..."
mpirun -np 3 ./es02_exe

## Generazione dei 3 video MP4 tramite ffmpeg
echo "Generazione video MP4 per le 3 simulazioni..."
ffmpeg -y -framerate 30 -i sim1/frame_%05d.pgm -c:v libx264 -pix_fmt yuv420p video_sim1.mp4
ffmpeg -y -framerate 30 -i sim2/frame_%05d.pgm -c:v libx264 -pix_fmt yuv420p video_sim2.mp4
ffmpeg -y -framerate 30 -i sim3/frame_%05d.pgm -c:v libx264 -pix_fmt yuv420p video_sim3.mp4

echo "Job MPI + OpenMP completato con successo!"