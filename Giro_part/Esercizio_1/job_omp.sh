#!/bin/bash
#SBATCH -J wave-omp
#SBATCH --time=0:10:00
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --output=output_%j.txt

## Interrompe lo script in caso di errori
set -e

## Caricamento moduli necessari
module purge
module load gcc
# Se sul cluster ffmpeg richiede un modulo dedicato, decommenta la riga seguente:
# module load ffmpeg

## Assicura l'esistenza della cartella sim e pulisce vecchi output
mkdir -p sim
rm -f sim/*.pgm onda_smorzata.mp4

## Configurazione thread OpenMP basata sulle risorse allocate da Slurm
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

## Compilazione
echo "Compilazione del codice..."
gcc -O3 -fopenmp main.c -o es01_exe -lm

## Esecuzione simulazione
echo "Esecuzione simulazione con $OMP_NUM_THREADS thread OpenMP..."
./es01_exe

## Generazione video
echo "Creazione del video MP4 con ffmpeg..."
ffmpeg -y -framerate 30 -i sim/frame_%05d.pgm -c:v libx264 -pix_fmt yuv420p onda_smorzata.mp4

echo "Job completato con successo!"