#!/bin/bash
#SBATCH -J wave-mpi-omp
#SBATCH --partition=edu_sapphire
#SBATCH --time=0:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=500M
#SBATCH --output=output_%j.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s361610@studenti.polito.it
## Exit immediately if a command exits with a non-zero status
set -e

## Caricamento moduli (se presenti nel sistema/cluster)
if command -v module &> /dev/null; then
    module purge
    module load openmpi/4.1.8_gcc11
    # module load ffmpeg   # Decommenta se ffmpeg sul cluster richiede un modulo dedicato
fi

## Configurazione thread OpenMP per ciascun processo}
export OMP_NUM_THREADS=32
## Creazione cartelle di output e pulizia vecchi file
mkdir -p sim1 sim2 sim3
rm -f sim1/*.pgm sim2/*.pgm sim3/*.pgm video_sim1.mp4 video_sim2.mp4 video_sim3.mp4

## Compilazione codice Ibrido (MPI + OpenMP)
echo "Compilazione del codice ibrido MPI + OpenMP..."
mpicc -O3 -fopenmp es_2.c -o es02_exe -lm

## Esecuzione simulazione MPI (3 processi)
echo "Esecuzione di 3 processi MPI con $OMP_NUM_THREADS thread OpenMP ciascuno..."
mpirun -np 3 ./es02_exe


## Binario FFmpeg
FFMPEG=./ffmpeg_tool/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg

## Generazione video MP4 (framerate a 25 fps)
$FFMPEG -y -framerate 25 -i sim1/frame_%05d.pgm -c:v libx264 -pix_fmt yuv420p video_sim1.mp4
$FFMPEG -y -framerate 25 -i sim2/frame_%05d.pgm -c:v libx264 -pix_fmt yuv420p video_sim2.mp4
$FFMPEG -y -framerate 25 -i sim3/frame_%05d.pgm -c:v libx264 -pix_fmt yuv420p video_sim3.mp4

# --- profile ---
echo "Profiling..."
WORKDIR="$PWD"
RESULTS="$PWD/vtune_results"

# Elimina il vecchio zip per non accumulare i dati dei vari nodi
rm -f "$WORKDIR/results.zip"

# Rinomina la vecchia cartella se presente e crea quella nuova pulita
if [ -d "$RESULTS" ]; then
    rm -rf "$RESULTS" 2>/dev/null || mv "$RESULTS" "${RESULTS}_old_$$"
fi
mkdir -p "$RESULTS"

# VTune Path
VTUNE="/share/apps/intel/oneapi/vtune/2025.0/bin64/vtune"


# Profilazione Hotspots su tutti e 3 i processi
echo "Avvio Hotspots sui 3 rank..."
mpirun -np 3 ./vtune_wrapper.sh hotspots ./es02_exe

# Profilazione Threading su tutti e 3 i processi
echo "Avvio Threading sui 3 rank..."
mpirun -np 3 ./vtune_wrapper.sh threading ./es02_exe
# Compress all the results into a zip file (path relativi, non assoluti).
cd "$WORKDIR"
zip -r results.zip vtune_results

echo "Profile results saved in: $WORKDIR/results.zip"


echo "Job MPI + OpenMP completato con successo!"
