#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>

// Funzione per allocare le matrici dinamicamente
double** matrix_alloc(int n) {
    double** matrix = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        matrix[i] = (double*)malloc(n * sizeof(double));
    }
    return matrix;
}

// Funzione per deallocare
void matrix_free(double** matrix, int n) {
    for (int i = 0; i < n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// Funzione per scrivere il frame su disco
void write_matrix(const char *dir, unsigned char *buffer, int n, int frame_idx) {
    char filename[64];
    snprintf(filename, sizeof(filename), "%s/frame_%05d.pgm", dir, frame_idx);

    FILE *fp = fopen(filename, "wb");
    if (!fp) return;
    fprintf(fp, "P5\n%d %d\n255\n", n, n);
    fwrite(buffer, sizeof(unsigned char), n * n, fp);
    fclose(fp);
}

// La funzione RUN riadattata per accettare parametri dinamici (M, N, dx, dt)
void run_simulation(double** u, double** u_past, double** u_fut, int rank, const char* sim_dir, int i2, int j2, double impulso2, int n_start, int M, int N, double dx, double dt) {
    
    // Costanti fisiche
    double gamma_val = 0.130;
    double c = 0.21;
    double coeff_forw = 1.0 / (gamma_val * dt + 1.0);
    double c2_dt2_dx2 = (c * c * dt * dt) / (dx * dx);
    double gamma_dt = gamma_val * dt;

    // Calcolo automatico dell'intervallo per ottenere ~250 frame (10 sec di video)
    int frame_interv = (N >= 250) ? (N / 250) : 1; 
    int frame_count = 0;

    // Preparazione del buffer e pre-colorazione dei bordi
    unsigned char* buffer = (unsigned char*)malloc(M * M * sizeof(unsigned char));
    double max_val = 0.8;
    double factor = 127.0 / max_val;
    unsigned char border_pixel = (unsigned char)127;

    for (int j = 0; j < M; j++) {
        buffer[0 * M + j] = border_pixel;
        buffer[(M - 1) * M + j] = border_pixel;
    }
    for (int i = 0; i < M; i++) {
        buffer[i * M + 0] = border_pixel;
        buffer[i * M + (M - 1)] = border_pixel;
    }

    double t_start = omp_get_wtime();

    #pragma omp parallel
    {
        for (int t = 0; t < N; t++) {

            // Iniezione impulso ritardato (solo Rank 2)
            #pragma omp single
            {
                if (rank == 2 && t == n_start) {
                    u[i2][j2] += impulso2;
                    u_past[i2][j2] += impulso2;
                }
            }

            // Calcolo Laplaciano e popolamento buffer
            #pragma omp for schedule(static) collapse(2)
            for (int i = 1; i < M - 1; i++) {
                for (int j = 1; j < M - 1; j++) {
                    double laplacian = u[i + 1][j] - 2.0 * u[i][j] + u[i - 1][j] +
                                       u[i][j + 1] - 2.0 * u[i][j] + u[i][j - 1];

                    u_fut[i][j] = coeff_forw * (c2_dt2_dx2 * laplacian + gamma_dt * u[i][j] - u_past[i][j] + 2.0 * u[i][j]);

                    // Converte in pixel SOLO nei frame che verranno salvati
                    if (t % frame_interv == 0) {
                        int value = (int)(127 + factor * u_fut[i][j]);
                        if (value > 255) value = 255;
                        if (value < 0)   value = 0;
                        buffer[i * M + j] = (unsigned char)value;
                    }
                }
            }

            // Scrittura seriale e rotazione puntatori
            #pragma omp single
            {
                if (t % frame_interv == 0) {
                    write_matrix(sim_dir, buffer, M, frame_count);
                    frame_count++;
                }

                double** temp = u_past;
                u_past = u;
                u = u_fut;
                u_fut = temp;
            }
        }
    }

    double t_end = omp_get_wtime();
    printf("Rank %d: Simulazione %s completata in %f sec (%d frame scritti).\n", rank, sim_dir, t_end - t_start, frame_count);
    
    free(buffer);
}

int main(int argc, char *argv[]) {
    int rank, num_procs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (num_procs < 3) {
        if (rank == 0) printf("Errore: Occorrono 3 processi MPI!\n");
        MPI_Finalize();
        return 1;
    }

    // Parametri dinamici della griglia (facilmente modificabili o passabili da argv in futuro)
    int M = 1000;
    int N = 1000;
    double dx = 0.01;
    double dt = 0.01;

    // Configurazione sorgenti per i vari rank
    const double impulso1 = -84.0;
    int i1 = M / 2, j1 = M / 2;

    int i2 = 0, j2 = 0, n_start = -1;
    double impulso2 = 0.0;
    char sim_dir[16];

    if (rank == 0) {
        snprintf(sim_dir, sizeof(sim_dir), "sim1");
    } else if (rank == 1) {
        snprintf(sim_dir, sizeof(sim_dir), "sim2");
        i2 = (3 * M) / 4; j2 = M / 3;
        impulso2 = 36.0; n_start = 0;
    } else if (rank == 2) {
        snprintf(sim_dir, sizeof(sim_dir), "sim3");
        i2 = (3 * M) / 4; j2 = (2 * M) / 3;
        impulso2 = 60.0; n_start = N / 7;
    }

    // Allocazione e inizializzazione a zero
    double** u = matrix_alloc(M);
    double** u_past = matrix_alloc(M);
    double** u_fut = matrix_alloc(M);

    #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            u[i][j] = u_past[i][j] = u_fut[i][j] = 0.0;
        }
    }

    // Impulso iniziale centrale per tutti i rank
    u[i1][j1] = u_past[i1][j1] = impulso1;

    // Impulso secondario per rank 1 (sim2) a t=0
    if (rank == 1 && n_start == 0) {
        u[i2][j2] = u_past[i2][j2] = impulso2;
    }

    // Avvio del calcolo passando tutti i parametri
    run_simulation(u, u_past, u_fut, rank, sim_dir, i2, j2, impulso2, n_start, M, N, dx, dt);

    // Pulizia finale
    matrix_free(u, M);
    matrix_free(u_past, M);
    matrix_free(u_fut, M);

    MPI_Finalize();
    return 0;
}