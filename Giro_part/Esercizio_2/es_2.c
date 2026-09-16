#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>

// Allocazione contigua per ottimizzare cache e comunicazioni MPI
double** matrix_alloc(int n) {
    double** matrix = (double**)malloc(n * sizeof(double*));
    matrix[0] = (double*)malloc(n * n * sizeof(double));
    for (int i = 1; i < n; i++) {
        matrix[i] = matrix[i - 1] + n;
    }
    return matrix;
}

void matrix_free(double** matrix, int n) {
    free(matrix[0]);
    free(matrix);
}

void write_matrix(const char *dir, unsigned char *buffer, int n, int frame_idx) {
    char filename[64];
    snprintf(filename, sizeof(filename), "%s/frame_%05d.pgm", dir, frame_idx);
    FILE *fp = fopen(filename, "wb");
    if (!fp) return;
    fprintf(fp, "P5\n%d %d\n255\n", n, n);
    fwrite(buffer, sizeof(unsigned char), n * n, fp);
    fclose(fp);
}

void run_simulation(double** u, double** u_past, double** u_fut, int rank, const char* sim_dir, int i2, int j2, double impulso2, int n_start, int M, int N, double dx, double dt) {
    double gamma_val = 0.130;
    double c = 0.21;
    double coeff_forw = 1.0 / (gamma_val * dt + 1.0);
    double c2_dt2_dx2 = (c * c * dt * dt) / (dx * dx);
    double gamma_dt = gamma_val * dt;

    int frame_interv = 1;
    int frame_count = 0;

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
            if (rank == 2 && t == n_start) {
                #pragma omp single
                {
                    u[i2][j2] += impulso2;
                    u_past[i2][j2] += impulso2;
                }
            }

          
            #pragma omp for schedule(static)
            for (int i = 1; i < M - 1; i++) {
                for (int j = 1; j < M - 1; j++) {
                    double laplacian = u[i + 1][j] - 2.0 * u[i][j] + u[i - 1][j] +
                                       u[i][j + 1] - 2.0 * u[i][j] + u[i][j - 1];

                    u_fut[i][j] = coeff_forw * (c2_dt2_dx2 * laplacian + gamma_dt * u[i][j] - u_past[i][j] + 2.0 * u[i][j]);
                    if (t % frame_interv == 0) {
                        int value = (int)(127 + factor * u_fut[i][j]);
                        if (value > 255) value = 255;
                        if (value < 0)   value = 0;
                        buffer[i * M + j] = (unsigned char)value;
                    }
                }
            }

            // Questo single genera una barriera implicita necessaria alla fine del calcolo spaziale:
            // si assicura che tutti i thread abbiano completato la matrice u_fut prima del file I/O 
            // e del cambio dei puntatori.
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
    double local_time = t_end - t_start;

    // Tutti i processi stampano il loro tempo in modo indipendente 
    printf("Rank %d (M=%d): Completata in %f sec (%d frame).\n", rank, M, local_time, frame_count);
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

    int M = 1000; // Dichiara la variabile e imposta il default

    if (argc > 1) {
        M = atoi(argv[1]); // Sovrascrive il valore se passato da terminale
    }
    int N = 1000;
    double dx = 0.01;
    double dt = 0.01;

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

    double** u = matrix_alloc(M);
    double** u_past = matrix_alloc(M);
    double** u_fut = matrix_alloc(M);

  
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            u[i][j] = u_past[i][j] = u_fut[i][j] = 0.0;
        }
    }

    u[i1][j1] = u_past[i1][j1] = impulso1;
    if (rank == 1 && n_start == 0) u[i2][j2] = u_past[i2][j2] = impulso2;

    run_simulation(u, u_past, u_fut, rank, sim_dir, i2, j2, impulso2, n_start, M, N, dx, dt);

    matrix_free(u, M);
    matrix_free(u_past, M);
    matrix_free(u_fut, M);

    MPI_Finalize();
    return 0;
}
