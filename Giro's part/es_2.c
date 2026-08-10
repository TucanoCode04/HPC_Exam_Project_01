#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>

void save_pgm(const char *filename, float *u, int M) {
    FILE *f = fopen(filename, "wb");
    if (!f) return;

    fprintf(f, "P5\n%d %d\n255\n", M, M);

    unsigned char *pixels = (unsigned char *)malloc(M * M * sizeof(unsigned char));

    // Soglia ottimizzata per far risaltare l'impatto delle due onde smorzate
    float u_min = -0.5f;
    float u_max = +0.5f;

    for (int i = 0; i < M * M; i++) {
        float val = u[i];
        float norm = (val - u_min) / (u_max - u_min);

        if (norm < 0.0f) norm = 0.0f;
        if (norm > 1.0f) norm = 1.0f;

        pixels[i] = (unsigned char)(norm * 255.0f);
    }

    fwrite(pixels, sizeof(unsigned char), M * M, f);
    fclose(f);
    free(pixels);
}

int main(int argc, char *argv[]) {
    int rank, num_procs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (num_procs < 3) {
        if (rank == 0) {
            printf("Errore: Occorrono 3 processi MPI! Esegui con: mpirun -np 3 ...\n");
        }
        MPI_Finalize();
        return 1;
    }

    const int M = 400;
    const int N = 700;
    const float dx = 0.015f;
    const float dt = 0.01f;
    const float gamma_val = 0.130f;
    const float c = 0.21f;

    // Onda 1 (centrale)
    const float impulso1 = -84.0f;
    int i1 = M / 2;
    int j1 = M / 2;

    // Onda 2
    int i2 = 0, j2 = 0;
    float impulso2 = 0.0f;
    int n_start = 0;
    char sim_dir[10];

    if (rank == 0) {
        snprintf(sim_dir, sizeof(sim_dir), "sim1");
    } 
    else if (rank == 1) {
        snprintf(sim_dir, sizeof(sim_dir), "sim2");
        i2 = (3 * M) / 4;
        j2 = M / 3;
        impulso2 = 36.0f;
        n_start = 0;
    } 
    else if (rank == 2) {
        snprintf(sim_dir, sizeof(sim_dir), "sim3");
        i2 = (3 * M) / 4;
        j2 = (2 * M) / 3;
        impulso2 = 60.0f;
        n_start = N / 7;
    }

    size_t size = M * M * sizeof(float);
    float *u_prev = (float *)malloc(size);
    float *u_curr = (float *)malloc(size);
    float *u_next = (float *)malloc(size);

    if (!u_prev || !u_curr || !u_next) {
        MPI_Finalize();
        return 1;
    }

    for (int i = 0; i < M * M; i++) {
        u_prev[i] = 0.0f;
        u_curr[i] = 0.0f;
        u_next[i] = 0.0f;
    }

    // Impulso iniziale Onda 1
    u_curr[i1 * M + j1] = impulso1;
    u_prev[i1 * M + j1] = impulso1;

    // Impulso iniziale Onda 2 (solo sim2)
    if (rank == 1) {
        u_curr[i2 * M + j2] += impulso2;
        u_prev[i2 * M + j2] += impulso2;
    }

    float alpha = (c * dt / dx) * (c * dt / dx);
    float beta = gamma_val * dt / 2.0f;
    float denom = 1.0f + beta;
    float c1 = 2.0f / denom;
    float c2 = (1.0f - beta) / denom;
    float c3 = alpha / denom;

    char filename[256];

    #pragma omp parallel
    {
        for (int n = 0; n < N; n++) {

            #pragma omp single
            {
                // Impulso ritardato Onda 2 (solo sim3 a t = tstart)
                if (rank == 2 && n == n_start) {
                    u_curr[i2 * M + j2] += impulso2;
                    u_prev[i2 * M + j2] += impulso2;
                }

                snprintf(filename, sizeof(filename), "%s/frame_%05d.pgm", sim_dir, n);
                save_pgm(filename, u_curr, M);
            }

            #pragma omp for collapse(2) schedule(static)
            for (int i = 1; i < M - 1; i++) {
                for (int j = 1; j < M - 1; j++) {
                    float laplacian = u_curr[(i + 1) * M + j] + u_curr[(i - 1) * M + j] +
                                      u_curr[i * M + (j + 1)] + u_curr[i * M + (j - 1)] -
                                      4.0f * u_curr[i * M + j];

                    u_next[i * M + j] = c1 * u_curr[i * M + j] 
                                      - c2 * u_prev[i * M + j] 
                                      + c3 * laplacian;
                }
            }

            #pragma omp single
            {
                float *temp = u_prev;
                u_prev = u_curr;
                u_curr = u_next;
                u_next = temp;
            }
        }
    }

    printf("Rank %d: Simulazione %s completata!\n", rank, sim_dir);

    free(u_prev);
    free(u_curr);
    free(u_next);

    MPI_Finalize();
    return 0;
}