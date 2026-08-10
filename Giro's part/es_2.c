#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>

// Funzione per il salvataggio dell'immagine PGM in scala di grigi
void save_pgm(const char *filename, float *u, int M) {
    FILE *f = fopen(filename, "wb");
    if (!f) return;

    fprintf(f, "P5\n%d %d\n255\n", M, M);

    unsigned char *pixels = (unsigned char *)malloc(M * M * sizeof(unsigned char));

    double max_val = 0.8;
    double factor = 127.0 / max_val;

    for (int i = 0; i < M * M; i++) {
        int value = (int)(127 + factor * u[i]);

        if (value > 255) value = 255;
        if (value < 0)   value = 0;

        pixels[i] = (unsigned char)value;
    }

    fwrite(pixels, sizeof(unsigned char), M * M, f);
    fclose(f);
    free(pixels);
}

// Funzione ausiliaria per distribuire l'impulso in modo sferico ed evitare l'anisotropia a quadrato
void apply_impulse(float *u_curr, float *u_prev, int M, int i0, int j0, float amp) {
    for (int r = -2; r <= 2; r++) {
        for (int c = -2; c <= 2; c++) {
            int pi = i0 + r;
            int pj = j0 + c;
            if (pi >= 0 && pi < M && pj >= 0 && pj < M) {
                float dist_sq = (float)(r * r + c * c);
                float weight = expf(-dist_sq / 2.0f);
                u_curr[pi * M + pj] += amp * weight;
                u_prev[pi * M + pj] += amp * weight;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, num_procs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (num_procs < 3) {
        if (rank == 0) {
            printf("Errore: Occorrono 3 processi MPI!\n");
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

    // Applicazione impulso sferico pulito Onda 1
    apply_impulse(u_curr, u_prev, M, i1, j1, impulso1);

    // Applicazione impulso sferico pulito Onda 2 (solo sim2 a t = 0)
    if (rank == 1) {
        apply_impulse(u_curr, u_prev, M, i2, j2, impulso2);
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
                // Inserimento pulito dell'impulso ritardato per sim3
                if (rank == 2 && n == n_start) {
                    apply_impulse(u_curr, u_prev, M, i2, j2, impulso2);
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