#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// Mappatura originale e scrittura PGM veloce
void save_pgm(const char *filename, float *u, int M) {
    FILE *f = fopen(filename, "wb");
    if (!f) return;

    fprintf(f, "P5\n%d %d\n255\n", M, M);

    unsigned char *pixels = (unsigned char *)malloc(M * M * sizeof(unsigned char));

    float u_min = -1.5f;
    float u_max = +1.5f;

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

int main() {
    // 1. PARAMETRI AGGIORNATI (Griglia 400x400, N=700, dx=0.01, dt=0.01)
    const int M = 400;
    const int N = 700;
    const float dx = 0.01f;
    const float dt = 0.01f;
    const float impulso = -84.0f;
    const float gamma_val = 0.130f;
    const float c = 0.21f;

    int i0 = M / 2;
    int j0 = M / 2;

    // Allocazione contigua 1D per massima efficienza di cache
    size_t size = M * M * sizeof(float);
    float *u_prev = (float *)malloc(size);
    float *u_curr = (float *)malloc(size);
    float *u_next = (float *)malloc(size);

    if (!u_prev || !u_curr || !u_next) return 1;

    for (int i = 0; i < M * M; i++) {
        u_prev[i] = 0.0f;
        u_curr[i] = 0.0f;
        u_next[i] = 0.0f;
    }

    u_curr[i0 * M + j0] = impulso;
    u_prev[i0 * M + j0] = impulso;

    float alpha = (c * dt / dx) * (c * dt / dx);
    float beta = gamma_val * dt / 2.0f;
    float denom = 1.0f + beta;
    float c1 = 2.0f / denom;
    float c2 = (1.0f - beta) / denom;
    float c3 = alpha / denom;

    char filename[256];

    printf("Simulazione in corso per N=%d passi su griglia %dx%d...\n", N, M, M);

    // 2. PARALLELIZZAZIONE OPENMP (Pool di thread creato all'esterno)
    #pragma omp parallel
    {
        for (int n = 0; n < N; n++) {

            // Salvataggio frame eseguito da un solo thread ad ogni passo
            #pragma omp single
            {
                snprintf(filename, sizeof(filename), "sim/frame_%05d.pgm", n);
                save_pgm(filename, u_curr, M);
            }

            // Calcolo parallelo distribuito tra i thread esistenti
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

            // Pointer swap eseguito da un solo thread con sincronizzazione automatica
            #pragma omp single
            {
                float *temp = u_prev;
                u_prev = u_curr;
                u_curr = u_next;
                u_next = temp;
            }
        }
    }

    printf("Simulazione completata con successo!\n");

    free(u_prev);
    free(u_curr);
    free(u_next);

    return 0;
}