#include <errno.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "wave_color_cuda.h"

#define STUDENT_ID 352165

#define GAMMA 0.067
#define WAVE_C 0.59
#define DT 0.1
#define DX 1.0

#define IMPULSE_AMPLITUDE 56.0
#define SIM2_SECOND_AMPLITUDE -35.0
#define SIM3_SECOND_AMPLITUDE 58.0
#define ZERO_BAND 0.03

#define DEFAULT_SIZE 512
#define DEFAULT_STEPS 300
#define DEFAULT_BLOCK_SIZE 256

typedef struct {
    int size;
    int steps;
    int omp_threads;
    int cuda_block_size;
} Config;

typedef struct {
    int i;
    int j;
    int start_step;
    double amplitude;
} WaveSource;

typedef struct {
    int rank;
    char output_dir[32];
    WaveSource sources[2];
    int source_count;
} Scenario;

#define COURANT (WAVE_C * DT / DX)
#define LAMBDA2 (COURANT * COURANT)
#define DAMPING (GAMMA * DT)
#define DENOMINATOR (1.0 + 0.5 * DAMPING)
#define PREVIOUS_WEIGHT (1.0 - 0.5 * DAMPING)

_Static_assert(COURANT <= 0.70710678, "c*dt/dx exceeds the 2D explicit stability limit 1/sqrt(2)");

static void apply_impulse(double *previous, double *current, int m,
                          const WaveSource *source) {
    int idx = source->i * m + source->j;
    previous[idx] += source->amplitude;
    current[idx] += source->amplitude;
}

static void compute_next(int m, const double *previous, const double *current,
                         double *next) {
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i < m - 1; ++i) {
        for (int j = 1; j < m - 1; ++j) {
            int idx = i * m + j;
            double laplacian = current[(i - 1) * m + j] +
                               current[(i + 1) * m + j] +
                               current[i * m + (j - 1)] +
                               current[i * m + (j + 1)] -
                               4.0 * current[idx];
            next[idx] = (2.0 * current[idx] -
                         PREVIOUS_WEIGHT * previous[idx] +
                         LAMBDA2 * laplacian) /
                        DENOMINATOR;
        }
    }

#pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) {
        next[i] = 0.0;
        next[(m - 1) * m + i] = 0.0;
        next[i * m] = 0.0;
        next[i * m + (m - 1)] = 0.0;
    }
}

#define COLOR_SCALE 58.0

static void build_scenario(const Config *cfg, int rank, Scenario *scenario) {
    scenario->rank = rank;
    snprintf(scenario->output_dir, sizeof(scenario->output_dir),
             "sim%d_ppm", rank + 1);

    scenario->source_count = 1;
    scenario->sources[0] = (WaveSource){
        .i = cfg->size / 2, .j = cfg->size / 2,
        .start_step = 0, .amplitude = IMPULSE_AMPLITUDE};

    if (rank == 1) {
        scenario->source_count = 2;
        scenario->sources[1] = (WaveSource){
            .i = (2 * cfg->size) / 3, .j = cfg->size / 4,
            .start_step = 0, .amplitude = SIM2_SECOND_AMPLITUDE};
    } else if (rank == 2) {
        scenario->source_count = 2;
        scenario->sources[1] = (WaveSource){
            .i = cfg->size / 3, .j = cfg->size / 3,
            .start_step = cfg->steps / 7, .amplitude = SIM3_SECOND_AMPLITUDE};
    }
}

static int ensure_output_dir(const char *path) {
    if (mkdir(path, 0755) == 0) {
        return 1;
    }
    return errno == EEXIST;
}

static int write_ppm_frame(const char *dir, int frame, const double *u,
                           int m, CudaColorizer *colorizer) {
    char path[64];
    snprintf(path, sizeof(path), "%s/frame_%05d.ppm", dir, frame);

    if (!cuda_write_ppm(path, u, m, COLOR_SCALE, ZERO_BAND, colorizer)) {
        fprintf(stderr, "Could not write '%s': %s.\n", path,
                cuda_colorizer_last_error());
        return 0;
    }
    return 1;
}

static int run_scenario(const Config *cfg, const Scenario *scenario) {
    if (!ensure_output_dir(scenario->output_dir)) {
        fprintf(stderr, "Rank %d could not create '%s'.\n",
                scenario->rank, scenario->output_dir);
        return 0;
    }
    if (!cuda_colorizer_select_device(scenario->rank)) {
        fprintf(stderr, "Rank %d could not select a CUDA device: %s.\n",
                scenario->rank, cuda_colorizer_last_error());
        return 0;
    }

    CudaColorizer *colorizer = NULL;
    if (!cuda_colorizer_create(&colorizer, cfg->size, cfg->cuda_block_size)) {
        fprintf(stderr, "Rank %d could not create CUDA colorizer: %s.\n",
                scenario->rank, cuda_colorizer_last_error());
        return 0;
    }

    size_t total = (size_t)cfg->size * (size_t)cfg->size;
    double *previous = (double *)calloc(total, sizeof(double));
    double *current = (double *)calloc(total, sizeof(double));
    double *next = (double *)calloc(total, sizeof(double));
    if (previous == NULL || current == NULL || next == NULL) {
        fprintf(stderr, "Rank %d could not allocate simulation matrices.\n",
                scenario->rank);
        cuda_colorizer_destroy(colorizer);
        free(previous);
        free(current);
        free(next);
        return 0;
    }

    for (int s = 0; s < scenario->source_count; ++s) {
        if (scenario->sources[s].start_step == 0) {
            apply_impulse(previous, current, cfg->size, &scenario->sources[s]);
        }
    }

    double start_time = omp_get_wtime();
    int ok = 1;
    for (int frame = 0; frame < cfg->steps; ++frame) {
        for (int s = 0; s < scenario->source_count; ++s) {
            if (frame != 0 && scenario->sources[s].start_step == frame) {
                apply_impulse(previous, current, cfg->size, &scenario->sources[s]);
            }
        }

        if (!write_ppm_frame(scenario->output_dir, frame, current, cfg->size,
                             colorizer)) {
            ok = 0;
            break;
        }

        if (frame + 1 < cfg->steps) {
            compute_next(cfg->size, previous, current, next);
            double *tmp = previous;
            previous = current;
            current = next;
            next = tmp;
        }
    }

    double elapsed = omp_get_wtime() - start_time;
    if (ok) {
        printf("Rank %d: simulation '%s' finished in %.3f s (%d OpenMP threads).\n",
               scenario->rank, scenario->output_dir, elapsed, omp_get_max_threads());
    }

    cuda_colorizer_destroy(colorizer);
    free(previous);
    free(current);
    free(next);
    return ok;
}

static void print_usage(const char *program) {
    fprintf(stderr,
            "Usage: %s [--size M] [--steps N] [--threads T] [--block-size B]\n"
            "Physical parameters and impulse amplitudes are fixed for student %d.\n",
            program, STUDENT_ID);
}

static int parse_int_arg(const char *value, int *out) {
    char *end = NULL;
    long parsed = strtol(value, &end, 10);
    if (end == value || *end != '\0') {
        return 0;
    }
    *out = (int)parsed;
    return 1;
}

static int parse_args(int argc, char **argv, Config *cfg) {
    cfg->size = DEFAULT_SIZE;
    cfg->steps = DEFAULT_STEPS;
    cfg->omp_threads = 0;
    cfg->cuda_block_size = DEFAULT_BLOCK_SIZE;

    for (int i = 1; i < argc; ++i) {
        const char *key = argv[i];
        if (strcmp(key, "--help") == 0 || strcmp(key, "-h") == 0) {
            return -1;
        }
        if (i + 1 >= argc) {
            fprintf(stderr, "Missing value for '%s'.\n", key);
            return 0;
        }

        const char *value = argv[++i];
        int *target = NULL;
        if (strcmp(key, "--size") == 0) target = &cfg->size;
        else if (strcmp(key, "--steps") == 0) target = &cfg->steps;
        else if (strcmp(key, "--threads") == 0) target = &cfg->omp_threads;
        else if (strcmp(key, "--block-size") == 0) target = &cfg->cuda_block_size;
        else {
            fprintf(stderr, "Unknown argument '%s'.\n", key);
            return 0;
        }

        if (!parse_int_arg(value, target)) {
            fprintf(stderr, "Invalid integer '%s' for '%s'.\n", value, key);
            return 0;
        }
    }

    if (cfg->size < 3 || cfg->steps <= 0 || cfg->omp_threads < 0 ||
        cfg->cuda_block_size <= 0) {
        return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    int rank = 0;
    int num_procs = 1;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    Config cfg;
    int parsed = parse_args(argc, argv, &cfg);
    if (parsed <= 0) {
        if (rank == 0) {
            print_usage(argv[0]);
        }
        MPI_Finalize();
        return parsed < 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    if (cfg.omp_threads > 0) {
        omp_set_num_threads(cfg.omp_threads);
    }

    if (num_procs < 3) {
        if (rank == 0) {
            fprintf(stderr, "This program requires at least 3 MPI ranks (got %d).\n",
                    num_procs);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int local_ok = 1;
    if (rank < 3) {
        Scenario scenario;
        build_scenario(&cfg, rank, &scenario);
        local_ok = run_scenario(&cfg, &scenario);
    }

    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Finalize();
    return global_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
