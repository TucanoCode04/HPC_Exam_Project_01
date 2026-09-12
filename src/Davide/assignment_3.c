#include <errno.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wave_color_cuda.h"
#include <mpi.h>
#include <sys/stat.h>

#define STUDENT_ID 352165
#define DEFAULT_GAMMA 0.067
#define DEFAULT_C 0.59
#define DEFAULT_DT 0.1
#define DEFAULT_DX 1.0
#define DEFAULT_SIZE 512
#define DEFAULT_STEPS 300
#define DEFAULT_AMPLITUDE 56.0
#define SIM2_SECOND_AMPLITUDE -35.0
#define SIM3_SECOND_AMPLITUDE 58.0
#define DEFAULT_ZERO_BAND 0.03

typedef struct {
    double gamma;
    double c;
    double dt;
    double dx;
    int size;
    int steps;
    int impulse_i;
    int impulse_j;
    double amplitude;
    double color_scale;
    double zero_band;
    int threads;
    const char *output_prefix;
} Config;

typedef struct {
    int i;
    int j;
    int start_step;
    double amplitude;
} WaveSource;

typedef struct {
    int id;
    char output_dir[512];
    WaveSource sources[2];
    int source_count;
} Scenario;

static void print_usage(const char *program) {
    fprintf(stderr,
            "Usage: %s [--gamma G] [--c C] [--dt DT] [--dx DX] "
            "[--size M] [--steps N] [--impulse-i I] [--impulse-j J] "
            "[--amplitude A] [--threads T] [--color-scale S] "
            "[--zero-band Z] [--output-prefix PREFIX]\n\n"
            "Defaults are the assignment parameters for student %d. "
            "Output folders are PREFIX1_ppm, PREFIX2_ppm, and PREFIX3_ppm; "
            "default PREFIX is 'sim'.\n",
            program, STUDENT_ID);
}

static int parse_int_arg(const char *value, int *out) {
    char *end = NULL;
    long parsed = strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < INT_MIN || parsed > INT_MAX) {
        return 0;
    }
    *out = (int)parsed;
    return 1;
}

static int parse_double_arg(const char *value, double *out) {
    char *end = NULL;
    double parsed = strtod(value, &end);
    if (end == value || *end != '\0' || !isfinite(parsed)) {
        return 0;
    }
    *out = parsed;
    return 1;
}

static double default_scale(const Config *cfg) {
    double scale = fabs(cfg->amplitude);
    if (fabs(SIM2_SECOND_AMPLITUDE) > scale) scale = fabs(SIM2_SECOND_AMPLITUDE);
    if (fabs(SIM3_SECOND_AMPLITUDE) > scale) scale = fabs(SIM3_SECOND_AMPLITUDE);
    return scale > 0.0 ? scale : 1.0;
}

static int parse_args(int argc, char **argv, Config *cfg) {
    cfg->gamma = DEFAULT_GAMMA;
    cfg->c = DEFAULT_C;
    cfg->dt = DEFAULT_DT;
    cfg->dx = DEFAULT_DX;
    cfg->size = DEFAULT_SIZE;
    cfg->steps = DEFAULT_STEPS;
    cfg->impulse_i = -1;
    cfg->impulse_j = -1;
    cfg->amplitude = DEFAULT_AMPLITUDE;
    cfg->color_scale = 0.0;
    cfg->zero_band = DEFAULT_ZERO_BAND;
    cfg->threads = 0;
    cfg->output_prefix = "sim";

    for (int i = 1; i < argc; ++i) {
        const char *key = argv[i];
        if (strcmp(key, "--help") == 0 || strcmp(key, "-h") == 0) {
            return -1;
        }
        if (i + 1 >= argc) {
            fprintf(stderr, "Missing value for argument '%s'.\n", key);
            return 0;
        }

        const char *value = argv[++i];
        if (strcmp(key, "--gamma") == 0) {
            if (!parse_double_arg(value, &cfg->gamma)) return 0;
        } else if (strcmp(key, "--c") == 0) {
            if (!parse_double_arg(value, &cfg->c)) return 0;
        } else if (strcmp(key, "--dt") == 0) {
            if (!parse_double_arg(value, &cfg->dt)) return 0;
        } else if (strcmp(key, "--dx") == 0) {
            if (!parse_double_arg(value, &cfg->dx)) return 0;
        } else if (strcmp(key, "--size") == 0) {
            if (!parse_int_arg(value, &cfg->size)) return 0;
        } else if (strcmp(key, "--steps") == 0) {
            if (!parse_int_arg(value, &cfg->steps)) return 0;
        } else if (strcmp(key, "--impulse-i") == 0) {
            if (!parse_int_arg(value, &cfg->impulse_i)) return 0;
        } else if (strcmp(key, "--impulse-j") == 0) {
            if (!parse_int_arg(value, &cfg->impulse_j)) return 0;
        } else if (strcmp(key, "--amplitude") == 0) {
            if (!parse_double_arg(value, &cfg->amplitude)) return 0;
        } else if (strcmp(key, "--color-scale") == 0) {
            if (!parse_double_arg(value, &cfg->color_scale)) return 0;
        } else if (strcmp(key, "--zero-band") == 0) {
            if (!parse_double_arg(value, &cfg->zero_band)) return 0;
        } else if (strcmp(key, "--threads") == 0) {
            if (!parse_int_arg(value, &cfg->threads)) return 0;
        } else if (strcmp(key, "--output-prefix") == 0) {
            cfg->output_prefix = value;
        } else {
            fprintf(stderr, "Unknown argument '%s'.\n", key);
            return 0;
        }
    }

    if (cfg->impulse_i < 0) cfg->impulse_i = cfg->size / 2;
    if (cfg->impulse_j < 0) cfg->impulse_j = cfg->size / 2;
    if (cfg->gamma < 0.0 || cfg->c <= 0.0 || cfg->dt <= 0.0 || cfg->dx <= 0.0 ||
        cfg->size < 3 || cfg->steps <= 0 || cfg->impulse_i >= cfg->size ||
        cfg->impulse_j >= cfg->size || cfg->threads < 0 ||
        cfg->color_scale < 0.0 || cfg->zero_band < 0.0 || cfg->zero_band >= 1.0) {
        return 0;
    }

    if (cfg->color_scale == 0.0) {
        cfg->color_scale = default_scale(cfg);
    }

    return 1;
}

static int ensure_output_dir(const char *path) {
    if (mkdir(path, 0755) == 0) {
        return 1;
    }
    return errno == EEXIST;
}

static int write_ppm_frame(const char *dir, int frame, const double *u,
                           const Config *cfg, CudaColorizer *colorizer) {
    char path[512];
    int needed = snprintf(path, sizeof(path), "%s/frame_%05d.ppm", dir, frame);
    if (needed < 0 || (size_t)needed >= sizeof(path)) {
        fprintf(stderr, "Output path is too long for frame %d.\n", frame);
        return 0;
    }

    if (!cuda_write_ppm(path, u, cfg->size, cfg->color_scale,
                        cfg->zero_band, colorizer)) {
        fprintf(stderr, "Could not write CUDA-colored PPM '%s': %s.\n",
                path, cuda_colorizer_last_error());
        return 0;
    }
    return 1;
}

static void apply_impulse(double *previous, double *current, int m,
                          const WaveSource *source) {
    if (source->i < 0 || source->i >= m || source->j < 0 || source->j >= m) {
        return;
    }
    int idx = source->i * m + source->j;
    previous[idx] += source->amplitude;
    current[idx] += source->amplitude;
}

static void compute_next(const Config *cfg, const double *previous,
                         const double *current, double *next) {
    const int m = cfg->size;
    const double courant = cfg->c * cfg->dt / cfg->dx;
    const double lambda2 = courant * courant;
    const double damping = cfg->gamma * cfg->dt;
    const double denominator = 1.0 + 0.5 * damping;
    const double previous_weight = 1.0 - 0.5 * damping;

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
                         previous_weight * previous[idx] +
                         lambda2 * laplacian) /
                        denominator;
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

static int build_scenario(const Config *cfg, int rank, Scenario *scenario) {
    if (rank < 0 || rank > 2) {
        return 0;
    }

    scenario->id = rank + 1;
    int needed = snprintf(scenario->output_dir, sizeof(scenario->output_dir),
                          "%s%d_ppm", cfg->output_prefix, scenario->id);
    if (needed < 0 || (size_t)needed >= sizeof(scenario->output_dir)) {
        fprintf(stderr, "Output directory prefix is too long.\n");
        return 0;
    }

    scenario->source_count = 1;
    scenario->sources[0].i = cfg->impulse_i;
    scenario->sources[0].j = cfg->impulse_j;
    scenario->sources[0].start_step = 0;
    scenario->sources[0].amplitude = cfg->amplitude;

    if (rank == 1) {
        scenario->source_count = 2;
        scenario->sources[1].i = (2 * cfg->size) / 3;
        scenario->sources[1].j = cfg->size / 4;
        scenario->sources[1].start_step = 0;
        scenario->sources[1].amplitude = SIM2_SECOND_AMPLITUDE;
    } else if (rank == 2) {
        scenario->source_count = 2;
        scenario->sources[1].i = cfg->size / 3;
        scenario->sources[1].j = cfg->size / 3;
        scenario->sources[1].start_step = cfg->steps / 7;
        scenario->sources[1].amplitude = SIM3_SECOND_AMPLITUDE;
    }

    return 1;
}

static int run_scenario(const Config *cfg, const Scenario *scenario, int rank) {
    if (!ensure_output_dir(scenario->output_dir)) {
        fprintf(stderr, "Rank %d could not create output directory '%s'.\n",
                rank, scenario->output_dir);
        return 0;
    }
    if (!cuda_colorizer_select_device(rank)) {
        fprintf(stderr, "Rank %d could not select a CUDA device: %s.\n",
                rank, cuda_colorizer_last_error());
        return 0;
    }

    CudaColorizer *colorizer = NULL;
    if (!cuda_colorizer_create(&colorizer, cfg->size)) {
        fprintf(stderr, "Rank %d could not create CUDA colorizer: %s.\n",
                rank, cuda_colorizer_last_error());
        return 0;
    }

    size_t total = (size_t)cfg->size * (size_t)cfg->size;
    double *previous = (double *)calloc(total, sizeof(double));
    double *current = (double *)calloc(total, sizeof(double));
    double *next = (double *)calloc(total, sizeof(double));
    if (previous == NULL || current == NULL || next == NULL) {
        fprintf(stderr, "Rank %d could not allocate simulation matrices.\n", rank);
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
            if (scenario->sources[s].start_step == frame && frame != 0) {
                apply_impulse(previous, current, cfg->size, &scenario->sources[s]);
            }
        }

        if (!write_ppm_frame(scenario->output_dir, frame, current, cfg, colorizer)) {
            ok = 0;
            break;
        }

        if (frame + 1 < cfg->steps) {
            compute_next(cfg, previous, current, next);
            double *tmp = previous;
            previous = current;
            current = next;
            next = tmp;
        }
    }

    double elapsed = omp_get_wtime() - start_time;
    if (ok) {
        printf("Rank %d generated CUDA-colored simulation %d in '%s' with %d OpenMP thread(s) in %.3f seconds.\n",
               rank, scenario->id, scenario->output_dir, omp_get_max_threads(), elapsed);
    }

    cuda_colorizer_destroy(colorizer);
    free(previous);
    free(current);
    free(next);
    return ok;
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

    if (cfg.threads > 0) {
        omp_set_num_threads(cfg.threads);
    }

    if (rank == 0) {
        double stability = cfg.c * cfg.dt / cfg.dx;
        double limit = 1.0 / sqrt(2.0);
        if (stability > limit) {
            fprintf(stderr,
                    "Warning: c * dt / dx = %.6f is above the common 2D explicit "
                    "stability limit %.6f.\n",
                    stability, limit);
        }
    }

    if (num_procs < 3) {
        if (rank == 0) {
            fprintf(stderr, "Part 3 requires at least 3 MPI processes.\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int local_ok = 1;
    if (rank < 3) {
        Scenario scenario;
        local_ok = build_scenario(&cfg, rank, &scenario) &&
                   run_scenario(&cfg, &scenario, rank);
    }

    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Finalize();
    return global_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
