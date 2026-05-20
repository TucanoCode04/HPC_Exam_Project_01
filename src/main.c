#include <errno.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#define MKDIR(path) _mkdir(path)
#else
#include <sys/stat.h>
#define MKDIR(path) mkdir(path, 0755)
#endif

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
    double pgm_scale;
    int threads;
    const char *output_dir;
} Config;

static void print_usage(const char *program) {
    fprintf(stderr,
            "Usage: %s --gamma G --c C --dt DT --dx DX --size M --steps N "
            "[--impulse-i I --impulse-j J] [--amplitude A] [--threads T] "
            "[--scale S] [--output DIR]\n",
            program);
}

static int parse_int_arg(const char *value, int *out) {
    char *end = NULL;
    long parsed = strtol(value, &end, 10);
    if (end == value || *end != '\0') {
        return 0;
    }
    if (parsed < -2147483647L || parsed > 2147483647L) {
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

static int parse_args(int argc, char **argv, Config *cfg) {
    cfg->gamma = -1.0;
    cfg->c = -1.0;
    cfg->dt = -1.0;
    cfg->dx = -1.0;
    cfg->size = 0;
    cfg->steps = 0;
    cfg->impulse_i = -1;
    cfg->impulse_j = -1;
    cfg->amplitude = 1.0;
    cfg->pgm_scale = 0.0;
    cfg->threads = 0;
    cfg->output_dir = "sim";

    for (int i = 1; i < argc; ++i) {
        if (i + 1 >= argc) {
            fprintf(stderr, "Missing value for argument '%s'.\n", argv[i]);
            return 0;
        }

        const char *key = argv[i];
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
        } else if (strcmp(key, "--scale") == 0) {
            if (!parse_double_arg(value, &cfg->pgm_scale)) return 0;
        } else if (strcmp(key, "--threads") == 0) {
            if (!parse_int_arg(value, &cfg->threads)) return 0;
        } else if (strcmp(key, "--output") == 0) {
            cfg->output_dir = value;
        } else {
            fprintf(stderr, "Unknown argument '%s'.\n", key);
            return 0;
        }
    }

    if (cfg->gamma < 0.0 || cfg->c <= 0.0 || cfg->dt <= 0.0 || cfg->dx <= 0.0 ||
        cfg->size < 3 || cfg->steps <= 0 ||
        cfg->impulse_j >= cfg->size || cfg->threads < 0 ||
        cfg->pgm_scale < 0.0) {
        return 0;
    }

    if (cfg->impulse_i < 0) {
        cfg->impulse_i = cfg->size / 2;
    }
    if (cfg->impulse_j < 0) {
        cfg->impulse_j = cfg->size / 2;
    }
    if (cfg->impulse_i >= cfg->size || cfg->impulse_j >= cfg->size) {
        return 0;
    }

    if (cfg->pgm_scale == 0.0) {
        cfg->pgm_scale = fabs(cfg->amplitude);
        if (cfg->pgm_scale == 0.0) {
            cfg->pgm_scale = 1.0;
        }
    }

    return 1;
}

static int ensure_output_dir(const char *path) {
    if (MKDIR(path) == 0) {
        return 1;
    }
    return errno == EEXIST;
}

static unsigned char scale_to_byte(double value, double max_abs) {
    if (max_abs <= 0.0) {
        return 127;
    }

    double normalized = 0.5 + 0.5 * (value / max_abs);
    if (normalized < 0.0) normalized = 0.0;
    if (normalized > 1.0) normalized = 1.0;
    return (unsigned char)lrint(normalized * 255.0);
}

static int write_pgm(const char *dir, int frame, const double *u, int m,
                     double pgm_scale) {
    char path[512];
    snprintf(path, sizeof(path), "%s/frame_%05d.pgm", dir, frame);

    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        fprintf(stderr, "Could not open '%s' for writing.\n", path);
        return 0;
    }

    int total = m * m;

    fprintf(file, "P5\n%d %d\n255\n", m, m);

    unsigned char *pixels = (unsigned char *)malloc((size_t)total);
    if (pixels == NULL) {
        fclose(file);
        fprintf(stderr, "Could not allocate PGM pixel buffer.\n");
        return 0;
    }

    for (int idx = 0; idx < total; ++idx) {
        pixels[idx] = scale_to_byte(u[idx], pgm_scale);
    }

    size_t written = fwrite(pixels, sizeof(unsigned char), (size_t)total, file);
    free(pixels);
    fclose(file);

    if (written != (size_t)total) {
        fprintf(stderr, "Incomplete write for frame %d.\n", frame);
        return 0;
    }

    return 1;
}

static void compute_next(const Config *cfg, const double *previous,
                         const double *current, double *next) {
    const int m = cfg->size;
    const double lambda2 = (cfg->c * cfg->dt / cfg->dx) *
                           (cfg->c * cfg->dt / cfg->dx);
    const double damping = cfg->gamma * cfg->dt;
    const double denominator = 1.0 + 0.5 * damping;
    const double previous_weight = 1.0 - 0.5 * damping;

#pragma omp parallel for schedule(static)
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

int main(int argc, char **argv) {
    Config cfg;
    if (!parse_args(argc, argv, &cfg)) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (cfg.threads > 0) {
        omp_set_num_threads(cfg.threads);
    }

    double stability = cfg.c * cfg.dt / cfg.dx;
    if (stability > 1.0 / sqrt(2.0)) {
        fprintf(stderr,
                "Warning: c * dt / dx = %.6f is above the common 2D explicit "
                "stability limit %.6f.\n",
                stability, 1.0 / sqrt(2.0));
    }

    if (!ensure_output_dir(cfg.output_dir)) {
        fprintf(stderr, "Could not create output directory '%s'.\n", cfg.output_dir);
        return EXIT_FAILURE;
    }

    int total = cfg.size * cfg.size;
    double *previous = (double *)calloc((size_t)total, sizeof(double));
    double *current = (double *)calloc((size_t)total, sizeof(double));
    double *next = (double *)calloc((size_t)total, sizeof(double));

    if (previous == NULL || current == NULL || next == NULL) {
        fprintf(stderr, "Could not allocate simulation matrices.\n");
        free(previous);
        free(current);
        free(next);
        return EXIT_FAILURE;
    }

    current[cfg.impulse_i * cfg.size + cfg.impulse_j] = cfg.amplitude;
    previous[cfg.impulse_i * cfg.size + cfg.impulse_j] = cfg.amplitude;

    double start_time = omp_get_wtime();

    if (!write_pgm(cfg.output_dir, 0, current, cfg.size, cfg.pgm_scale)) {
        free(previous);
        free(current);
        free(next);
        return EXIT_FAILURE;
    }

    for (int frame = 1; frame < cfg.steps; ++frame) {
        memset(next, 0, (size_t)total * sizeof(double));
        compute_next(&cfg, previous, current, next);

        if (!write_pgm(cfg.output_dir, frame, next, cfg.size, cfg.pgm_scale)) {
            free(previous);
            free(current);
            free(next);
            return EXIT_FAILURE;
        }

        double *tmp = previous;
        previous = current;
        current = next;
        next = tmp;
    }

    double elapsed = omp_get_wtime() - start_time;
    printf("Generated %d frames in '%s' using %d OpenMP thread(s) in %.3f seconds.\n",
           cfg.steps, cfg.output_dir, omp_get_max_threads(), elapsed);

    free(previous);
    free(current);
    free(next);
    return EXIT_SUCCESS;
}
