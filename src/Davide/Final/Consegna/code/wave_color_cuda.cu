#include "wave_color_cuda.h"

#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Used by cuda_write_ppm's per-frame peak-finding reduction, below. */
struct AbsValue {
    __host__ __device__ double operator()(double x) const { return fabs(x); }
};

/*
 * now_seconds - current monotonic time, in seconds.
 *
 * Returns: seconds since an unspecified fixed point, as a double.
 * Suitable for measuring elapsed time, not wall-clock date/time.
 */
static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

struct CudaColorizer {
    int matrix_size;
    int block_size;
    size_t total_pixels;
    double *device_u;
    unsigned char *device_rgb;
    unsigned char *host_rgb;
    char *ascii_buffer;
    size_t ascii_capacity;
};

static char last_error[256] = "no error";

/*
 * set_cuda_error - record a CUDA API failure as the last error.
 *
 * operation: short label for what was being attempted (e.g. "cudaMalloc").
 * error:     the CUDA error code returned.
 *
 * Returns: nothing.
 */
static void set_cuda_error(const char *operation, cudaError_t error) {
    snprintf(last_error, sizeof(last_error), "%s: %s", operation,
             cudaGetErrorString(error));
}

/*
 * set_text_error - record a plain-text failure as the last error.
 *
 * message: error message to store.
 *
 * Returns: nothing.
 */
static void set_text_error(const char *message) {
    snprintf(last_error, sizeof(last_error), "%s", message);
}

/*
 * to_byte - clamp a value to [0, 255] and round to the nearest byte.
 *
 * value: input value (any range).
 *
 * Returns: value clamped to [0, 255], rounded to the nearest unsigned
 * char.
 */
__device__ static unsigned char to_byte(double value) {
    if (value < 0.0) value = 0.0;
    if (value > 255.0) value = 255.0;
    return (unsigned char)(value + 0.5);
}

/*
 * colorize_kernel - map one scalar field value to an RGB color, one GPU
 * thread per pixel.
 *
 * u:           input scalar field, length total.
 * rgb:         output buffer, length 3*total (R,G,B per pixel).
 * total:       number of pixels (matrix_size^2).
 * color_scale: normalization scale (already the effective per-frame
 *              scale, floor already applied by the caller).
 * zero_band:   fraction of the normalized magnitude below which the
 *              pixel renders white.
 *
 * White near zero, fading to red/orange for positive values and blue for
 * negative, saturating at +-color_scale.
 *
 * Returns: nothing (writes rgb[3*idx .. 3*idx+2] for this thread's idx).
 */
__global__ static void colorize_kernel(const double *u, unsigned char *rgb,
                                       int total, double color_scale,
                                       double zero_band) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    double normalized = color_scale > 0.0 ? u[idx] / color_scale : 0.0;
    if (normalized > 1.0) normalized = 1.0;
    if (normalized < -1.0) normalized = -1.0;

    normalized = copysign(sqrt(fabs(normalized)), normalized);

    double magnitude = fabs(normalized);
    double fade = 0.0;
    if (magnitude > zero_band) {
        fade = (magnitude - zero_band) / (1.0 - zero_band);
        if (fade > 1.0) fade = 1.0;
    }

    unsigned char r = 255, g = 255, b = 255;
    if (fade > 0.0 && normalized > 0.0) {
        r = 255;
        g = to_byte(255.0 * (1.0 - 0.85 * fade));
        b = to_byte(255.0 * (1.0 - fade));
    } else if (fade > 0.0) {
        r = to_byte(255.0 * (1.0 - fade));
        g = to_byte(255.0 * (1.0 - 0.55 * fade));
        b = 255;
    }

    rgb[3 * idx] = r;
    rgb[3 * idx + 1] = g;
    rgb[3 * idx + 2] = b;
}

/* cuda_colorizer_last_error - see wave_color_cuda.h. */
const char *cuda_colorizer_last_error(void) {
    return last_error;
}

/* cuda_colorizer_select_device - see wave_color_cuda.h. */
int cuda_colorizer_select_device(int rank) {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess) {
        set_cuda_error("cudaGetDeviceCount", error);
        return 0;
    }
    if (device_count <= 0) {
        set_text_error("no CUDA devices available");
        return 0;
    }

    error = cudaSetDevice(rank % device_count);
    if (error != cudaSuccess) {
        set_cuda_error("cudaSetDevice", error);
        return 0;
    }
    return 1;
}

/* cuda_colorizer_create - see wave_color_cuda.h. */
int cuda_colorizer_create(CudaColorizer **colorizer, int matrix_size, int block_size) {
    if (colorizer == NULL || matrix_size <= 0 || block_size <= 0) {
        set_text_error("invalid CUDA colorizer arguments");
        return 0;
    }
    *colorizer = NULL;

    CudaColorizer *ctx = (CudaColorizer *)calloc(1, sizeof(*ctx));
    if (ctx == NULL) {
        set_text_error("could not allocate CUDA colorizer context");
        return 0;
    }

    ctx->matrix_size = matrix_size;
    ctx->block_size = block_size;
    ctx->total_pixels = (size_t)matrix_size * (size_t)matrix_size;
    size_t scalar_bytes = ctx->total_pixels * sizeof(double);
    size_t rgb_bytes = ctx->total_pixels * 3u * sizeof(unsigned char);

    cudaError_t error = cudaMalloc((void **)&ctx->device_u, scalar_bytes);
    if (error != cudaSuccess) {
        set_cuda_error("cudaMalloc device_u", error);
        cuda_colorizer_destroy(ctx);
        return 0;
    }

    error = cudaMalloc((void **)&ctx->device_rgb, rgb_bytes);
    if (error != cudaSuccess) {
        set_cuda_error("cudaMalloc device_rgb", error);
        cuda_colorizer_destroy(ctx);
        return 0;
    }

    error = cudaHostAlloc((void **)&ctx->host_rgb, rgb_bytes, cudaHostAllocDefault);
    if (error != cudaSuccess) {
        set_cuda_error("cudaHostAlloc host_rgb", error);
        cuda_colorizer_destroy(ctx);
        return 0;
    }

    ctx->ascii_capacity = 32u + ctx->total_pixels * 12u;
    ctx->ascii_buffer = (char *)malloc(ctx->ascii_capacity);
    if (ctx->ascii_buffer == NULL) {
        set_text_error("could not allocate ASCII PPM buffer");
        cuda_colorizer_destroy(ctx);
        return 0;
    }

    *colorizer = ctx;
    return 1;
}

/* cuda_colorizer_destroy - see wave_color_cuda.h. */
void cuda_colorizer_destroy(CudaColorizer *colorizer) {
    if (colorizer == NULL) {
        return;
    }
    if (colorizer->device_u != NULL) {
        cudaFree(colorizer->device_u);
    }
    if (colorizer->device_rgb != NULL) {
        cudaFree(colorizer->device_rgb);
    }
    if (colorizer->host_rgb != NULL) {
        cudaFreeHost(colorizer->host_rgb);
    }
    free(colorizer->ascii_buffer);
    free(colorizer);
}

/*
 * append_byte_ascii - write a byte's decimal digits, no padding.
 *
 * out:   write position; the caller must ensure room for up to 3 chars.
 * value: byte to format (0-255).
 *
 * Returns: pointer to just past the last digit written.
 */
static inline char *append_byte_ascii(char *out, unsigned char value) {
    if (value >= 100) {
        *out++ = (char)('0' + value / 100);
        *out++ = (char)('0' + (value / 10) % 10);
        *out++ = (char)('0' + value % 10);
    } else if (value >= 10) {
        *out++ = (char)('0' + value / 10);
        *out++ = (char)('0' + value % 10);
    } else {
        *out++ = (char)('0' + value);
    }
    return out;
}

/*
 * format_ppm_ascii - render colorizer->host_rgb as an ASCII P3 PPM into
 * colorizer->ascii_buffer.
 *
 * colorizer: context holding the RGB data (host_rgb) and the pre-sized
 *            output buffer (ascii_buffer) to write into.
 *
 * Returns: number of bytes written to ascii_buffer.
 */
static size_t format_ppm_ascii(CudaColorizer *colorizer) {
    char *out = colorizer->ascii_buffer;
    int m = colorizer->matrix_size;

    out += sprintf(out, "P3\n%d %d\n255\n", m, m);

    const unsigned char *rgb = colorizer->host_rgb;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            size_t idx = ((size_t)i * (size_t)m + (size_t)j) * 3u;
            out = append_byte_ascii(out, rgb[idx]);
            *out++ = ' ';
            out = append_byte_ascii(out, rgb[idx + 1]);
            *out++ = ' ';
            out = append_byte_ascii(out, rgb[idx + 2]);
            *out++ = (j + 1 < m) ? ' ' : '\n';
        }
    }

    return (size_t)(out - colorizer->ascii_buffer);
}

/* cuda_write_ppm - see wave_color_cuda.h. */
int cuda_write_ppm(const char *path, const double *u, int matrix_size,
                   double color_scale, double zero_band,
                   CudaColorizer *colorizer, CudaFrameTiming *timing) {
    if (path == NULL || u == NULL || colorizer == NULL ||
        matrix_size != colorizer->matrix_size || color_scale <= 0.0 ||
        zero_band < 0.0 || zero_band >= 1.0) {
        set_text_error("invalid CUDA PPM write arguments");
        return 0;
    }

    size_t scalar_bytes = colorizer->total_pixels * sizeof(double);
    size_t rgb_bytes = colorizer->total_pixels * 3u * sizeof(unsigned char);
    double t0, t1;

    t0 = now_seconds();
    cudaError_t error = cudaMemcpy(colorizer->device_u, u, scalar_bytes,
                                   cudaMemcpyHostToDevice);
    t1 = now_seconds();
    if (timing) timing->h2d_s = t1 - t0;
    if (error != cudaSuccess) {
        set_cuda_error("cudaMemcpy host to device", error);
        return 0;
    }

    /* An explicit sync here (rather than letting the D2H memcpy below
     * implicitly wait for the kernel) exists purely so kernel_s and d2h_s
     * measure two separate things instead of one blurred together -- a
     * small stall reintroduced specifically for this measurement, not
     * needed for correctness. kernel_s now covers both the peak-finding
     * reduction below and the colorize kernel itself -- both are GPU
     * compute work back-to-back, not worth separate timing buckets. */
    t0 = now_seconds();

    /* Adaptive per-frame color scale: a fixed color_scale (calibrated to
     * the original impulse amplitude) reads as pale almost immediately,
     * because a 2D wave's peak amplitude falls off with distance from the
     * source (geometric spreading) on top of gamma damping -- by frame 50
     * of 400 the true peak is already ~5% of the impulse amplitude with a
     * fixed scale. Finding each frame's own peak |u| and normalizing
     * against that keeps every frame visually vivid throughout. The
     * caller-supplied color_scale becomes a floor, not the scale itself:
     * once the wave has genuinely decayed close to zero, this stops
     * shrinking the scale further, so the video still fades to white at
     * the very end rather than renormalizing numerical noise to full
     * brightness. */
    double frame_peak = thrust::transform_reduce(
        thrust::device_pointer_cast(colorizer->device_u),
        thrust::device_pointer_cast(colorizer->device_u) + colorizer->total_pixels,
        AbsValue(), 0.0, thrust::maximum<double>());
    double effective_scale = frame_peak > color_scale ? frame_peak : color_scale;

    int blocks = (int)((colorizer->total_pixels + (size_t)colorizer->block_size - 1u) /
                       (size_t)colorizer->block_size);
    colorize_kernel<<<blocks, colorizer->block_size>>>(colorizer->device_u,
                                                        colorizer->device_rgb,
                                                        (int)colorizer->total_pixels,
                                                        effective_scale, zero_band);
    error = cudaGetLastError();
    if (error == cudaSuccess) {
        error = cudaDeviceSynchronize();
    }
    t1 = now_seconds();
    if (timing) timing->kernel_s = t1 - t0;
    if (error != cudaSuccess) {
        set_cuda_error("colorize_kernel launch", error);
        return 0;
    }

    t0 = now_seconds();
    error = cudaMemcpy(colorizer->host_rgb, colorizer->device_rgb, rgb_bytes,
                       cudaMemcpyDeviceToHost);
    t1 = now_seconds();
    if (timing) timing->d2h_s = t1 - t0;
    if (error != cudaSuccess) {
        set_cuda_error("cudaMemcpy device to host", error);
        return 0;
    }

    t0 = now_seconds();
    size_t content_size = format_ppm_ascii(colorizer);
    t1 = now_seconds();
    if (timing) timing->format_s = t1 - t0;

    t0 = now_seconds();
    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        snprintf(last_error, sizeof(last_error), "could not open '%s' for writing", path);
        return 0;
    }

    size_t written = fwrite(colorizer->ascii_buffer, 1, content_size, file);
    int closed_ok = (fclose(file) == 0);
    t1 = now_seconds();
    if (timing) timing->write_s = t1 - t0;

    if (written != content_size || !closed_ok) {
        snprintf(last_error, sizeof(last_error), "incomplete write for '%s'", path);
        return 0;
    }
    return 1;
}
