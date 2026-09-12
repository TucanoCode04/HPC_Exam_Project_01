#include "wave_color_cuda.h"

#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static void set_cuda_error(const char *operation, cudaError_t error) {
    snprintf(last_error, sizeof(last_error), "%s: %s", operation,
             cudaGetErrorString(error));
}

static void set_text_error(const char *message) {
    snprintf(last_error, sizeof(last_error), "%s", message);
}

__device__ static unsigned char to_byte(double value) {
    if (value < 0.0) value = 0.0;
    if (value > 255.0) value = 255.0;
    return (unsigned char)(value + 0.5);
}

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

const char *cuda_colorizer_last_error(void) {
    return last_error;
}

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

int cuda_write_ppm(const char *path, const double *u, int matrix_size,
                   double color_scale, double zero_band,
                   CudaColorizer *colorizer) {
    if (path == NULL || u == NULL || colorizer == NULL ||
        matrix_size != colorizer->matrix_size || color_scale <= 0.0 ||
        zero_band < 0.0 || zero_band >= 1.0) {
        set_text_error("invalid CUDA PPM write arguments");
        return 0;
    }

    size_t scalar_bytes = colorizer->total_pixels * sizeof(double);
    size_t rgb_bytes = colorizer->total_pixels * 3u * sizeof(unsigned char);

    cudaError_t error = cudaMemcpy(colorizer->device_u, u, scalar_bytes,
                                   cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        set_cuda_error("cudaMemcpy host to device", error);
        return 0;
    }

    int blocks = (int)((colorizer->total_pixels + (size_t)colorizer->block_size - 1u) /
                       (size_t)colorizer->block_size);
    colorize_kernel<<<blocks, colorizer->block_size>>>(colorizer->device_u,
                                                        colorizer->device_rgb,
                                                        (int)colorizer->total_pixels,
                                                        color_scale, zero_band);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        set_cuda_error("colorize_kernel launch", error);
        return 0;
    }

    error = cudaMemcpy(colorizer->host_rgb, colorizer->device_rgb, rgb_bytes,
                       cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        set_cuda_error("cudaMemcpy device to host", error);
        return 0;
    }

    size_t content_size = format_ppm_ascii(colorizer);

    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        snprintf(last_error, sizeof(last_error), "could not open '%s' for writing", path);
        return 0;
    }

    size_t written = fwrite(colorizer->ascii_buffer, 1, content_size, file);
    int closed_ok = (fclose(file) == 0);

    if (written != content_size || !closed_ok) {
        snprintf(last_error, sizeof(last_error), "incomplete write for '%s'", path);
        return 0;
    }
    return 1;
}
