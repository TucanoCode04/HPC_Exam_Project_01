#ifndef WAVE_COLOR_CUDA_H
#define WAVE_COLOR_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CudaColorizer CudaColorizer;

/* One frame's cost broken down by phase, in seconds. Every cuda_write_ppm
 * call overwrites this with just that frame's numbers (the caller
 * accumulates across frames itself, same pattern as color_time/compute_time
 * in assignment_3.c) -- this is measurement-only instrumentation to decide
 * whether parallelizing the ASCII formatting step is actually worth it. */
typedef struct {
    double h2d_s;
    double kernel_s;
    double d2h_s;
    double format_s;
    double write_s;
} CudaFrameTiming;

/*
 * cuda_colorizer_select_device - bind this process to a CUDA device.
 *
 * rank: MPI rank; the device used is rank % device_count, spreading ranks
 *       across whatever GPUs are visible to the job.
 *
 * Returns: 1 on success, 0 if no CUDA device is available or the call
 * fails (see cuda_colorizer_last_error()).
 */
int cuda_colorizer_select_device(int rank);

/*
 * cuda_colorizer_create - allocate a colorizer context for one scenario.
 *
 * colorizer:   output parameter, set to the new context on success.
 * matrix_size: grid size (matrix is matrix_size x matrix_size).
 * block_size:  CUDA threads per block for colorize_kernel.
 *
 * Allocates device buffers, a pinned host RGB buffer, and the ASCII PPM
 * output buffer once, to be reused across every frame of the scenario.
 *
 * Returns: 1 on success, 0 on failure (see cuda_colorizer_last_error()).
 */
int cuda_colorizer_create(CudaColorizer **colorizer, int matrix_size, int block_size);

/*
 * cuda_colorizer_destroy - free a colorizer context.
 *
 * colorizer: context to free; NULL is accepted and ignored.
 *
 * Returns: nothing.
 */
void cuda_colorizer_destroy(CudaColorizer *colorizer);
/*
 * cuda_write_ppm - colorize one frame on the GPU and write it as a PPM.
 *
 * path:        output file path.
 * u:           scalar field to colorize (matrix_size x matrix_size).
 * matrix_size: grid size; must match the size colorizer was created with.
 * color_scale: a FLOOR, not the color scale itself: each frame is
 *              normalized against its own peak |u| (found via a GPU
 *              reduction) so colors stay vivid as the wave spreads/
 *              decays; color_scale only takes over once the frame's true
 *              peak drops below it, so the video still fades to white
 *              once the wave has genuinely become negligible instead of
 *              renormalizing numerical noise to full brightness forever.
 * zero_band:   fraction of the (adaptive) scale below which a pixel
 *              renders white; must be in [0, 1).
 * colorizer:   context from cuda_colorizer_create().
 * timing:      output parameter, per-phase timing for this call
 *              (h2d/kernel/d2h/format/write); may be NULL.
 *
 * Returns: 1 on success, 0 on failure (see cuda_colorizer_last_error()).
 */
int cuda_write_ppm(const char *path, const double *u, int matrix_size,
                   double color_scale, double zero_band,
                   CudaColorizer *colorizer, CudaFrameTiming *timing);

/*
 * cuda_colorizer_last_error - message for the most recent failure.
 *
 * Returns: a pointer to a static, human-readable error string. Valid
 * until the next failing call in this module.
 */
const char *cuda_colorizer_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
