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

int cuda_colorizer_select_device(int rank);
int cuda_colorizer_create(CudaColorizer **colorizer, int matrix_size, int block_size);
void cuda_colorizer_destroy(CudaColorizer *colorizer);
/* timing may be NULL if the caller doesn't want the per-phase breakdown.
 * color_scale is a FLOOR, not the color scale itself: each frame is
 * normalized against its own peak |u| (found via a GPU reduction) so
 * colors stay vivid as the wave spreads/decays; color_scale only takes
 * over once the frame's true peak drops below it, so the video still
 * fades to white once the wave has genuinely become negligible instead
 * of renormalizing numerical noise to full brightness forever. */
int cuda_write_ppm(const char *path, const double *u, int matrix_size,
                   double color_scale, double zero_band,
                   CudaColorizer *colorizer, CudaFrameTiming *timing);
const char *cuda_colorizer_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
