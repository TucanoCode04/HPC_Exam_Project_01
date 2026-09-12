#ifndef WAVE_COLOR_CUDA_H
#define WAVE_COLOR_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CudaColorizer CudaColorizer;

int cuda_colorizer_select_device(int rank);
int cuda_colorizer_create(CudaColorizer **colorizer, int matrix_size, int block_size);
void cuda_colorizer_destroy(CudaColorizer *colorizer);
int cuda_write_ppm(const char *path, const double *u, int matrix_size,
                   double color_scale, double zero_band,
                   CudaColorizer *colorizer);
const char *cuda_colorizer_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
