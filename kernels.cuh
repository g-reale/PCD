#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include "globals.cuh"
#include "simulations.cuh"

#ifdef __cplusplus
extern "C" {
#endif

void start_cuda(simulation *sim, float delta_t);
void restart_cuda(simulation *sim, float delta_t);
void update_cuda(float delta_t);
float * diffusion_cuda(int id);
void reduction_cuda(float *difference);
void destroy_cuda();

#ifdef __cplusplus
}
#endif

#endif
