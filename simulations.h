#ifndef SIMUMLATIONS_H
#define SIMUMLATIONS_H

#include <omp.h>

#include "globals.h"

typedef struct{
    float delta_x;
    float diffusion;
    size_t n_threads;
    
    float_2D simspace[2];
    size_t current_space;
}simulation;

simulation start_simulation(float_2D simspace, float delta_x, float diffusion, size_t n_threads);
float_2D simulate_OMP(simulation * sim, float delta_t, double * elapsed_time);
float_2D simulate_base(simulation * sim, float delta_t, double * elapsed_time);
void destroy_simulation(simulation sim);

#endif
