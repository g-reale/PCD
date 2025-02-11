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

#define SIMULATION_INTERFACE(plataform) float_2D simulate_##plataform(simulation * sim, float delta_t, double * elapsed_time, float * difference, [[maybe_unused]] void * misc_data)
simulation start_simulation(float_2D simspace, float delta_x, float diffusion, size_t n_threads);
SIMULATION_INTERFACE(OMP);
SIMULATION_INTERFACE(base);
SIMULATION_INTERFACE(MPI);
void destroy_simulation(simulation sim);

#endif
