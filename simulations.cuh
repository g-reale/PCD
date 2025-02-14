#ifndef SIMUMLATIONS_H
#define SIMUMLATIONS_H
#include "globals.cuh"

typedef struct{
    float delta_x;
    float diffusion;
    size_t n_threads;
    
    float_2D simspace[2];
    size_t current_space;
}simulation;



#define SIMULATION_INTERFACE(plataform) float_2D simulate_##plataform(simulation * sim, float delta_t, double * elapsed_time, float * difference)

simulation start_simulation(float_2D simspace, float delta_x, float diffusion, size_t n_threads);
SIMULATION_INTERFACE(OMP);
SIMULATION_INTERFACE(base);
SIMULATION_INTERFACE(cuda);
void destroy_simulation(simulation sim);

#endif

