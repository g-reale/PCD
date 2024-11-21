#include <omp.h>
#include <stddef.h>
#include <stdlib.h>

#include "globals.h"
#include "simulations.h"

simulation start_simulation(float_2D simspace, float delta_x, float diffusion, size_t n_threads){
    simulation sim = {
        delta_x,
        diffusion,
        n_threads,
        {{0,0,NULL},{0,0,NULL}},
        0
    };
    start2D(simspace.height,simspace.width,sim.simspace[0],float);
    start2D(simspace.height,simspace.width,sim.simspace[1],float);
    for(size_t i = 0; i < simspace.height; i++){
    for(size_t j = 0; j < simspace.width; j++){
        sim.simspace[0].data[i][j] = simspace.data[i][j];
    }}
    return sim;
}

float_2D simulate_OMP(simulation * sim, float delta_t, double * elapsed_time){
    
    //prepare to return the elapsed time
    double start = omp_get_wtime();

    //handling thread count
    int old_dynamic = omp_get_dynamic();
    omp_set_dynamic(false);
    omp_set_num_threads(sim->n_threads);
    
    //unpacking the parameters for faster access
    float constant = sim->diffusion * delta_t / (sim->delta_x * sim->delta_x);
    float ** current = sim->simspace[sim->current_space].data;
    sim->current_space = !sim->current_space;
    float ** next = sim->simspace[sim->current_space].data;
    size_t height = sim->simspace[0].height;
    size_t width = sim->simspace[0].width;

    //calculate the next iteration
    #pragma omp parallel for
        for(size_t i = 1; i < height - 1; i++){
        for(size_t j = 1; j < width - 1; j++){
            next[i][j] = (current[i][j] + 
                            constant * 
                            (current[i - 1][j] + 
                            current[i + 1][j] +
                            current[i][j - 1] +   
                            current[i][j + 1] - 
                            4 * current[i][j]));
        }}    

    omp_set_dynamic(old_dynamic);
    *elapsed_time = omp_get_wtime() - start;
    return sim->simspace[sim->current_space];
} 

void destroy_simulation(simulation sim){
    destroy2D(sim.simspace[0]);
    destroy2D(sim.simspace[1]);
}

