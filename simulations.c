#include <omp.h>
#include <math.h>
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

SIMULATION_INTERFACE(OMP){
    
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
    float diff = 0;

    //calculate the next iteration and the difference as required by the problem prompt
    #pragma omp parallel for reduction(+:diff)
        for(size_t i = 1; i < height - 1; i++){
        for(size_t j = 1; j < width - 1; j++){
            next[i][j] = (current[i][j] + 
                            constant * 
                            (current[i - 1][j] + 
                            current[i + 1][j] +
                            current[i][j - 1] +   
                            current[i][j + 1] - 
                            4 * current[i][j]));
            diff += fabs(next[i][j] - current[i][j]);
        }}

    
    omp_set_dynamic(old_dynamic);
    *elapsed_time = omp_get_wtime() - start;
    *difference = diff;
    return sim->simspace[sim->current_space];
} 

//base single threaded implementation
SIMULATION_INTERFACE(base){
    
    //prepare to return the elapsed time
    double start = omp_get_wtime();

    //unpacking the parameters for faster access
    float constant = sim->diffusion * delta_t / (sim->delta_x * sim->delta_x);
    float ** current = sim->simspace[sim->current_space].data;
    sim->current_space = !sim->current_space;
    float ** next = sim->simspace[sim->current_space].data;
    size_t height = sim->simspace[0].height;
    size_t width = sim->simspace[0].width;
    float diff = 0;

    //calculate the next iteration and the difference as required by the problem prompt
    for(size_t i = 1; i < height - 1; i++){
    for(size_t j = 1; j < width - 1; j++){
        next[i][j] = (current[i][j] + 
                        constant * 
                        (current[i - 1][j] + 
                        current[i + 1][j] +
                        current[i][j - 1] +   
                        current[i][j + 1] - 
                        4 * current[i][j]));
        diff += fabs(next[i][j] - current[i][j]);
    }}    

    *elapsed_time = omp_get_wtime() - start;
    *difference = diff;
    return sim->simspace[sim->current_space];
}

void destroy_simulation(simulation sim){
    destroy2D(sim.simspace[0]);
    destroy2D(sim.simspace[1]);
}

