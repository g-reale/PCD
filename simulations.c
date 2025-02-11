#include <omp.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#include "globals.h"
#include "simulations.h"
#include "MPIconfig.h"

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
    omp_set_dynamic(0);
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

SIMULATION_INTERFACE(MPI){
    
    
    //prepare to return the elapsed time
    double start = omp_get_wtime();
    
    //handling thread count
    int old_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
    omp_set_num_threads(sim->n_threads);
    
    //unpacking the parameters for faster access
    float constant = sim->diffusion * delta_t / (sim->delta_x * sim->delta_x);
    float ** current = sim->simspace[sim->current_space].data;
    sim->current_space = !sim->current_space;
    float ** next = sim->simspace[sim->current_space].data;
    size_t height = sim->simspace[0].height;
    size_t width = sim->simspace[0].width;
    float diff = 0;
    
    //use a modified OMP implementation for calculating the owner's matrix
    MPI_context * context = (MPI_context*) misc_data;
    int population = context->population;
    int me = context->me;
    int predecessor = context->predecessor;
    int sucessor = context->sucessor;
    
    int delta = ceilDiv(height - 2, population);
    int begin = me * delta + 1;
    int end = clamp(begin + delta, 0, (int) (height - 2));

    //calculate the next iteration and the difference as required by the problem prompt
    #pragma omp parallel for reduction(+:diff)
        for(size_t i = begin; i < end; i++){
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

        
    //Synchronize the border with the neighboring processes
    //async send to predecessor
    // printf("starting requests...%d\n",me);

    MPI_Request request;
    if(-1 < predecessor){
        MPI_Isend(
            next[begin],
            width,
            MPI_FLOAT,
            predecessor,
            0,
            MPI_COMM_WORLD,
            &request
        );
        // printf("first send... %d\n",me);
    }

    //async send to sucessor
    if(sucessor < population){
        MPI_Isend(
            next[end-1],
            width,
            MPI_FLOAT,
            sucessor,
            0,
            MPI_COMM_WORLD,
            &request
        );
        // printf("second send... %d\n",me);
    }

    //sync rcv from predecessor
    if(-1 < predecessor){
        MPI_Recv(
            next[begin-1],
            width,
            MPI_FLOAT,
            predecessor,
            0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );
        // printf("first receive... %d\n",me);
    }

    //sync rcv from sucessor
    if(sucessor < population){
        MPI_Recv(
            next[end],
            width,
            MPI_FLOAT,
            sucessor,
            0,   
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );
        // printf("second receive... %d\n",me);
    }

    //reduce the difference between processes
    // printf("exiting... %d\n",me);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&diff, difference, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    //return the data
    omp_set_dynamic(old_dynamic);
    *elapsed_time = omp_get_wtime() - start;
    return sim->simspace[sim->current_space];
}

void destroy_simulation(simulation sim){
    destroy2D(sim.simspace[0]);
    destroy2D(sim.simspace[1]);
}

