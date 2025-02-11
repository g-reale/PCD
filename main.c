#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "simulations.h"
#include "globals.h"
#include "MPIconfig.h"

#define DISP_HEIGHT 4

int main(int argc, char ** argv){

    size_t M;
    size_t N;
    size_t threads;
    size_t iterations;
    float diffusion;
    float time_step;
    float delta_x;
 
    //pop the command line arguments
    if(argc == 9){
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        threads = atoi(argv[3]);
        iterations = atoi(argv[4]);
        diffusion = atof(argv[5]);
        time_step = atof(argv[6]);
        delta_x = atof(argv[7]);
    }
    //if no arguments use default values
    else{
        M = 10000;
        N = 10000;
        threads = 2;
        iterations = 100;
        diffusion = 0.1;
        time_step = 0.01;
        delta_x = 1.0;
    }

    //simulation mode
    MPI_context * context = start_MPI_context(&argc,&argv);
    
    //print csv header on stdout
    //user can redirect th std to any file they wish
    if(context->me == 0) 
        fprintf(stdout,"iteration;exec time (1 thread);exec time (%ld threads);exec time (%ld threads linear);speedup;speedup (linear);efficiency;efficiency (linear);difference (1 thread);difference(%ld threads)",threads,threads,threads);

    //space where the simulation will take place
    float_2D simspace;
    start2D(M,N,simspace,float);
    
    //fill the middle of the matrix w\ a dummy input
    simspace.data[M/2][N/2] = 1e10;

    //configuring simulation
    simulation sim_multi_thread = start_simulation(simspace,delta_x,diffusion,threads);
    simulation sim_single_thread = start_simulation(simspace,delta_x,diffusion,1);
    double t_iter_single_thread = 0;
    double t_iter_multi_thread = 0;
    
    double time_multi_thread = 0;
    double time_single_thread = 0;
    double time_linear = 0;

    float difference_multi_thread;
    float difference_single_thread;

    //running the bad boy..
    for(size_t i = 0; i < iterations; i++){
        
        //run the single and multi threaded simulations
        simulate_MPI(&sim_multi_thread,time_step,&t_iter_multi_thread,&difference_multi_thread,context); 
        
        if(context->me == 0){
            simulate_base(&sim_single_thread,time_step,&t_iter_single_thread,&difference_single_thread,NULL); //discard the single threaded frame    
            
            //calculate the metrics
            time_multi_thread += t_iter_multi_thread;
            time_single_thread += t_iter_single_thread;
            time_linear = time_single_thread / ((double)threads);
            double speedup = time_single_thread / time_multi_thread;
            double efficiency = speedup / ((double)threads);
            double speedup_linear = time_single_thread / time_linear;
            double efficiency_linear = speedup_linear / ((double)threads);
    
            //output to stdout
            fprintf(stdout,"\n%ld;%f;%f;%f;%f;%f;%f;%f;%f;%f",i,time_single_thread,time_multi_thread,time_linear,speedup,speedup_linear,efficiency,efficiency_linear,difference_single_thread,difference_multi_thread);
        }
    }

    //destroying all allocated memory an terminating the program
    destroy_simulation(sim_multi_thread);
    destroy_simulation(sim_single_thread);
    destroy2D(simspace);
    destroy_MPI_context(context);
    return 0;
}