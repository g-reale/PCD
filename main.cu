#include <ncurses.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "matrix.cuh"
#include "outputwin.cuh"
#include "simulations.cuh"
#include "globals.cuh"
#include "kernels.cuh"

#define DISP_HEIGHT 4

int main(int argc, char ** argv){

    // int visual;
    size_t M;
    size_t N;
    size_t threads;
    size_t iterations;
    float diffusion;
    float time_step;
    float delta_x;
 
    //pop the command line arguments
    if(argc == 9){
        // visual = atoi(argv[1]);
        M = atoi(argv[2]);
        N = atoi(argv[3]);
        threads = atoi(argv[4]);
        iterations = atoi(argv[5]);
        diffusion = atof(argv[6]);
        time_step = atof(argv[7]);
        delta_x = atof(argv[8]);
    }
    //if no arguments use default values
    else{
        // visual = 1;
        M = 1000;
        N = 1000;
        threads = 8;
        iterations = 50;
        diffusion = 0.1;
        time_step = 0.01;
        delta_x = 1.0;
    }


    //visual mode
    // if(visual){

    //     initscr();

    //     //space where the simulation will take place
    //     float_2D simspace;
    //     start2D(M,N,simspace,float);

    //     //display labels
    //     char_2D labels;
    //     start2D(6,10,labels,char);
    //     strcpy(labels.data[0],"time:\0");
    //     strcpy(labels.data[1],"threads:\0");
    //     strcpy(labels.data[2],"d_coeff:\0");
    //     strcpy(labels.data[3],"speedup:\0");
    //     strcpy(labels.data[4],"efficy:\0");
    //     strcpy(labels.data[5],"diff:\0");

    //     //getting the screen layout
    //     size_t y0 = 0;
    //     size_t width, height;
    //     getmaxyx(stdscr, height, width);

    //     //drawing the widgets
    //     size_t mtrx_height = height - DISP_HEIGHT;
    //     matrix mtrx = start_mtrx(mtrx_height,width,y0,0,M,N);
    //     y0 += mtrx_height;
    //     outputwin outwin = start_outputwin(DISP_HEIGHT, width, y0, 0, labels, 3);
    //     update_single_outputwin(outwin,1,"%2ld",threads);
    //     update_single_outputwin(outwin,2,"%1.3f",diffusion);
    //     wrefresh(outwin.win);

    //     //user defined simpace initial state
    //     configure_mtrx(mtrx,simspace);
        
    //     //configuring the simulation 
    //     simulation sim_multi_thread = start_simulation(simspace,delta_x,diffusion,threads);
    //     simulation sim_single_thread = start_simulation(simspace,delta_x,diffusion,1);
        
    //     //runing the bad boy
    //     noecho();
    //     curs_set(0);
    //     nodelay(stdscr,true);
    //     double time_multi_thread;
    //     double time_single_thread;
    //     float difference_multi_thread;
    //     float difference_single_thread;
    //     int input = 0; 
    //     do{
    //         //run the serial and parallel simulations
    //         float_2D current_frame = simulate_OMP(&sim_multi_thread,time_step,&time_multi_thread,&difference_multi_thread); 
    //         simulate_base(&sim_single_thread,time_step,&time_single_thread,&difference_single_thread);
    //         update_mtrx(&mtrx,current_frame);
            
    //         //calculate the metrics
    //         double speedup = time_single_thread / time_multi_thread;
    //         double efficiency = speedup / ((double)threads); 
            
    //         //output the metrics
    //         update_single_outputwin(outwin,0,"%1.3f",time_multi_thread);
    //         update_single_outputwin(outwin,3,"%2.1f",speedup);
    //         update_single_outputwin(outwin,4,"%2.1f",efficiency);
    //         update_single_outputwin(outwin,5,"%2.1f",fabs(difference_multi_thread - difference_single_thread));
    //         wrefresh(outwin.win);
            
    //         //stop if requested
    //         input = wgetch(stdscr);
    //     }while(input != 'q');

    //     //destroying all allocated memory an terminating the program
    //     destroy_mtrx(mtrx);
    //     destroy_outputwin(outwin);
    //     destroy_simulation(sim_multi_thread);
    //     destroy_simulation(sim_single_thread);
    //     destroy2D(simspace);
    //     destroy2D(labels);
    //     endwin();
    //     return 0;
    // }
    
    //simulation mode
    
    //print csv header on stdout
    //user can redirect th std to any file they wish
    fprintf(stdout,"iteration;exec time (1 thread);exec time (%ld threads);exec time (%ld threads linear);speedup;speedup (linear);efficiency;efficiency (linear);difference (1 thread);difference(%ld threads)",threads,threads,threads);

    //space where the simulation will take place
    float_2D simspace;
    start2D(M,N,simspace,float);
    
    //fill the middle of the matrix w\ a dummy input
    simspace.data[M/2][N/2] = 1e10;
    simspace.data[M/2-1][N/2] = 1; //up
    simspace.data[M/2+1][N/2] = 2; //down
    simspace.data[M/2][N/2-1] = 3; //left
    simspace.data[M/2][N/2+1] = 4; //right

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

    start_cuda(&sim_multi_thread,time_step);

    //running the bad boy... again
    for(size_t i = 0; i < iterations; i++){
        
        //run the single and multi threaded simulations
        simulate_cuda(&sim_multi_thread,time_step,&t_iter_multi_thread,&difference_multi_thread); 
        simulate_base(&sim_single_thread,time_step,&t_iter_single_thread,&difference_single_thread); //discard the single threaded frame    
        
        //calculate the metrics
        time_multi_thread += t_iter_multi_thread;
        time_single_thread += t_iter_single_thread;
        time_linear = time_single_thread / ((double)threads);
        double speedup = time_single_thread / time_multi_thread;
        double efficiency = speedup / ((double)threads);
        double speedup_linear = time_single_thread / time_linear;
        double efficiency_linear = speedup_linear / ((double)threads);

        
        // float ** multi = sim_multi_thread.simspace[sim_multi_thread.current_space].data;
        // float ** single = sim_single_thread.simspace[sim_single_thread.current_space].data;
        // for(int j = 0; j < M; j++){
        // for(int k = 0; k < N; k++){
        //   printf("\n%f %f", multi[j][k],single[j][k]);
        // }}

        //output to stdout
        fprintf(stdout,"\n%ld;%f;%f;%f;%f;%f;%f;%f;%f;%f",i,time_single_thread,time_multi_thread,time_linear,speedup,speedup_linear,efficiency,efficiency_linear,difference_single_thread,difference_multi_thread);
    }

    //destroying all allocated memory an terminating the program
    destroy_cuda();
    destroy_simulation(sim_multi_thread);
    destroy_simulation(sim_single_thread);
    destroy2D(simspace);
    return 0;
}