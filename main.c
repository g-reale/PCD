#include <ncurses.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "matrix.h"
#include "outputwin.h"
#include "simulations.h"
#include "globals.h"

#define M 100
#define N 100
#define DISP_HEIGHT 4
#define DIFUSION 0.1
#define TIME_STEP 0.01
#define DELTA_X 1.0
#define THREADS 16

int main(){
    initscr();

    //space where the simulation will take place
    float_2D simspace;
    start2D(M,N,simspace,float);

    //display labels
    char_2D labels;
    start2D(5,10,labels,char);
    strcpy(labels.data[0],"time:\0");
    strcpy(labels.data[1],"threads:\0");
    strcpy(labels.data[2],"d_coeff:\0");
    strcpy(labels.data[3],"speedup:\0");
    strcpy(labels.data[4],"efficy:\0");

    //getting the screen layout
    size_t y0 = 0;
    size_t width, height;
    getmaxyx(stdscr, height, width);

    //drawing the widgets
    size_t mtrx_height = height - DISP_HEIGHT;
    matrix mtrx = start_mtrx(mtrx_height,width,y0,0,M,N);
    y0 += mtrx_height;
    outputwin outwin = start_outputwin(DISP_HEIGHT, width, y0, 0, labels, 3);
    update_single_outputwin(outwin,1,"%2d",THREADS);
    update_single_outputwin(outwin,2,"%1.3f",DIFUSION);
    wrefresh(outwin.win);

    //user defined simpace initial state
    configure_mtrx(mtrx,simspace);
    
    //configuring the simulation 
    simulation sim_multi_thread = start_simulation(simspace,DELTA_X,DIFUSION,THREADS);
    simulation sim_single_thread = start_simulation(simspace,DELTA_X,DIFUSION,1);
    
    //runing the bad boy
    noecho();
    curs_set(0);
    nodelay(stdscr,true);
    double time_multi_thread;
    double time_single_thread;
    int input = 0; 
    do{
        //run the serial and parallel simulations
        float_2D current_frame = simulate_OMP(&sim_multi_thread,0.01,&time_multi_thread); 
        simulate_OMP(&sim_single_thread,0.01,&time_single_thread); //discard the single threaded frame
        update_mtrx(&mtrx,current_frame);
        
        //calculate the metrics
        double speedup = time_single_thread / time_multi_thread;
        double efficiency = speedup / ((double)THREADS); 
        
        //output the metrics
        update_single_outputwin(outwin,0,"%1.3f",time_multi_thread);
        update_single_outputwin(outwin,3,"%2.1f",speedup);
        update_single_outputwin(outwin,4,"%2.1f",efficiency);
        wrefresh(outwin.win);
        
        //stop if requested
        input = wgetch(stdscr);
    }while(input != 'q');

    //destroying all allocated memory an terminating the program
    destroy_mtrx(mtrx);
    destroy_outputwin(outwin);
    destroy_simulation(sim_multi_thread);
    destroy_simulation(sim_single_thread);
    destroy2D(simspace);
    destroy2D(labels);
    endwin();
    return 0;
}