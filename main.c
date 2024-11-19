#include <ncurses.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "matrix.h"
#include "outputwin.h"
#include "globals.h"

#define M 2000
#define N 2000

int main(){
    initscr();
    
    float_2D simspace;
    start2D(M,N,simspace,float);

    float max = 0;
    for(size_t i = 0; i < simspace.height; i++){
    for(size_t j = 0; j < simspace.width; j++){
        simspace.data[i][j] = j * j + i * i;
        if(max < fabs(simspace.data[i][j]))
            max = fabs(simspace.data[i][j]);
        // fprintf(stderr,"%ld %ld %f\n",i,j,simspace.data[i][j]);
    }}
    // fprintf(stderr,"\n\n\n");

    char_2D labels;
    start2D(5,10,labels,char);
    labels.data[0] = "time:\0";
    labels.data[1] = "threads:\0";
    labels.data[2] = "d_coeff:\0";
    labels.data[3] = "speedup:\0";
    labels.data[4] = "efficy:\0";

    char_2D values;
    start2D(5,10,values,char);
    for(size_t i = 0; i < 5; i++)
        values.data[i] = "-\\-\0";

    size_t width, height;
    size_t y0 = 0;
    getmaxyx(stdscr, height, width);

    size_t mtrx_height = height - 4;
    matrix mtrx = start_mtrx(mtrx_height,width,y0,0,M,N);
    y0 += mtrx_height;
    
    configure_mtrx(mtrx);
    // mtrx.cell_max = 1828377600;
    // update_mtrx(mtrx,simspace);
    refresh();
    
    outputwin outwin = start_outputwin(height - mtrx_height, width, y0, 0, labels, 3);
    update_outputwin(outwin,values);    

    getch();
    destroy_mtrx(mtrx);
    destroy_outputwin(outwin);
    destroy2D(simspace);
    destroy2D(labels);
    destroy2D(values);

    endwin();
    return 0;
}

//gcc matrix.c main.c -o main -fsanitize=address -lncurses -lm 