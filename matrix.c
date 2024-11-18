#include "matrix.h"
#include "globals.h"
#include <ncurses.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>

matrix startmtrx(size_t height, size_t width, size_t y0, size_t x0, size_t data_height, size_t data_width){
    
    matrix mtrx = {
        newwin(height,width,y0,x0),
        0.0f,
        data_width,
        data_height,
        width,
        height,
        (size_t *) malloc(sizeof(size_t) * width),
        (size_t *) malloc(sizeof(size_t) * height),
    };

    //refresh to tell ncurses about the new window
    refresh();

    //create the downsampling plan on the x axis 
    float ratio = ((float)data_width) / ((float)width);
    size_t integer = floor(ratio);
    float decimal = ratio - floor(ratio);
    float error = 0;

    for(size_t i = 0; i < width; i++){
        mtrx.steps_x[i] = integer + floor(error);
        error += decimal - floor(error);
    }

    //create the downsampling plan on the y axis
    ratio = ((float)data_height) / ((float)height);
    integer = floor(ratio);
    decimal = ratio - floor(ratio);
    error = 0;

    for(size_t i = 0; i < height; i++){
        mtrx.steps_y[i] = integer + floor(error);
        error += decimal - floor(error);
    }

    return mtrx;
}

void loaddata(matrix * mtrx,float ** data, size_t data_height, size_t data_width){
    
    if(data_height != mtrx->data_height || data_width != mtrx->data_width)
        return;
    
    // push the color to restore it's original information later
    // pushcolor();
    
    size_t row = 0;
    size_t column = 0;

    for(size_t i = 0; i < mtrx->width; i++){
        size_t step_x = mtrx->steps_x[i];

        for(size_t j = 0; j < mtrx->height; j++){
            size_t step_y = mtrx->steps_y[j];

            //calculate the sum according to the downsampling plan
            float sum = 0;
            for(size_t k = row; k < row + step_x; k++){
            for(size_t l = column; l < column + step_y; l++){
                sum += data[k][l];
            }}

            // fprintf(stderr,"%lu %lu %f\n",i,j,sum);
            // fflush(stderr);
            
            //normalize the sum from [0,1]
            if(mtrx->cell_max < sum)
                mtrx->cell_max = sum;
            if(mtrx->cell_max)
                sum = sum/mtrx->cell_max;

            // fprintf(stderr,"%lu %lu %f %f",i,j,sum,mtrx->cell_max);
            // fflush(stderr);

            //print the sum using ncurses
            mvwprintw(mtrx->win,j,i,"%c",getgrayscale2(sum));
            
            //update the starting column of the sum
            column += step_y;
        }

        //update the starting row of the sum
        row += step_x;
        column = 0;
    }

    //update the window
    wrefresh(mtrx->win);

    //restore the original color information
    // popcolor();
}

void destroymtrx(matrix * mtrx){
    delwin(mtrx->win);
    free(mtrx->steps_x);
    free(mtrx->steps_y);
}