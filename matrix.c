#include "matrix.h"
#include "globals.h"
#include <ncurses.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>

matrix start_mtrx(size_t height, size_t width, size_t y0, size_t x0, size_t data_height, size_t data_width){
    
    if(height < 2)
        height = 2;
    if(width < 2)
        width = 2;

    height -= 2; //make space for the box
    width -= 2; //make space for the box

    matrix mtrx = {
        newwin(height + 2,width + 2,y0,x0),
        0.0f,
        data_height,
        data_width,
        {0,NULL},
        {0,NULL}  
    };

    //refresh to tell ncurses about the new window
    refresh();

    //create the downsampling plan on the x axis 
    start1D(width,mtrx.steps_x,size_t);
    
    float ratio = ((float)mtrx.data_width) / ((float)mtrx.steps_x.height);
    size_t integer = floor(ratio);
    float decimal = ratio - floor(ratio);
    float error = 0;

    for(size_t i = 0; i < mtrx.steps_x.height; i++){
        mtrx.steps_x.data[i] = integer + floor(error);
        error += decimal - floor(error);
    }

    //create the downsampling plan on the y axis
    start1D(height,mtrx.steps_y,size_t);
    
    ratio = ((float)mtrx.data_height) / ((float)mtrx.steps_y.height);
    integer = floor(ratio);
    decimal = ratio - floor(ratio);
    error = 0;

    for(size_t i = 0; i < mtrx.steps_y.height; i++){
        mtrx.steps_y.data[i] = integer + floor(error);
        error += decimal - floor(error);
    }

    //put a box around the window
    box(mtrx.win,'|','-'); 
    wrefresh(mtrx.win);
    return mtrx;
}

void configure_mtrx(matrix mtrx){
    noecho();
    wmove(mtrx.win,1,1);
    keypad(mtrx.win,true);

    size_t input;
    int done = false;
    size_t cursor_y = 1;
    size_t cursor_x = 1;

    do{
        input = wgetch(mtrx.win);
        fprintf(stderr,"%c %ld %ld\n",(char)input, cursor_y, cursor_x);
        switch (input){
            case KEY_UP:
                cursor_y = clamp(cursor_y-1,1,mtrx.steps_y.height);
                wmove(mtrx.win,cursor_y,cursor_x);
                wrefresh(mtrx.win);
                break;
            
            case KEY_DOWN:
                cursor_y = clamp(cursor_y+1,1,mtrx.steps_y.height);
                wmove(mtrx.win,cursor_y,cursor_x);
                wrefresh(mtrx.win);
                break;

            case KEY_LEFT:
                cursor_x = clamp(cursor_x-1,1,mtrx.steps_x.height);
                wmove(mtrx.win,cursor_y,cursor_x);
                wrefresh(mtrx.win);
                break;
            
            case KEY_RIGHT:
                cursor_x = clamp(cursor_x+1,1,mtrx.steps_x.height);
                wmove(mtrx.win,cursor_y,cursor_x);
                wrefresh(mtrx.win);
                break;

            default:
                fprintf(stderr,"default %c\n",(char)input);
                if(input == '\n' || input == '\t' || input == 'q'){
                    fprintf(stderr,"done\n");
                    done = true;
                }
                break;

            fflush(stderr);
        }
    }while (!done);
}

void update_mtrx(matrix mtrx, float_2D simspace){
    
    if(mtrx.data_height != simspace.height || mtrx.data_width != simspace.width)
        return;
    
    size_t row = 0;
    size_t column = 0;

    for(size_t i = 0; i < mtrx.steps_y.height; i++){
        size_t step_y = mtrx.steps_y.data[i];
        
        for(size_t j = 0; j < mtrx.steps_x.height; j++){
            size_t step_x = mtrx.steps_x.data[j];

            //calculate the sum according to the downsampling plan
            float sum = 0;
            for(size_t k = column; k < column + step_y; k++){
            for(size_t l = row; l < row + step_x; l++){
                sum += simspace.data[k][l];

                if(i == 0)
                    fprintf(stderr,"%f %f %f\n",sum,simspace.data[k][l],mtrx.cell_max);
            }}

            //normalize the sum from [0,1]
            if(mtrx.cell_max < sum)
                mtrx.cell_max = sum;
            if(mtrx.cell_max)
                sum = sum/mtrx.cell_max;

            //print the sum using ncurses
            // size_t index = fabs(sum) * (float)GRAY_SCALE_1.height;
            // index %= GRAY_SCALE_1.height;
            char gamma = getgrayscale(GRAY_SCALE_1,sum);
            fprintf(stderr,"%ld %ld %c %f\n",i,j,gamma,sum);
            mvwprintw(mtrx.win,i+1,j+1,"%c",gamma);
            
            //update the starting row of the sum
            row += step_x;
        }

        //update the starting column of the sum
        row = 0;
        column += step_y;
    }

    //update the window
    wrefresh(mtrx.win);

    //restore the original color information
    // popcolor();
}

void destroy_mtrx(matrix mtrx){
    delwin(mtrx.win);
    destroy1D(mtrx.steps_y);
    destroy1D(mtrx.steps_x);
}