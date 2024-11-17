#include "matrix.h"
#include <ncurses.h>

matrix startmtrx(unsigned int heigth, unsigned int width, unsigned int y0, unsigned int x0, unsigned int data_heigth, unsigned int data_width){
    
    matrix mtrx = {
        newwin(heigth,width,y0,x0),
        0,
        0,
        data_width,
        data_heigth,
        ((float)heigth) / ((float)data_heigth),
        ((float)width) / ((float)data_width),
    };

    return mtrx;
}

void loaddata(matrix * mtrx,float ** data, unsigned int heigth, unsigned int width){
    unsigned int mtrx_heigth;
    unsigned int mtrx_width;
    getmaxyx(mtrx->win,mtrx_heigth,mtrx_width);

    //the data set must at least be the size of the matix
    if(heigth < mtrx_heigth || width < mtrx_width)
        return;
    
    float h_ratio = ((float)mtrx_heigth)/((float)heigth);
    float w_ratio = ((float)mtrx_width)/((float)width);

    //load the data into the mtrx's window
    //downsample the data table to fit the mtrx
    float sample_x = 0;
    float sample_y = 0;
    for(unsigned int i = 0; i < mtrx_width; i++){
    for(unsigned int j = 0; j < mtrx_heigth; j++){
        float sample = data[(int)sample_x][(int)sample_y];
        if(mtrx->cell_max < sample)
            mtrx->cell_max = sample;
        sample = 1000 * sample/mtrx->cell_max;

        init_color(COLOR_CYAN,sample,sample,sample);
        
        sample_x += w_ratio;
    }
        sample_y += h_ratio;
    }
}

void destroymtrx(matrix * mtrx){
    delwin(mtrx->win);
}