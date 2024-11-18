#ifndef MATRIX_H
#define MATRIX_H

#include <ncurses.h>
#include <stddef.h>

typedef struct {
    WINDOW * win;
    float cell_max;
    size_t data_width;
    size_t data_height;
    size_t width;
    size_t height;
    size_t * steps_x;
    size_t * steps_y;
}matrix;


matrix startmtrx(size_t height, size_t width, size_t y0, size_t x0, size_t data_height, size_t data_width);
void loaddata(matrix * mtrx, float ** data, size_t data_height, size_t data_width);
void destroymtrx(matrix * mtrx);

#endif