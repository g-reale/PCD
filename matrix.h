#ifndef MATRIX_H
#define MATRIX_H

#include <ncurses.h>
#include <stddef.h>
#include "globals.h"

typedef struct {
    WINDOW * win;
    float cell_max;
    size_t data_height;
    size_t data_width;
    size_t_1D steps_y;
    size_t_1D steps_x;

    // size_t height;
    // size_t width;
    // size_t * steps_y;
    // size_t * steps_x;
}matrix;


matrix start_mtrx(size_t height, size_t width, size_t y0, size_t x0, size_t data_height, size_t data_width);
void update_mtrx(matrix mtrx, float_2D simspace);
void configure_mtrx(matrix mt);
void destroy_mtrx(matrix mtrx);

#endif