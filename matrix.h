#include <ncurses.h>

//only a single matrix must exist
bool initialized = false;
typedef struct {
    WINDOW * win;
    float cell_min = 0;
    float cell_max = 0;
    unsigned int data_width = 0;
    unsigned int data_heigth = 0;
    float h_ratio;
    float w_ratio;
}matrix;


matrix startmtrx(unsigned int heigth, unsigned int width, unsigned int y0, unsigned int x0);
void loaddata(matrix * mtrx, float ** data, unsigned int heigth, unsigned int width);
void destroymtrx(,matrix * mtrx);