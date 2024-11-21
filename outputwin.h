#ifndef OUTPUTWIN_H
#define OUTPUTWIN_H

#include <stddef.h>
#include <ncurses.h>
#include "globals.h"

typedef struct{
    WINDOW * win;
    size_t_1D fields_y;
    size_t_1D fields_x;
}outputwin;

outputwin start_outputwin(size_t height, size_t width, size_t y0, size_t x0, char_2D labels, size_t labels_per_line);
void update_outputwin(outputwin outwin, char_2D values);
void destroy_outputwin(outputwin outwin);

#define update_single_outputwin(outwin, index, format, string)\
    if(index < outwin.fields_y.height){\
        mvwprintw(outwin.win,outwin.fields_y.data[index],outwin.fields_x.data[index],format,string);\
    }

#endif