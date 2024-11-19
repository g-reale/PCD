#include <ncurses.h>
#include <stddef.h>
#include <math.h>
#include <stdlib.h>
#include "outputwin.h"

outputwin start_outputwin(size_t height, size_t width, size_t y0, size_t x0, char_2D labels, size_t labels_per_line){

    //override the user's options if they are malformed
    if(height < 2)
        height = 2;

    if(width < 2)
        width = 2;

    if(labels_per_line * (height-2) < labels.height)
        labels_per_line = labels.height * (height-2);

    outputwin outwin = {
        newwin(height,width,y0,x0),
        {0,NULL},
        {0,NULL},
    };

    //refresh to tell ncurses about the new window
    refresh();

    //print the labels and save the positions where data should be displayed during update 
    start1D(labels.height,outwin.fields_y,size_t);
    start1D(labels.height,outwin.fields_x,size_t);
    size_t labels_on_line = 0;
    size_t step_x = (width-2) / labels_per_line;
    size_t x = 1;
    size_t y = 1;

    for(size_t i = 0; i < labels.height; i++){
        if(labels_on_line == labels_per_line){
            labels_on_line = 0;
            y++;
            x=1;
        }
        
        mvwprintw(outwin.win,y,x,"%s",labels.data[i]);
        getyx(outwin.win,outwin.fields_y.data[i],outwin.fields_x.data[i]);
        labels_on_line++;
        x += step_x;
    }

    //add a box and refresh
    box(outwin.win,'|','-');
    wrefresh(outwin.win);
    return outwin;
}

void update_outputwin(outputwin outwin,char_2D values){
    //update the data fields
    size_t iterations = values.height < outwin.fields_y.height ? values.height : outwin.fields_y.height;
    for(size_t i = 0; i < iterations; i++)
        mvwprintw(outwin.win,outwin.fields_y.data[i],outwin.fields_x.data[i],"%s",values.data[i]);
    wrefresh(outwin.win);
}

void destroy_outputwin(outputwin outwin){
    delwin(outwin.win);
    destroy1D(outwin.fields_y);
    destroy1D(outwin.fields_x);
}