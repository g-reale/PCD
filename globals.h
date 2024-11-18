#ifndef GLOBALS_H
#define GLOBALS_H

#include <ncurses.h>
#include <stddef.h>

static const char GRAY_SCALE_CHRS[] = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'.";
static const size_t GRAY_SCALE_SIZE = 71;  
static const char GRAY_SCALE_CHRS_2[] =  " .:-=+*#%@";
static const size_t GRAY_SCALE_SIZE_2 = 10;  


static inline char getgrayscale(float brightness){
    return GRAY_SCALE_CHRS[(int)(brightness * GRAY_SCALE_SIZE)];
}

static inline char getgrayscale2(float brightness){
    return GRAY_SCALE_CHRS_2[(int)(brightness * GRAY_SCALE_SIZE_2)];
}

// static inline void setgrayscale(WINDOW * win, float brightness){
//     char num[4];
//     sprintf(num,"%d",(int)brightness);
//     wprintw(win, "\033[38;2;255;0;0m");
// }

// static const int CUSTOM_PAIR = 1;
// static const int CUSTOM_COLOR = COLOR_CYAN;
// static short _r, _g, _b;
// static short _current_atribute = SHRT_MAX;

// static inline void pushcolor(){color_content(CUSTOM_COLOR,&_r,&_g,&_b);}
// static inline void popcolor(){init_color(CUSTOM_COLOR,_r,_g,_b);}
// static inline void setcolor(WINDOW * win, short r, short g, short b){
//     init_color(CUSTOM_COLOR,r,g,b);
//     init_pair(CUSTOM_PAIR,CUSTOM_COLOR,COLOR_BLACK);
//     wattron(win,COLOR_PAIR(CUSTOM_PAIR));
// }

// static inline void initgraycolorspace(){
//     pushcolor();
//     for (short i = 0; i < 1000; i++) {
//         init_color(CUSTOM_PAIR, i, i, i);
//         init_pair(i + 1, CUSTOM_PAIR, COLOR_BLACK);
//     }
//     popcolor();
// }
// static inline void setgrayscale(WINDOW * win, float brightness){
//     short aux = (short)brightness; 

//     if(_current_atribute != SHRT_MAX)
//         attroff(_current_atribute);
    
//     _current_atribute = aux;
//     wattron(win,COLOR_PAIR(_current_atribute));


//     attr_t attrs;
//     short current_color_pair;
//     wattr_get(win,&attrs, &current_color_pair, NULL);
//     fprintf(stderr, "Brightness: %f, Using color pair: %d, Can change color: %d\n",
//             brightness, aux, can_change_color());
// }

#endif