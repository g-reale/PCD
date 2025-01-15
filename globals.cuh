#ifndef GLOBALS_H
#define GLOBALS_H

#include <ncurses.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>

#define declare1D(type)\
typedef struct {\
    size_t height;\
    type * data;\
} type##_1D;

#define start1D(h,var,type)\
    var.height = h;\
    var.data = (type*)malloc(sizeof(type) * h);

#define destroy1D(var)\
    var.height = 0;\
    free(var.data);\

#define declare2D(type)\
typedef struct{\
    size_t height;\
    size_t width;\
    type ** data;\
} type##_2D\

#define start2D(h,w,var,type)\
    var.height = h;\
    var.width = w;\
    var.data = (type**)malloc(sizeof(type*) * h);\
    for(size_t i = 0; i < h; i++){\
        var.data[i] = (type*)malloc(sizeof(type) * w);\
    }

#define destroy2D(var)\
    for(size_t i = 0; i < var.height; i++)\
        free(var.data[i]);\
    free(var.data);\
    var.height = 0;\
    var.width = 0;

declare1D(size_t);
declare1D(float);
declare1D(char);
declare2D(size_t);
declare2D(float);
declare2D(char);

static const char_1D GRAY_SCALE = {
    71,
    (char *)"$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'."
};

static const char_1D GRAY_SCALE_1 = {
    10,
    (char *)" .:-=+*#%@"
};

#define clamp(value,min,max)\
    (value) < (min) ? (min) : (max) < (value) ? (max) : (value)

#define ceildiv(numerator,denominator)\
    ((numerator / denominator) + (numerator % denominator > 0))

#define ceil2div(numerator)\
    ((numerator+1)>>1)

static inline char getgrayscale(char_1D grayscale,float brightness){
    size_t index = fabs(brightness * (float)grayscale.height);
    index = index < grayscale.height ? index : grayscale.height-1; 
    return grayscale.data[index];
}

static inline char getarrowkeys(int key, size_t * cursor_y, size_t * cursor_x, size_t y_min, size_t  x_min, size_t y_max, size_t x_max){
    switch (key){
        case KEY_UP:
            *cursor_y = clamp((*cursor_y)-1,y_min,y_max);
            break;
        
        case KEY_DOWN:
            *cursor_y = clamp((*cursor_y)+1,y_min,y_max);
            break;

        case KEY_LEFT:
            *cursor_x = clamp((*cursor_x)-1,x_min,x_max);
            break;
        
        case KEY_RIGHT:
            *cursor_x = clamp((*cursor_x)+1,x_min,x_max);
            break;
        
        default:
            return 0;
    }
    return key;
} 

// static inline int ceildiv(int numerator, int denominator){
//     return (numerator / denominator) + (numerator % denominator > 0);
// }

static inline float * flatten(float_2D matrix){
    float * flatend = (float*)malloc(sizeof(float) * matrix.height * matrix.width);
    size_t k = 0;
    for(size_t i = 0; i < matrix.height; i++){
    for(size_t j = 0; j < matrix.width; j++,k++)
        flatend[k] = matrix.data[i][j];
    }
    return flatend;
}

static inline void fold(float * flatend, float_2D matrix){
    size_t k = 0;
    for(size_t i = 0; i < matrix.height; i++){
    for(size_t j = 0; j < matrix.width; j++,k++)
        matrix.data[i][j] = flatend[k];
    }
} 

#endif


