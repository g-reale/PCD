#ifndef GLOBALS_H
#define GLOBALS_H

#include <stddef.h>
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

#define clamp(value,min,max)\
    (value) < (min) ? (min) : (max) < (value) ? (max) : (value)


#define ceilDiv(num,denum)\
    (num + denum - 1) / (denum)

#endif