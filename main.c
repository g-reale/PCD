#include <ncurses.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "matrix.h"
#include "globals.h"

#define N 1000
#define M 2000

int main(){
    initscr();
    float ** data = (float**)malloc(sizeof(float*) * M);
    for(size_t i = 0; i < M; i++){
        data[i] = (float*)malloc(sizeof(float) * N);
        for(size_t j = 0; j < N; j++)
            data[i][j] =  j * i;
    }

    matrix mtrx = startmtrx(30,120,0,0,N,M);
    loaddata(&mtrx,data,N,M);
    loaddata(&mtrx,data,N,M);
    refresh();
    getch();
    destroymtrx(&mtrx);
    
    endwin();
    for(size_t i = 0; i < M; i++)
        free(data[i]);
    free(data);
}

//gcc matrix.c main.c -o main -fsanitize=address -lncurses -lm 