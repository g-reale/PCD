#ifndef MPI_H
#define MPI_H

#include <mpi.h>

typedef struct{
    int me;
    int predecessor;
    int sucessor;
    int population;
}MPI_context;

MPI_context * start_MPI_context(int * argc, char *** argv);
void destroy_MPI_context(MPI_context * context);
// void updateMPI();

#endif