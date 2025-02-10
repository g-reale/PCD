#include "MPIconfig.h"

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void startMPI(int * argc, char *** argv){
    int this;
    MPI_Init(argc,argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &this);
    printf("%d\n",this);
    MPI_Finalize();
    exit(0);
}