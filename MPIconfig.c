#include "MPIconfig.h"
#include "simulations.h"

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

MPI_context * start_MPI_context(int * argc, char *** argv){
    static int initialized = 0;
    
    if(initialized)
        return NULL;

    initialized = 1;
    MPI_context * context = (MPI_context*)malloc(sizeof(MPI_context));
    MPI_Init(argc,argv);
    MPI_Comm_size(MPI_COMM_WORLD, &context->population); 
    MPI_Comm_rank(MPI_COMM_WORLD, &context->me);

    context->predecessor = context->me - 1;
    context->sucessor = context->me + 1;
    return context;
}

void destroy_MPI_context(MPI_context * context){
    free(context);
    MPI_Finalize();
    exit(0);
}