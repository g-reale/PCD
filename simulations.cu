#include <omp.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <cuda.h>

#include "globals.cuh"
#include "simulations.cuh"

simulation start_simulation(float_2D simspace, float delta_x, float diffusion, size_t n_threads){
    simulation sim = {
        delta_x,
        diffusion,
        n_threads,
        {{0,0,NULL},{0,0,NULL}},
        0
    };
    start2D(simspace.height,simspace.width,sim.simspace[0],float);
    start2D(simspace.height,simspace.width,sim.simspace[1],float);
    for(size_t i = 0; i < simspace.height; i++){
    for(size_t j = 0; j < simspace.width; j++){
        sim.simspace[0].data[i][j] = simspace.data[i][j];
    }}
    return sim;
}

SIMULATION_INTERFACE(OMP){
    
    //prepare to return the elapsed time
    double start = omp_get_wtime();

    //handling thread count
    int old_dynamic = omp_get_dynamic();
    omp_set_dynamic(false);
    omp_set_num_threads(sim->n_threads);
    
    //unpacking the parameters for faster access
    float constant = sim->diffusion * delta_t / (sim->delta_x * sim->delta_x);
    float ** current = sim->simspace[sim->current_space].data;
    sim->current_space = !sim->current_space;
    float ** next = sim->simspace[sim->current_space].data;
    size_t height = sim->simspace[0].height;
    size_t width = sim->simspace[0].width;
    float diff = 0;

    //calculate the next iteration and the difference as required by the problem prompt
    #pragma omp parallel for reduction(+:diff)
        for(size_t i = 1; i < height - 1; i++){
        for(size_t j = 1; j < width - 1; j++){
            next[i][j] = (current[i][j] + 
                            constant * 
                            (current[i - 1][j] + 
                            current[i + 1][j] +
                            current[i][j - 1] +   
                            current[i][j + 1] - 
                            4 * current[i][j]));
            diff += fabs(next[i][j] - current[i][j]);
        }}

    
    omp_set_dynamic(old_dynamic);
    *elapsed_time = omp_get_wtime() - start;
    *difference = diff;
    return sim->simspace[sim->current_space];
} 

//base single threaded implementation
SIMULATION_INTERFACE(base){
    
    //prepare to return the elapsed time
    double start = omp_get_wtime();

    //unpacking the parameters for faster access
    float constant = sim->diffusion * delta_t / (sim->delta_x * sim->delta_x);
    float ** current = sim->simspace[sim->current_space].data;
    sim->current_space = !sim->current_space;
    float ** next = sim->simspace[sim->current_space].data;
    size_t height = sim->simspace[0].height;
    size_t width = sim->simspace[0].width;
    float diff = 0;

    //calculate the next iteration and the difference as required by the problem prompt
    for(size_t i = 1; i < height - 1; i++){
    for(size_t j = 1; j < width - 1; j++){
        next[i][j] = (current[i][j] + 
                        constant * 
                        (current[i - 1][j] + 
                        current[i + 1][j] +
                        current[i][j - 1] +   
                        current[i][j + 1] - 
                        4 * current[i][j]));
        diff += fabs(next[i][j] - current[i][j]);
    }}    

    *elapsed_time = omp_get_wtime() - start;
    *difference = diff;
    return sim->simspace[sim->current_space];
}

void destroy_simulation(simulation sim){
    destroy2D(sim.simspace[0]);
    destroy2D(sim.simspace[1]);
}

__constant__ float D_CONSTANT;
__constant__ int D_HEIGHT;
__constant__ int D_WIDTH;
__constant__ int D_THREAD;

__global__ void diffusion_kernel(float * current, float * next, float *  difference){
    extern __shared__ float shared[];
    
    int local_i = threadIdx.x;
    int local_y = local_i / D_THREAD;
    int local_x = local_i % D_THREAD;
    
    int global_i = blockDim.x * blockIdx.x + (local_y + 1) * D_WIDTH + local_x + 1;//start at coordinates (1,1)
    int global_y = global_i / D_WIDTH;
    int global_x = global_i % D_WIDTH;

    float new_val = 0.0f;
    float old_val = 0.0f;
    
    //only compute for threads inside the border
    if(global_x < D_WIDTH -1 && global_y < D_HEIGHT -1){
    
        //copy the most used data to local/shared memory
        shared[local_i] = current[global_i];
        old_val = shared[local_i]; 

        __syncthreads(); //only the 4 neighbors of a single thread need to be synchronized 
        //synchronizing everyone is overkill
        
        //perform the calculation on the center using only shared memory (the vast majority)
        if(0 < local_x && 0 < local_y && local_y < D_THREAD - 1 && local_x < D_THREAD - 1){
            new_val = old_val + 
            D_CONSTANT * 
            (shared[local_i - D_THREAD] +
                shared[local_i + D_THREAD] +
                shared[local_i - 1] +
                shared[local_i + 1] +
                4 * old_val
            );
        }
        //perform the calculations on the border that need at most 2 accesses to global memory
        // i could separate this else into various else ifs and it would be faster, but it would also be gigantic...so no 
        else{
            new_val = old_val + 
            D_CONSTANT * 
            (((local_y == 0) ?            current[global_i - D_WIDTH] : shared[local_i - D_THREAD]) +
            ((local_y == D_THREAD - 1) ?  current[global_i + D_WIDTH] : shared[local_i + D_THREAD]) + 
                    ((local_x == 0) ?             current[global_i - 1] : shared[local_i - 1]) +
                    ((local_x == D_THREAD - 1) ?  current[global_i + 1] : shared[local_i + 1]) +
                    4 * old_val
                );
        }
        next[global_i] = new_val;
    }
    else{
        shared[local_i] = 0;
    }
    
    //reduce the difference, i can reutilize the shared memory for that
    difference[local_i] = new_val - old_val;
}

__global__ void reduction_kernel(float * vector, int * size){
    extern __shared__ float shared[];
    
    int local_i = threadIdx.x;
    int global_i = blockDim.x * blockIdx.x + local_i;

    if(global_i < *size)
        shared[local_i] = vector[global_i];
    __syncthreads();

    int reduction = 2;
    int threads = blockDim.x;

    while(reduction <= threads){
        if(!(local_i%reduction))
            shared[local_i] += shared[local_i + (reduction>>1)];
        reduction = reduction << 1;
        __syncthreads();
    }

    vector[blockIdx.x] = shared[0];
}

float * d_simspace[2];
float * d_difference;
float * cpy_buffer;
int initialized = 0;
int size = 0;
int threads_n = 0;
int blocks_n = 0;
int shared_size = 0;

void start_cuda(simulation * sim, float delta_t){

    //copy the the fixed values to the constant memory
    float constant = sim->diffusion * delta_t / (sim->delta_x * sim->delta_x);
    int height = (int)sim->simspace[0].height;
    int width = (int)sim->simspace[0].width;
    int threads = (int)sim->n_threads;
    
    cudaMemcpyToSymbol(D_CONSTANT,&constant,sizeof(float));
    cudaMemcpyToSymbol(D_HEIGHT,&height,sizeof(int));
    cudaMemcpyToSymbol(D_WIDTH,&width,sizeof(int));
    cudaMemcpyToSymbol(D_THREAD,&threads,sizeof(int));

    //allocate the simulation spaces/ difference vector
    size = sizeof(float) * height * width;
    float * current = flatten(sim->simspace[sim->current_space]);
    cudaMalloc(&d_difference,size);
    cudaMalloc(&d_simspace[0], size);
    cudaMalloc(&d_simspace[1], size);
    cudaMemcpy(d_simspace[sim->current_space], current, size, cudaMemcpyHostToDevice);

    //calculate the costants
    threads_n = threads * threads;
    blocks_n = ceildiv((width-2) * (height-2), threads_n);
    shared_size = threads_n * sizeof(float);

    //allocate the intermediary host buffer
    cpy_buffer = (float*)malloc(size);
    free(current);
}

void destroy_cuda(){
    cudaFree(d_simspace[0]);
    cudaFree(d_simspace[1]);
    cudaFree(d_difference);
    free(cpy_buffer);
}

void restart_cuda(simulation * sim, float delta_t){
    if(initialized)
        destroy_cuda();
    start_cuda(sim,delta_t);
}

SIMULATION_INTERFACE(cuda){

    if(!initialized){
        start_cuda(sim,delta_t);
        initialized = 1;
    }

    //call the kernel
    diffusion_kernel<<<blocks_n,threads_n,shared_size+1>>>(d_simspace[sim->current_space],d_simspace[!sim->current_space],d_difference);

    //copy the result to the host
    sim->current_space = !sim->current_space;   
    cudaMemcpy(cpy_buffer,d_simspace[sim->current_space],size,cudaMemcpyDeviceToHost);
    
    //reduce the difference vector
    int reduction_size = blocks_n;
    int * d_size;
    cudaMalloc((void**)&d_size,sizeof(int));
    cudaMemcpy(d_size,size,sizeof(int),cudaMemcpyHostToDevice);
    while(reduction_size){
        int new_size = ceildiv(reduction_size,threads_n);
        reduction_kernel<<new_size,threads_n>>(,)
        reduction_size = 
    }

    printf("done copying\n");
    fold(cpy_buffer,sim->simspace[sim->current_space]);
    printf("done folding\n");
    return sim->simspace[sim->current_space];
}

