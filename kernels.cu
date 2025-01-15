#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <cuda.h>

#include "globals.cuh"
#include "kernels.cuh"

float * d_simspace[2];
float * d_difference;
float * cpy_buffer;
int * d_size;
int initialized = 0;
int size = 0;
int threads_n = 0;
int blocks_n = 0;
int shared_size = 0;

__constant__ float D_CONSTANT;
__constant__ int D_HEIGHT;
__constant__ int D_WIDTH;
__constant__ int D_THREAD;
__constant__ int D_BLOCKS_PER_LINE;
__constant__ int D_SIZE;

__global__ void diffusion_kernel(float * current, float * next, float * difference){
    extern __shared__ float shared[];

    int local_i = threadIdx.x;
    int local_x = threadIdx.x % D_THREAD;
    int local_y = threadIdx.x / D_THREAD;
    
    int block_x = blockIdx.x % D_BLOCKS_PER_LINE;
    int block_y = blockIdx.x / D_BLOCKS_PER_LINE;

    int global_x = block_x * D_THREAD + local_x;
    int global_y = block_y * D_THREAD + local_y;
    int global_i = global_x + global_y * D_WIDTH;

    shared[local_i] = global_i < D_SIZE ? current[global_i] : 0;
    __syncthreads();

    float up =      local_y ? shared[local_i - D_THREAD] :
                    global_y ?  current[global_i - D_WIDTH] : 
                    0;

    float down =    local_y < D_THREAD - 1?  shared[local_i + D_THREAD] :
                    global_y < D_HEIGHT - 1 ? current[global_i + D_WIDTH] : 
                    0;

    float left =    local_x ? shared[local_i - 1] :
                    global_x ? current[global_i - 1] :
                    0;

    float rigth =   local_x < D_THREAD - 1 ? shared[local_i + 1] :
                    global_x < D_WIDTH -1 ? current[global_i + 1] :
                    0;   
    
    float me = shared[local_i];    

    float new_me = me + D_CONSTANT * (up + down + left + rigth - 4 * me);
    next[global_i] = new_me;
    float diff = me - new_me;
    diff = diff < 0 ? -diff : diff;
    difference[blockDim.x * blockIdx.x + threadIdx.x] = diff;
}

// __global__ void diffusion_kernel(float * current, float * next, float * difference){
    
//     int global_i = blockDim.x * blockIdx.x + threadIdx.x + D_WIDTH + 1;
//     int global_x = global_i % D_WIDTH;
//     int global_y = global_i / D_WIDTH;

//     float up = D_WIDTH <= global_i ?  current[global_i - D_WIDTH] : 0;
//     float down = global_i + D_WIDTH < D_WIDTH * D_HEIGHT ? current[global_i + D_WIDTH] : 0;
//     float left = 1 <= global_i ? current[global_i - 1] : 0;
//     float rigth = global_i < D_WIDTH * D_HEIGHT ? current[global_i + 1] : 0;
//     float me = current[global_i];

//     float new_me = me + D_CONSTANT * (up + down + left + rigth - 4 * me);
//     next[global_i] = new_me;
//     float diff = me - new_me;
//     diff = diff < 0 ? -diff : diff;
//     difference[blockDim.x * blockIdx.x + threadIdx.x] = diff;
// }

__global__ void reduction_kernel(float * difference, int * size){
    extern __shared__ float shared[];

    int local_i = threadIdx.x;
    int global_i = blockDim.x * blockIdx.x + threadIdx.x; 

    int len = *size;
    shared[local_i] = global_i < len ? difference[global_i] : 0; 
    
    int stride = ceil2div(blockDim.x);
    while(1 < stride){
        __syncthreads();
        if(local_i + stride < blockDim.x){
            // printf("[%d - %d,%d]: [%d - %d]%f += [%d - %d]%f\n",blockIdx.x,blockIdx.x * blockDim.x, (blockIdx.x+1) * blockDim.x - 1,local_i,global_i,shared[local_i],local_i + stride,global_i + stride,shared[local_i + stride]);
            shared[local_i] += shared[local_i + stride];

            // if(blockIdx.x == 0){
            //     if(local_i == 0 ){
            //         printf("\nstride: %d\n", stride);
            //     }
            // }
            
        }

        stride = ceil2div(stride);
    }

    if (local_i == 0){
        // printf("[%d] = %f\n",blockIdx.x,shared[0] + shared[1]);
        difference[blockIdx.x] = shared[0] + shared[1];
    }
}

void start_cuda(simulation * sim, float delta_t){
    initialized = 1;

    //copy the the fixed values to the constant memory
    float constant = sim->diffusion * delta_t / (sim->delta_x * sim->delta_x);
    int height = (int)sim->simspace[0].height;
    int width = (int)sim->simspace[0].width;
    int threads = (int)sim->n_threads;
    int blocks_per_line = (width + threads - 1) / threads;
    int space = height * width;
    
    cudaMemcpyToSymbol(D_CONSTANT,&constant,sizeof(float));
    cudaMemcpyToSymbol(D_HEIGHT,&height,sizeof(int));
    cudaMemcpyToSymbol(D_WIDTH,&width,sizeof(int));
    cudaMemcpyToSymbol(D_THREAD,&threads,sizeof(int));
    cudaMemcpyToSymbol(D_BLOCKS_PER_LINE,&blocks_per_line,sizeof(int));
    cudaMemcpyToSymbol(D_SIZE,&space,sizeof(int));

    //allocate the simulation spaces/ difference vector
    size = sizeof(float) * height * width;
    float * current = flatten(sim->simspace[sim->current_space]);
    cudaMalloc((void**)&d_size,sizeof(int));
    cudaMalloc((void**)&d_difference,size);
    cudaMalloc((void**)&d_simspace[0], size);
    cudaMalloc((void**)&d_simspace[1], size);
    cudaMemcpy(d_simspace[sim->current_space], current, size, cudaMemcpyHostToDevice);

    //calculate the costants
    threads_n = threads * threads;
    // blocks_n = ceildiv((width-2) * (height-2), threads_n);
    blocks_n = ceildiv((width) * (height), threads_n);
    shared_size = threads_n * sizeof(float);

    //allocate the intermediary host buffer
    cpy_buffer = (float*)malloc(size);
    free(current);
}

void destroy_cuda(){
    initialized = 0;
    cudaFree(d_simspace[0]);
    cudaFree(d_simspace[1]);
    cudaFree(d_difference);
    cudaFree(d_size);
    free(cpy_buffer);
}

void restart_cuda(simulation * sim, float delta_t){
    if(initialized)
        destroy_cuda();
    start_cuda(sim,delta_t);
}

float * diffusion_cuda(int id){
    id = id % 2;
    diffusion_kernel<<<blocks_n,threads_n,shared_size+1>>>(d_simspace[id],d_simspace[!id],d_difference);
    cudaMemcpy(cpy_buffer,d_simspace[!id],size,cudaMemcpyDeviceToHost);
    return cpy_buffer;
}

void reduction_cuda(float * difference){
    int reduction_size = size/sizeof(float);
    while(1 < reduction_size){
        cudaMemcpy(d_size,&reduction_size,sizeof(int),cudaMemcpyHostToDevice);
        reduction_size = ceildiv(reduction_size,threads_n);
        // printf("kernel call!\n");
        reduction_kernel<<<reduction_size,threads_n,shared_size + 1>>>(d_difference,d_size);
    }
    cudaMemcpy(difference,d_difference,sizeof(float),cudaMemcpyDeviceToHost);
}

// SIMULATION_INTERFACE(cuda){

//     //prepare to return the elapsed time
//     double start = omp_get_wtime();

//     //call the kernel
//     diffusion_kernel<<<>>>(d_simspace[sim->current_space],d_simspace[!sim->current_space],);
    
//     //copy the result to the host
//     sim->current_space = !sim->current_space;   
//     cudaMemcpy(cpy_buffer,d_simspace[sim->current_space],size,cudaMemcpyDeviceToHost);

//     // reduce the difference
//     int reduction_size = size/sizeof(float);
//     while(1 < reduction_size){
//         cudaMemcpy(d_size,&reduction_size,sizeof(int),cudaMemcpyHostToDevice);
//         reduction_size = ceildiv(reduction_size,threads_n);
//         reduction_kernel<<<reduction_size,threads_n,shared_size + 1>>>(d_difference,d_size);
//     }
//     cudaMemcpy(difference,d_difference,sizeof(float),cudaMemcpyDeviceToHost);
//     *elapsed_time = omp_get_wtime() - start;

//     //fold the simspace to the expected format 
//     fold(cpy_buffer,sim->simspace[sim->current_space]);
//     return sim->simspace[sim->current_space];
// }