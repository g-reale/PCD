#include <cuda.h>
#include <stddef.h>

__constant__ float D_CONSTANT;
__constant__ size_t D_HEIGHT;
__constant__ size_t D_WIDTH;

__global__ diffusion_kernel(float * current, float * next, float * difference){
    extern __shared__ float shared[];
    
    size_t global_x = 1 + blockDim.x * blockIdx.x + threadIdx.x;
    size_t global_y = 1 + blockDim.y * blockIdx.y + threadIdx.y;
    size_t global_i = global_y + D_HEIGHT * global_x;

    //if the thread is out of bounds return
    if(D_WIDTH <= global_x || D_HEIGHT <= global_y)
        return;

    //copy the most used data to local memory
    size_t local_x = threadIdx.x;
    size_t local_y = threadIdx.y;
    size_t local_i = local_y + blockDim.y * local_x;
    shared[local_i] = current[global_i];
    __syncthreads(); //only the 4 neighbors of a single thread need to be synchronized 
                     //synchronizing everyone is overkill

    //discard the threads around the border of the matrix
    if(0 == global_x || 0 == global_y || (D_HEIGHT-1) == global_y || (D_WIDTH-1) == global_x)
        return;

    //perform the calculation on the center using only shared memory (the vast majority)
    if(1 < local_x || 1 < local_y || local_y < (D_HEIGHT-2) || local_x < (D_WIDTH-2)){
        next[global_y][global_x] = shared[local_y][local_x] + 
                                    D_CONSTANT * 
                                    (shared[local_i-1] +
                                    shared[local_i+1] +
                                    shared[local_i - blockDim.x] +
                                    shared[local_i + blockDim.x] +
                                    4 * shared[local_i]
                                );
    }
    //perform the calculations on the border that need a single access to global memory
    //i could separate this else into various else ifs and it would be faster, but it would also be gigantic...so no 
    else{
        next[global_y][global_x] = shared[local_y][local_x] + 
                                    D_CONSTANT * 
                                    (local_y == 1 ?            current[global_i-1] : shared[local_i-1] +
                                    local_y == D_HEIGHT - 1 ?  current[global_i+1] : shared[local_i+1] + 
                                    local_x == 1 ?             current[global_i - D_WIDTH] : shared[local_i - blockDim.x] +
                                    local_x == D_WIDTH - 1 ?   current[global_y + D_WIDTH] : shared[local_i + blockDim.x] +
                                    4 * shared[local_i]
                                    );
    }

    //reduce the difference, i can reutilize the shared memory for that
    
    
}

SIMULATION_INTERFACE(cuda){
    
    //unpack the data and flatten the matrices
    size_t size = sizeof(float) * height * width;
    float * current = flatten(sim->simspace[sim->current_space]);
    sim->current_space = !sim->current_space;
    
    size_t threads = sim->n_threads;
    float constant = sim->diffusion * delta_t / (sim->delta_x * sim->delta_x);
    size_t height = sim->simspace[0].height;
    size_t width = sim->simspace[0].width;
    float diff = 0;


    //copy the diffusion constant and simpaces sizes to the device's constant memory space
    //could do this asyncly...
    cudaMemcpyToSymbol(D_CONSTANT,&constant,sizeof(float));
    cudaMemcpyToSymbol(D_HEIGHT,&HEIGHT,sizeof(size_t));
    cudaMemcpyToSymbol(D_WIDTH,&width,sizeof(size_t));


    //copy the current and next simulation spaces to global memory
    float * d_current;
    float * d_next;
    cudaMalloc(&d_current, size);
    cudaMalloc(&d_next, size);
    cudaMemcpy(d_current, current, size, cudaMemcpyHostToDevice);
    
    //calculate the correct block and thread indexers
    dim3 threads(threads,threads);
    dim3 blocks(ceildiv(width-2,threads),ceildiv(height-2,threads));

    //call the kernel
    size_t shared_size = threads * threads * sizeof(float);
    diffusion_kernel<<<blocks,threads,shared_size>>>(d_current,d_next);
}