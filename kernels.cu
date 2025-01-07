// #include <cuda.h>
// #include <stddef.h>
// #include "simulations.cuh"
// #include "globals.cuh"

// __constant__ float D_CONSTANT;
// __constant__ size_t D_HEIGHT;
// __constant__ size_t D_WIDTH;

// __global__ void diffusion_kernel(float * current, float * next){
//     extern __shared__ float shared[];
    
//     size_t global_x = 1 + blockDim.x * blockIdx.x + threadIdx.x;
//     size_t global_y = 1 + blockDim.y * blockIdx.y + threadIdx.y;
//     size_t global_i = D_WIDTH * global_y + global_x;

//     //discard the threads around the border of the matrix
//     if(D_WIDTH - 1 <= global_x || D_HEIGHT - 1<= global_y)
//         return;

//     //copy the most used data to local/shared memory
//     size_t local_x = threadIdx.x;
//     size_t local_y = threadIdx.y;
//     size_t local_i = blockDim.x * local_y + local_x;
//     shared[local_i] = current[global_i];
//     __syncthreads(); //only the 4 neighbors of a single thread need to be synchronized 
//                      //synchronizing everyone is overkill

//     //perform the calculation on the center using only shared memory (the vast majority)
//     if(0 < local_x || 0 < local_y || local_y < blockDim.y - 1 || local_x < blockDim.x - 1){
//         next[global_i] = shared[local_i] + 
//                          D_CONSTANT * 
//                          (shared[local_i - blockDim.x] +
//                          shared[local_i + blockDim.x] +
//                          shared[local_i - 1] +
//                          shared[local_i + 1] +
//                          4 * shared[local_i]
//                          );
//     }
//     //perform the calculations on the border that need a single access to global memory
//     //i could separate this else into various else ifs and it would be faster, but it would also be gigantic...so no 
//     else{
//         next[global_i] = shared[local_i] + 
//                          D_CONSTANT * 
//                          (((local_y == 1) ?               current[global_i - D_WIDTH] : shared[local_i - blockDim.x]) +
//                           ((local_y == blockDim.y - 1) ?  current[global_i + D_WIDTH] : shared[local_i + blockDim.x]) + 
//                           ((local_x == 1) ?               current[global_i - 1] : shared[local_i - 1]) +
//                           ((local_x == blockDim.x - 1) ?  current[global_i + 1] : shared[local_i + 1]) +
//                          4 * shared[local_i]
//                          );
//     }

//     //reduce the difference, i can reutilize the shared memory for that
    
// }

// float * d_simspace[2];
// float * cpy_buffer;
// size_t initialized = 0;
// size_t size = 0;

// void start_cuda(simulation * sim, float delta_t){

//     //copy the the fixed values to the constant memory
//     float constant = sim->diffusion * delta_t / (sim->delta_x * sim->delta_x);
//     size_t height = sim->simspace[0].height;
//     size_t width = sim->simspace[0].width;
//     cudaMemcpyToSymbol(D_CONSTANT,&constant,sizeof(float));
//     cudaMemcpyToSymbol(D_HEIGHT,&height,sizeof(size_t));
//     cudaMemcpyToSymbol(D_WIDTH,&width,sizeof(size_t));

//     //allocate the simulation spaces
//     size = sizeof(float) * height * width;
//     float * current = flatten(sim->simspace[sim->current_space]);
//     float * next = flatten(sim->simspace[!sim->current_space]);
//     cudaMalloc(&d_simspace[0], size);
//     cudaMalloc(&d_simspace[1], size);
//     cudaMemcpy(d_simspace[sim->current_space], current, size, cudaMemcpyHostToDevice);

//     //alocate the intermediary host buffer
//     cpy_buffer = (float*)malloc(size);

//     free(current);
//     free(next);
// }


// void destroy_cuda(){
//     cudaFree(d_simspace[0]);
//     cudaFree(d_simspace[1]);
//     free(cpy_buffer);
// }

// void restart_cuda(simulation * sim, float delta_t){
//     if(initialized)
//         destroy_cuda();
//     start_cuda(sim,delta_t);
// }

// extern SIMULATION_INTERFACE(cuda){

//     if(!initialized){
//         start_cuda(sim,delta_t);
//         initialized = 1;
//     }

//     size_t height = sim->simspace[0].height;
//     size_t width = sim->simspace[0].width;
//     size_t threads = sim->n_threads;
//     //float diff = 0;
    
//     //calculate the correct block and thread indexers
//     dim3 threads_n(threads,threads);
//     dim3 blocks_n(ceildiv(width-2,threads),ceildiv(height-2,threads));

//     //call the kernel
//     size_t shared_size = threads * threads * sizeof(float);
//     diffusion_kernel<<<blocks_n,threads_n,shared_size>>>(d_simspace[sim->current_space],d_simspace[!sim->current_space]);
//     sim->current_space = !sim->current_space;
    
//     //copy the result to the host
//     cudaMemcpy(cpy_buffer,d_simspace[sim->current_space],size,cudaMemcpyDeviceToHost);
//     fold(cpy_buffer,sim->simspace[sim->current_space]);
//     return sim->simspace[sim->current_space];
// }

