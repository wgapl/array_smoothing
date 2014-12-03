#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "compare.h"
#include "gputimer.h"

// Reference
__global__ void smooth(float * v_new, const float * v) {
    int myIdx = threadIdx.x + blockDim.x * blockIdx.x;
    int numThreads = blockDim.x * gridDim.x;
    int myLeftIdx = (myIdx == 0) ? 0 : myIdx - 1;
    int myRightIdx = (myIdx == (numThreads - 1)) ? numThreads - 1 : myIdx + 1;
    float myElt = v[myIdx];
    float myLeftElt = v[myLeftIdx];
    float myRightElt = v[myRightIdx];
    v_new[myIdx] = 0.25f * myLeftElt + 0.5f * myElt + 0.25f * myRightElt;
    //printf("%d %f %f\n", myIdx, v[myIdx], v_new[myIdx]);
}

// Your code
__global__ void smooth_shared(float * v_new, const float * v) {
    extern __shared__ float s[];
    // TODO: Complete the rest of this function
    int myId = threadIdx.x + blockIdx.x*blockDim.x;
    s[myId] = v[myId];
    __syncthreads();
    if (myId == 0) {
        v_new[myId] = 0.25f * s[myId] + 0.5f * s[myId] + 0.25f * s[myId+1];
        return;
    }
    if (myId == gridDim.x * blockDim.x - 1) {
        v_new[myId] = 0.25f * s[myId-1] + 0.5f * s[myId] + 0.25f * s[myId];
        return;
    }
    if (((myId + 1) % blockDim.x) == 0) {
        v_new[myId] = 0.25f * v[myId-1] + 0.5f * v[myId] + 0.25f * v[myId+1];
        return;
    }
    if ( (myId % blockDim.x) == 0) {
        v_new[myId] = 0.25f * v[myId-1] + 0.5f * v[myId] + 0.25f * v[myId+1];
        return;
    }
    
    v_new[myId] = 0.25f * s[myId-1] + 0.5f * s[myId] + 0.25f * s[myId+1];
    //printf("%d %f %f  %f\n", myId, s[myId], v[myId], v_new[myId]);
}

int main(int argc, char **argv)
{

    const int ARRAY_SIZE = 4096;
    const int BLOCK_SIZE = 256;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    // generate the input array on the host
    float h_in[ARRAY_SIZE];
    float h_cmp[ARRAY_SIZE];
    float h_out[ARRAY_SIZE];
    float h_out_shared[ARRAY_SIZE];
    for(int i = 0; i < ARRAY_SIZE; i++) {
        // generate random float in [0, 1]
        h_in[i] = (float)random()/(float)RAND_MAX;
    }
    for(int i = 0; i < ARRAY_SIZE; i++) {
        h_cmp[i] = (0.25f * h_in[(i == 0) ? 0 : i-1] +
                    0.50f * h_in[i] +
                    0.25f * h_in[(i == (ARRAY_SIZE - 1)) ? ARRAY_SIZE - 1 : i+1]);
    }

    // declare GPU memory pointers
    float * d_in, * d_out, * d_out_shared;

    // allocate GPU memory
    cudaMalloc((void **) &d_in, ARRAY_BYTES);
    cudaMalloc((void **) &d_out, ARRAY_BYTES);
    cudaMalloc((void **) &d_out_shared, ARRAY_BYTES);

    // transfer the input array to the GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice); 

    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // launch the kernel
    smooth<<<ARRAY_SIZE / BLOCK_SIZE, BLOCK_SIZE>>>(d_out, d_in);
    GpuTimer timer;
    timer.Start();
    smooth_shared<<<ARRAY_SIZE / BLOCK_SIZE, BLOCK_SIZE, ARRAY_SIZE * sizeof(float)>>>(d_out_shared, d_in);
    timer.Stop();

    printf("Your code executed in %g ms\n", timer.Elapsed());
    // cudaEventSynchronize(stop);
    // float elapsedTime;
    // cudaEventElapsedTime(&elapsedTime, start, stop);    

    // copy back the result from GPU
    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_shared, d_out_shared, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    /*for (int k = 0; k < ARRAY_SIZE; ++k)
    {
        //std::cout << h_out[k] << " " << h_cmp[k] << std::endl;
        printf("%d %f %f\n", k, h_out[k], h_cmp[k]);
    }
    */
    // testing for correctness
    compare(h_in, h_out, h_out_shared, h_cmp, ARRAY_SIZE);

    // free GPU memory allocation
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_out_shared);
}
