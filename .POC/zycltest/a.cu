#include <cuda_runtime.h>


// reduce to get the sum on gpu
__global__ void reduce_sum(float *d_in, float *d_out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        d_out[idx] = d_in[idx];
    __syncthreads();
    for (int s = 1; s < blockDim.x; s *= 2)
    {
        int t = 2 * s * blockDim.x;
        if (idx < t)
        {
            d_out[idx] += d_out[idx + s * blockDim.x];
        }
        __syncthreads();
    }
}

// invoke the reduce kernel
void reduce_sum_gpu(float *h_in, float *h_out, int n)
{
    int n_blocks = (n + blockDim.x - 1) / blockDim.x;
    int n_threads = blockDim.x;
    reduce_sum<<<n_blocks, n_threads>>>(h_in, h_out, n);
}