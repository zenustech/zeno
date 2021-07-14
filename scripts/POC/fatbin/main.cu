#include <cstdio>
#include <helper_cuda.h>

__global__ void kernel() {
    printf("FUCK U NVIDIA\n");
}

int main() {
    kernel<<<1, 1>>>();
    checkCudaErrors(cudaDeviceSynchronize());
    return 0;
}
