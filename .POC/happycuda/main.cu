#include "helper_cuda.h"
__global__ void a(){printf("hello\n");}
int main() {a<<<1,1>>>();checkCudaErrors(cudaDeviceSynchronize());}
