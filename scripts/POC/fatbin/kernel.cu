extern "C" __device__ void callee();

extern "C" __global__ void caller() {
    printf("caller\n");
    callee();
    printf("caller\n");
}
