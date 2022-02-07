extern __device__ int twice(int x);

extern "C" __global__ void caller() {
    int x = 233;
    printf("twice(%d) = %d\n", x, twice(x));
}
