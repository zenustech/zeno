#if !defined(__OPENCL__)
#include <cstdio>
#endif

#if defined(__OPENCL__)
__kernel void test() {
    //printf("hello from kernel!\n");
}
#endif

#if !defined(__OPENCL__)
int main() {
    printf("hello from the host!\n");
    test();
    return 0;
}
#endif
