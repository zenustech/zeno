#include <cstdio>

#if defined(__OPENCL_VERSION__)
__kernel void test() {
    printf("hello from kernel!\n");
}
#endif

#if !defined(__OPENCL_VERSION__)
int main() {
    printf("hello from the host!\n");
    test();
    return 0;
}
#endif
