#include <cstdio>
#include "impl_cuda.h"

using namespace fdb;

__global__ void a() { printf("a\n"); }

int main() {
    parallelFor(vec3S(2, 2, 4), [=] FDB_DEVICE (vec3S idx) {
        printf("are you ok? %ld %ld %ld\n", idx[0], idx[1], idx[2]);
    });
    synchronize();
    return 0;
}
