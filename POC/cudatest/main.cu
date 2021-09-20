#include <cstdio>
#if 1
#include "impl_cuda.h"
#else
#include "impl_host.h"
#endif
#include "HashTiledListGrid.h"

using namespace fdb;

__managed__ int total = 0;
__managed__ int count = 0;

int main() {
    HashTiledListGrid<int, 32> a;
    a.reserve_blocks(166444);

    auto av = a.view();
    parallel_for(vec3S(16, 8, 8), [=] FDB_DEVICE (vec3i c) {
        *av.append(c / 4) = c[0] + c[1] * 2 + c[2] * 4;
    });

    av.parallel_foreach([=] FDB_DEVICE (vec3i c, int &val) {
        printf("%d %d %d = %d\n", c[0], c[1], c[2], val);
        atomic_add(&total, val);
        atomic_add(&count, 1);
    });

    synchronize();
    printf("29184 = %d\n", total);
    printf("1024 = %d\n", count);
    return 0;
}
