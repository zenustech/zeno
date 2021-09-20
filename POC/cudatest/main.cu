#include <cstdio>
#include "impl_cuda.h"
//#include "impl_host.h"
#include "HashTiledListGrid.h"

using namespace fdb;

__managed__ int count = 0;

int main() {
    HashTiledListGrid<int> a;
    a.reserve_blocks(4096);

    auto av = a.view();
    parallel_for(vec3S(2, 2, 16), [=] FDB_DEVICE (vec3i c) {
        *av.append(vec3i(0, 0, 0)) = c[0] + c[1] * 2 + c[2] * 4;
    });

    av.parallel_foreach([=] FDB_DEVICE (vec3i c, int &val) {
        printf("%d %d %d = %d\n", c[0], c[1], c[2], val);
        atomic_add(&count, val);
    });

    synchronize();
    printf("%d\n", count);
    return 0;
}
