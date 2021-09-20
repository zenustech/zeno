#include <cstdio>
//#include "impl_cuda.h"
#include "impl_host.h"
#include "HashTiledListGrid.h"

using namespace fdb;

int main() {
    HashTiledListGrid<int> a;
    a.reserve_blocks(4096);

    auto av = a.view();
    parallel_for(vec3S(16, 4, 2), [=] FDB_DEVICE (vec3S c) {
        *av.append(c / 2) = c[0];
    });

    av.parallel_foreach([=] FDB_DEVICE (vec3S c, int &val) {
        printf("%ld = %d\n", c[0], val);
    });

    synchronize();
    return 0;
}
