#include <cstdio>
#if 1
#include "impl_cuda.h"
#else
#include "impl_host.h"
#endif
#include "HashListGrid.h"

using namespace fdb;

__managed__ int count = 0;

int main() {
    const int n = 8192 * 2;
    HashListGrid<int> a;
    a.reserve_blocks(n);

    auto av = a.view();
    parallel_for(n, [=] FDB_DEVICE (size_t c) {
        av.append(vec3i((c * 114514 + 31415) % 8, 0, 0)) = c;
    });

#if 0
    av.parallel_foreach([=] FDB_DEVICE (vec3i c, int &val) {
        printf("%d = %d\n", c[0], val);
        atomic_add(&count, 1);
    });
#else
    av.parallel_foreach_leaf([=] FDB_DEVICE (vec3i coord, auto &leaf) {
        leaf.foreach([&] (int &val) {
            printf("%d = %d\n", coord[0], val);
            atomic_add(&count, 1);
        });
    });
#endif

    synchronize();
    printf("%d = %d\n", n, count);
    return 0;
}
