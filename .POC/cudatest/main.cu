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
#if 1
    const int n = 8192 * 16;
    HashListGrid<int> a;
    a.reserve_blocks(65);

    auto av = a.view();
    parallel_for(n, [=] FDB_DEVICE (size_t i) {
        vec3i coord((i * 114514 + 31415) % 64, 0, 0);
        av.append(coord, i);
    });

#if 0
    av.parallel_foreach([=] FDB_DEVICE (vec3i coord, int &val) {
        printf("%d = %d\n", coord[0], val);
        atomic_add(&count, 1);
    });
#else
    av.parallel_foreach_leaf([=] FDB_DEVICE (vec3i coord, auto &leaf) {
        leaf.foreach_load([&] (int val) {
            //printf("%d = %d\n", coord[0], val);
            atomic_add(&count, 1);
        });
    });
#endif
    synchronize();
    printf("%d = %d\n", n, count);

#else
    Vector<int> a;
    a.resize(1024);

    auto av = a.view();
    parallel_for(a.size(), [=] FDB_DEVICE (size_t i) {
        printf("%d\n", av[i]);
    });

#endif
    return 0;
}
