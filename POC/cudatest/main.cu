#include <cstdio>
#include "impl_cuda.h"
//#include "impl_host.h"
#include "Vector.h"
#include "H21B3_Grid.h"

using namespace fdb;

int main() {
#if 1
    HashMap<int, int> a;
    a.reserve(16);
    {
        auto av = a.view();
        parallel_for(32, [=] FDB_DEVICE (size_t i) {
            av[i / 2] = i;
        });

        av.parallel_foreach([=] FDB_DEVICE (size_t i, int &val) {
            printf("%ld %d\n", i, val);
        });
    }

#else
    Vector<int> a;
    a.resize(5, 40);
    {
        auto av = a.view();
        parallel_for(a.size(), [=] FDB_DEVICE (size_t i) {
            printf("- %ld %d\n", i, av[i]);
            av[i] = 42;
        });
    }
    a.resize(8, 4);
    {
        auto av = a.view();
        parallel_for(a.size(), [=] FDB_DEVICE (size_t i) {
            printf("+ %ld %d\n", i, av[i]);
        });
    }

#endif

    synchronize();
    return 0;
}
