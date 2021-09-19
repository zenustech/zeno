#include <cstdio>
//#include "impl_cuda.h"
#include "impl_host.h"
#include "Vector.h"
#include "HashMap.h"

using namespace fdb;

int main() {
#if 1
    HashMap<int, int> a;
    a.reserve(4099);
    {
        auto av = a.view();
        parallel_for(4097, [=] FDB_DEVICE (int i) {
            i = (114514 * i) + 31415;
            av.emplace(i, i * 2 + 1);
        });

        av.parallel_foreach([=] FDB_DEVICE (int k, int &v) {
            if (k * 2 + 1 != v) {
                printf("error: %d != %d\n", k * 2 + 1, v);
            }
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
