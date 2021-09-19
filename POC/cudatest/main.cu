#include <cstdio>
#include "impl_cuda.h"
#include "Vector.h"
#include "HashMap.h"

using namespace fdb;

int main() {
#if 1
    HashMap<int, int> a;
    a.reserve(100);
    {
        auto av = a.view();
        /*parallelFor(0, [=] FDB_DEVICE (int i) {
            printf("emplace %d %d\n", i, i * 2);
            av.emplace(i, i * 2 + 1);
        });*/

        parallelFor(1, [=] FDB_DEVICE (int i) {
            printf("at %d %d\n", i, av.at(i));
        });
    }

#else
    Vector<int> a;
    a.resize(5, 40);
    {
        auto av = a.view();
        parallelFor(a.size(), [=] FDB_DEVICE (size_t i) {
            printf("- %ld %d\n", i, av[i]);
            av[i] = 42;
        });
    }
    a.resize(8, 4);
    {
        auto av = a.view();
        parallelFor(a.size(), [=] FDB_DEVICE (size_t i) {
            printf("+ %ld %d\n", i, av[i]);
        });
    }

#endif

    synchronize();
    return 0;
}
