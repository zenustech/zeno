#include <cstdio>
#include "impl_host.h"
#include "Vector.h"

using namespace fdb;

__global__ void a() { printf("a\n"); }

int main() {
    Vector<int> a;
    a.resize(5, 40);
    auto av = a.view();
    parallelFor(a.size(), [=] FDB_DEVICE (size_t i) {
        printf("- %ld %d\n", i, av[i]);
        av[i] = 42;
    });
    a.resize(8, 4);
    parallelFor(a.size(), [=] FDB_DEVICE (size_t i) {
        printf("+ %ld %d\n", i, av[i]);
    });
    synchronize();
    return 0;
}
