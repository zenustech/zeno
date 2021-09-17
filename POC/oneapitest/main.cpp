//#include "ImplIntel.h"
#include "ImplHost.h"
#include "Vector.h"
#include <cstdio>

using namespace ImplHost;

int main(void) {
    Queue q;

    Vector<int, Allocator> v(100, q.allocator());
    auto vAxr = v.begin();
    Vector<size_t, Allocator> c(1, q.allocator());
    auto cAxr = c.begin();
    cAxr[0] = 0;

    q.parallel_for(Dim3(100, 1, 1), [=](Dim3 idx) {
        size_t id = make_atomic_ref(cAxr[0])++;
        vAxr[id] = idx.x;
    });
    for (int i = 0; i < 100; i++) {
        printf("%d\n", v[i]);
    }
    return 0;
}
