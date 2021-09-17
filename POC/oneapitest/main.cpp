#include "ImplIntel.h"
#include "Vector.h"
#include <cstdio>

using namespace ImplIntel;

int main(void) {
    Queue q;
    q.print_device_info();

    Vector<int, SharedAllocator> v(100, q.shared_allocator());
    auto vAxr = v.begin();
    Vector<size_t, SharedAllocator> c(1, q.shared_allocator());
    auto cAxr = c.begin();
    cAxr[0] = 0;

    q.parallel_for(Dim3(100, 1, 1), [=](Dim3 idx) {
        size_t id = AtomicRef<size_t>(cAxr[0])++;
        //size_t id = cAxr[0]++;
        vAxr[id] = idx.x;
    });
    for (int i = 0; i < 100; i++) {
        printf("%d\n", v[i]);
    }
    return 0;
}
