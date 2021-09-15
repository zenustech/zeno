#include <memory>
#include <array>
#include "ImplSycl.h"
#include "PointerMap.h"


using namespace fdb;


class kernel0;


int main() {
#if 1
    //L1PointerMap<float, 1, 2, 3> arr;
    L1DenseMap<float, 1, 2, 3> arr;

    fdb::enqueue([&] (auto dev) {
        auto arrAxr = arr.accessor<fdb::Access::discard_write>(dev);
        dev.template parallelFor<kernel0, 1>(arr.size(), [=] (size_t id) {
            *arrAxr(id) = id;
        });
    });

    {
        auto arrAxr = arr.accessor<fdb::Access::read>(fdb::HOST);
        for (int i = 0; i < arr.size(); i++) {
            printf(" %.3f", *arrAxr(i));
        }
        printf("\n");
    }
#else
    NDBuffer<size_t> arr(32);
    arr.construct(FDB_BAD_OFFSET);

    {
        auto arrAxr = arr.accessor<fdb::Access::read>(fdb::HOST);
        for (int i = 0; i < arr.shape(); i++) {
            printf(" %zd", (size_t)*arrAxr(i));
        }
        printf("\n");
    }
#endif

    return 0;
}
