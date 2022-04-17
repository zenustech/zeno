#include <memory>
#include <array>
#include "ImplSycl.h"
#include "PointerMap.h"


using namespace fdb;


class kernel0;


int main() {
    L1PointerMap<float, 1, 2, 3> arr;
    //Vector<float> arr(32);

    fdb::enqueue([&] (auto dev) {
        auto arrAxr = arr.activateAccessor<fdb::Access::discard_write>(dev);
        dev.template parallelFor<kernel0, 1>(arr.size(), [=] (size_t id) {
            arrAxr(id)() = id;
        });
    });

    {
        auto arrAxr = arr.accessor<fdb::Access::read>(fdb::HOST);
        for (int i = 0; i < arr.size(); i++) {
            printf(" %.3f", arrAxr(i)());
        }
        printf("\n");
    }

    return 0;
}
