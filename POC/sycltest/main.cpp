#include <memory>
#include <array>
#include "sycl_sink.h"


using namespace fdb;


class kernel0;


int main() {
    ExtensibleArray<float> arr(4);

    arr.resize(16);

    /*fdb::enqueue([&] (fdb::DeviceHandler dev) {
        auto arrAxr = arr.accessor<fdb::Access::discard_write>(dev);
        dev.parallelFor<kernel0, 1>(arr.size(), [=] (size_t id) {
            arrAxr(id) = id;
        });
    });*/

    /*{
        auto arrAxr = arr.accessor<fdb::Access::read>(fdb::host);
        for (int i = 0; i < 16; i++) {
            printf(" %.3f", arrAxr(i));
        }
        printf("\n");
    }*/

    return 0;
}
