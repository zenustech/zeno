#include <memory>
#include <array>
#include "sycl_sink.h"


using namespace fdb;


class kernel0;

int main() {
    NDArray<float, 2> img({512, 512});

    fdb::getQueue().submit([&] (fdb::DeviceHandler dev) {
        auto imgAxr = img.accessor<fdb::Access::discard_write>(dev);
        dev.parallelFor<kernel0>(fdb::vec1S(16), [=] (size_t id) {
            imgAxr(id, id) = id;
        });
    });

    {
        auto imgAxr = img.accessor<fdb::Access::read>(fdb::host);
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                printf("%f\n", imgAxr(i, j));
            }
        }
    }
}
