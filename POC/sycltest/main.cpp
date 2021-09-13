#include <memory>
#include <array>
#include "sycl_sink.h"


using namespace fdb;


class kernel0;

int main() {
    NDArray<float, 2> img({16, 16});

    fdb::getQueue().submit([&] (fdb::DeviceHandler dev) {
        auto imgAxr = img.accessor<fdb::Access::discard_write>(dev);
        dev.parallelFor<kernel0>(img.shape(), [=] (fdb::vec2S id) {
            imgAxr(id[0], id[1]) = (id[0] + id[1]) % 2;
        });
    });

    {
        auto imgAxr = img.accessor<fdb::Access::read>(fdb::host);
        for (int j = 0; j < 16; j++) {
            for (int i = 0; i < 16; i++) {
                printf(" %.3f", imgAxr(i, j));
            }
            printf("\n");
        }
    }
}
