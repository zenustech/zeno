#include <memory>
#include <array>
#include "sycl_sink.h"


using namespace zinc;


static CommandQueue &getQueue() {
    static auto p = std::make_unique<CommandQueue>();
    return *p;
}


class kernel0;

int main() {
    NDArray<int> arr(16);

    getQueue().submit([&] (DeviceHandler dev) {
        auto arrAxr = arr.accessor<Access::discard_write>(dev);
        dev.parallelFor<kernel0>((size_t)16, [=] (size_t id) {
            arrAxr[id] = id;
        });
    });

    {
        auto arrAxr = arr.accessor<Access::read>();
        for (int i = 0; i < 16; i++) {
            printf("%d\n", arrAxr[i]);
        }
    }
}
