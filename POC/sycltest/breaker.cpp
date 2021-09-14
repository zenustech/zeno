#include <CL/sycl.hpp>
#include <memory>
#include <array>
#include "vec.h"


sycl::queue &getQueue() {
    static auto p = std::make_unique<sycl::queue>();
    return *p;
}


struct Instance {
    sycl::queue que;
    struct BufInfo {
        uintptr_t base = 0;
        size_t size = 0;
        sycl::buffer<char> buf;
        bool used = true;
    };

    std::map<uintptr_t, size_t> buflut;
    std::list<BufInfo> bufs;
    uintptr_t top = 0x10000;

    void *malloc(size_t size) {
        auto ret = top;
        BufInfo buf{top, size, buf};
        top += size;
        buflut[ret] = bufs.size();
        bufs.push_back(std::move(buf));
        return (void *)ret;
    }

    void free(void *ptr) {
        auto loc = (uintptr_t)ptr;
        if (auto it = buflut.find(loc); it != buflut.end()) {
            auto &buf = bufs.at(it->second);
            buf.used = false;
        }
    }
};


int main() {
    Instance inst;

    int *dat = inst.malloc(32);

    getQueue().submit([&] (sycl::handler &cgh) {
        auto datAxr = img.accessor<fdb::Access::discard_write>(dev);
        cgh.parallel_for<kernel0>(img.shape(), [=] (sycl::item<2> id) {
            inst.
        });
    });
}
