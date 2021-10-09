#include <CL/sycl.hpp>
#include <memory>
#include <array>
#include <list>
#include "vec.h"


sycl::queue &getQueue() {
    static auto p = std::make_unique<sycl::queue>();
    return *p;
}


struct Instance {
    sycl::queue que;
    sycl::buffer<char> rootbuf{(char *)nullptr, 1024};

    struct Chunk {
        uintptr_t base = 0;
        size_t size = 0;
        bool used = true;
    };

    std::map<uintptr_t, Chunk *> chklut;
    std::list<Chunk> chks;
    uintptr_t top = 0x10000;

    void *malloc(size_t size) {
        auto ret = top;
        Chunk chk{top, size};
        top += size;
        chks.push_back(std::move(chk));
        chklut[ret] = &chks.back();
        return (void *)ret;
    }

    void free(void *ptr) {
        auto loc = (uintptr_t)ptr;
        if (auto it = chklut.find(loc); it != chklut.end()) {
            auto &chk = *it->second;
            chk.used = false;
        }
    }
};

class kernel0;

int main() {
    Instance inst;

    int *dat = (int *)inst.malloc(32 * sizeof(int));

    inst.que.submit([&] (sycl::handler &cgh) {
        auto rootaxr = inst.rootbuf.get_access<sycl::access::mode::read_write>(cgh);
        cgh.parallel_for<kernel0>(sycl::range<1>(32), [=] (sycl::item<1> id) {
            rootaxr[id[0]] = id[0];
        });
    });

    inst.free(dat);
}
