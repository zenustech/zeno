#include <CL/sycl.hpp>
#include <optional>
#include <memory>
#include <array>
#include "vec.h"

using namespace fdb;


struct OutOfMemoryException : std::exception {
    const char *what() noexcept {
        return "SYCL runs out of memory";
    }
};


struct Instance {
    sycl::queue m_queue;

    sycl::buffer<char> m_rootBuffer;
    size_t m_maxSize;

    std::optional<std::decay_t<decltype(m_rootBuffer.get_access<sycl::access::mode::read_write>(std::declval<sycl::handler &>()))>> m_deviceAxr;
    std::optional<std::decay_t<decltype(m_rootBuffer.get_access<sycl::access::mode::read_write>())>> m_hostAxr;

    Instance(size_t maxSize = 0x100000)  // 256 MB
        : m_rootBuffer((char *)nullptr, maxSize)
        , m_maxSize(maxSize)
    {
        {
            m_hostAxr = m_rootBuffer.get_access<sycl::access::mode::read_write>();
            m_hostOffset = &m_hostAxr.value()[0];
        }

        m_queue.submit([&] (sycl::handler &cgh) {
            m_deviceAxr = m_rootBuffer.get_access<sycl::access::mode::read_write>(cgh);
            m_deviceOffset = &m_deviceAxr.value()[0];
        }).wait();
    }

    char *m_deviceOffset = nullptr;
    char *m_hostOffset = nullptr;
    uintptr_t m_watermark = 0x0;

    void *deviceToHost(void *ptr) {
        return (void *)(m_hostOffset + ((char *)ptr - m_deviceOffset));
    }

    void *hostToDevice(void *ptr) {
        return (void *)(m_deviceOffset + ((char *)ptr - m_hostOffset));
    }

    void *allocate(size_t size) {
        uintptr_t res = m_watermark;
        if (m_watermark + size > m_maxSize)
            throw OutOfMemoryException();
        m_watermark += size;
        return (void *)(m_deviceOffset + res);
    }

    void free(void *ptr) {
        (void)ptr;
    }

    template <class Key, class Kernel>
    void parallelFor(vec3S dim, Kernel kernel) {
        m_queue.submit([&] (sycl::handler &cgh) {
            cgh.parallel_for<Key>(
                    sycl::range<3>(dim[0], dim[1], dim[2]),
                    [=] (sycl::item<3> id) {
                        vec3S idx(id[0], id[1], id[2]);
                        //kernel(std::as_const(idx));
                    });
        }).wait();
    }
};

class kernel0;

int main() {
    Instance inst;

    uintptr_t result = (uintptr_t)inst.allocate(32 * sizeof(int));

    inst.parallelFor<kernel0>({32, 1, 1}, [=] (vec3S idx) {
        ((int *)result)[idx[0]] = idx[0];
    });

    auto h_result = inst.deviceToHost((void *)result);
    for (int i = 0; i < 32; i++) {
        printf("%d\n", ((int *)h_result)[i]);
    }

    return 0;
}
