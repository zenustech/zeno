#include <memory>
#include <array>
#include "sycl_sink.h"


namespace fdb {

template <class T>
struct ExtensibleArray {
    NDArray<T> m_arr;
    size_t m_size = 0;

    size_t size() const {
        return m_size;
    }

    size_t capacity() const {
        return m_arr.shape();
    }

    template <class Handler>
    void __recapacity(size_t n, Handler hand) {
        auto old_buffer = std::move(m_arr.m_buffer);
        m_arr.reshape(n);
        __partial_memcpy(hand, m_arr.m_buffer, old_buffer, m_size);
    }

    template <class Handler>
    void reserve(size_t n, Handler hand) {
        if (n > capacity()) {
            __recapacity(n, hand);
        }
    }

    template <class Handler>
    void shrink_to_fit(Handler hand) {
        if (capacity() > m_size) {
            __recapacity(m_size, hand);
        }
    }

    template <class Handler>
    void resize(size_t n, Handler hand) {
        reserve(n, hand);
        m_size = n;
    }

    void clear() {
        m_size = 0;
    }
};

}


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
