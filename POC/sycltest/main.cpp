#include <memory>
#include <array>
#include "sycl_sink.h"


namespace fdb {

template <class T>
struct ExtensibleArray {
    NDArray<T> m_arr;
    size_t m_size = 0;

    ExtensibleArray(size_t n, T *data = nullptr)
        : m_arr(n, data)
        , m_size(n)
    {}

    template <class ArrAxr>
    struct _Accessor {
        ArrAxr m_arrAxr;

        _Accessor(ArrAxr &&arrAxr)
            : m_arrAxr(std::move(arrAxr))
        {}

        using ReferenceT = typename ArrAxr::ReferenceT;

        inline ReferenceT operator[](size_t index) const {
            return m_arrAxr[index];
        }

        inline ReferenceT operator()(size_t index) const {
            return operator[](index);
        }
    };

    template <auto Mode = Access::read_write, class Handler>
    auto accessor(Handler hand) {
        auto arrAxr = m_arr.template accessor<Mode>(hand);
        return _Accessor<std::decay_t<decltype(arrAxr)>>(std::move(arrAxr));
    }

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
    ExtensibleArray<float> arr(4);

    fdb::getQueue().submit([&] (fdb::DeviceHandler dev) {
        auto arrAxr = arr.accessor<fdb::Access::discard_write>(dev);
        dev.parallelFor<kernel0, 1>(arr.size(), [=] (size_t id) {
            arrAxr(id) = id;
        });
        arr.resize(16, dev);
    });

    {
        auto arrAxr = arr.accessor<fdb::Access::read>(fdb::host);
        for (int i = 0; i < 16; i++) {
            printf(" %.3f", arrAxr(i));
        }
        printf("\n");
    }
}
