#include <CL/sycl.hpp>
#include <iostream>
#include <memory>
#include <array>
#include "vec.h"


struct HostHandler {};


struct DeviceHandler {
    sycl::handler *m_cgh;

    DeviceHandler(sycl::handler &cgh) : m_cgh(&cgh) {}
};


template <class T, size_t Dim = 1>
struct NDArray {
    static_assert(Dim > 0, "dimension of NDArray can't be 0");

    sycl::buffer<T, Dim> m_buffer;
    vec<Dim, size_t> m_shape;

    NDArray() = default;

    explicit NDArray(vec<Dim, size_t> shape, T *data = nullptr)
        : m_buffer(data, vec_to_other<sycl::range<Dim>>(shape))
        , m_shape(shape)
    {}

    template <std::enable_if_t<Dim == 1, int> = 0>
    explicit NDArray(size_t length, T *data = nullptr)
        : NDArray(vec<1, size_t>(length), data)
    {}

    auto const &shape() const {
        return m_shape;
    }

    void reshape(vec<Dim, size_t> shape) {
        m_buffer = sycl::buffer<T, Dim>(
                (T *)nullptr, vec_to_other<sycl::range<Dim>>(shape));
        m_shape = shape;
    }

    template <std::enable_if_t<Dim == 1, int> = 0>
    void reshape(size_t length) {
        reshape(sycl::range<Dim>(length));
    }

    template <sycl::access::mode Mode, sycl::access::target Target>
    struct Accessor {
        sycl::accessor<T, Dim, Mode, Target> acc;

        template <class ...Args>
        Accessor(NDArray &parent, Args &&...args)
            : acc(parent.m_buffer.template get_access<Mode>(
                        std::forward<Args>(args)...))
        {}

        using ReferenceT = std::conditional_t<Mode == sycl::access::mode::read,
              T const &, T &>;

        inline ReferenceT operator[](vec<Dim, size_t> indices) const {
            return acc[vec_to_other<sycl::id<Dim>>(indices)];
        }

        template <std::enable_if_t<Dim == 1, int> = 0>
        inline ReferenceT operator[](size_t index) const {
            return operator[](vec<1, size_t>(index));
        }
    };

    template <auto Mode = sycl::access::mode::read_write>
    auto accessor(HostHandler = {}) {
        return Accessor<Mode, sycl::access::target::host_buffer>(*this);
    }

    template <auto Mode = sycl::access::mode::read_write>
    auto accessor(DeviceHandler dev) {
        return Accessor<Mode, sycl::access::target::global_buffer>(*this, *dev.m_cgh);
    }
};


class kernel0;

int main() {
    sycl::queue que;

    NDArray<int> arr(16);

    que.submit([&] (sycl::handler &cgh) {
        DeviceHandler dev(cgh);
        auto arrAxr = arr.accessor(dev);
        cgh.parallel_for<kernel0>(sycl::range<1>(16), [=] (sycl::id<1> id) {
            arrAxr[id[0]] = id[0];
        });
    });
    que.wait();

    {
        auto arrAxr = arr.accessor();
        for (int i = 0; i < 16; i++) {
            printf("%d\n", arrAxr[i]);
        }
    }
}
