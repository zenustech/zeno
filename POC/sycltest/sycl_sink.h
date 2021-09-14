#pragma once

#include <CL/sycl.hpp>
#include "vec.h"


namespace fdb {


static constexpr struct HostHandler {} host;


struct DeviceHandler {
    sycl::handler *m_cgh;

    DeviceHandler(sycl::handler &cgh) : m_cgh(&cgh) {}

    template <class Key, size_t Dim, class Kernel>
    void parallelFor(vec<Dim, size_t> range, Kernel kernel) const {
        m_cgh->parallel_for<Key>(vec_to_other<sycl::range<Dim>>(range), [=] (sycl::id<Dim> idx) {
            auto id = other_to_vec<Dim>(idx);
            kernel(std::as_const(id));
        });
    }

    template <size_t Dim, class Kernel>
    void parallelFor(vec<Dim, size_t> range, Kernel kernel) const {
        parallelFor<Kernel>(range, kernel);
    }
};


struct CommandQueue {
    sycl::queue m_que;

    template <bool IsBlocked = true, class Functor>
    void submit(Functor const &functor) {
        auto event = m_que.submit([&] (sycl::handler &cgh) {
            DeviceHandler dev(cgh);
            functor(std::as_const(dev));
        });
        if constexpr (IsBlocked) {
            event.wait();
        }
    }

    void wait() {
        m_que.wait();
    }
};


enum class Access {
    read = (int)sycl::access::mode::read,
    write = (int)sycl::access::mode::write,
    read_write = (int)sycl::access::mode::read_write,
    discard_write = (int)sycl::access::mode::discard_write,
    discard_read_write = (int)sycl::access::mode::discard_read_write,
    atomic = (int)sycl::access::mode::atomic,
};


template <class T, size_t Dim = 1>
struct NDArray {
    static_assert(Dim > 0);

    sycl::buffer<T, Dim> m_buffer;
    vec<Dim, size_t> m_shape;

    NDArray() = default;

    explicit NDArray(vec<Dim, size_t> shape, T *data = nullptr)
        : m_buffer(data, vec_to_other<sycl::range<Dim>>(shape))
        , m_shape(shape)
    {}

    auto const &shape() const {
        return m_shape;
    }

    void reshape(vec<Dim, size_t> shape) {
        m_buffer = sycl::buffer<T, Dim>(
                (T *)nullptr, vec_to_other<sycl::range<Dim>>(shape));
        m_shape = shape;
    }

    template
        < auto Mode = sycl::access::mode::read_write
        , auto Target = sycl::access::target::global_buffer>
    struct _Accessor {
        sycl::accessor<T, Dim, Mode, Target> m_axr;

        template <class ...Args>
        _Accessor(NDArray &parent, Args &&...args)
            : m_axr(parent.m_buffer.template get_access<Mode>(
                        std::forward<Args>(args)...))
        {}

        using ReferenceT = std::conditional_t<Mode == sycl::access::mode::read,
              T const &, T &>;

        inline ReferenceT operator[](vec<Dim, size_t> indices) const {
            return m_axr[vec_to_other<sycl::id<Dim>>(indices)];
        }

        template <class ...Indices>
        inline ReferenceT operator()(Indices &&...indices) const {
            return operator[](vec<Dim, size_t>(std::forward<Indices>(indices)...));
        }
    };

    template <auto Mode = Access::read_write>
    auto accessor(HostHandler hand) {
        return _Accessor<(sycl::access::mode)(int)Mode,
               sycl::access::target::host_buffer>(*this);
    }

    template <auto Mode = Access::read_write>
    auto accessor(DeviceHandler hand) {
        return _Accessor<(sycl::access::mode)(int)Mode,
               sycl::access::target::global_buffer>(*this, *hand.m_cgh);
    }
};


CommandQueue &getQueue();


template <class T>
static void __partial_memcpy(HostHandler
        , sycl::buffer<T, 1> &dst
        , sycl::buffer<T, 1> &src
        , size_t n
        ) {
    auto dstAxr = dst.template get_access<sycl::access::mode::write>();
    auto srcAxr = dst.template get_access<sycl::access::mode::read>();
    for (size_t id = 0; id < n; id++) {
        dstAxr[id] = srcAxr[id];
    }
}

template <class T>
class __partial_memcpy_kernel;

template <class T>
static void __partial_memcpy(DeviceHandler dev
        , sycl::buffer<T, 1> &dst
        , sycl::buffer<T, 1> &src
        , size_t n
        ) {
    auto dstAxr = dst.template get_access<sycl::access::mode::write>(*dev.m_cgh);
    auto srcAxr = dst.template get_access<sycl::access::mode::read>(*dev.m_cgh);
    dev.parallelFor<__partial_memcpy_kernel<T>>([=] (size_t id) {
        dstAxr[id] = srcAxr[id];
    });
}


}
