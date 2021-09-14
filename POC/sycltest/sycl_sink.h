#pragma once

#include <CL/sycl.hpp>
#include "vec.h"
#include <memory>


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


/*static sycl::queue &__get_sycl_queue() {
    static auto p = std::make_unique<sycl::queue>();
    return *p;
}*/


template <bool IsBlocked = true, class Functor>
void enqueue(Functor const &functor) {
    auto event = sycl::queue().submit([&] (sycl::handler &cgh) {
        DeviceHandler dev(cgh);
        functor(std::as_const(dev));
    });
    if constexpr (IsBlocked) {
        event.wait();
    }
}


enum class Access {
    read = (int)sycl::access::mode::read,
    write = (int)sycl::access::mode::write,
    read_write = (int)sycl::access::mode::read_write,
    discard_write = (int)sycl::access::mode::discard_write,
    discard_read_write = (int)sycl::access::mode::discard_read_write,
    atomic = (int)sycl::access::mode::atomic,
};


template <class T>
class __partial_memcpy_kernel;

template <class T>
static void __partial_memcpy
        ( sycl::buffer<T, 1> &dst
        , sycl::buffer<T, 1> &src
        , size_t n
        ) {
    enqueue([&] (fdb::DeviceHandler dev) {
        auto dstAxr = dst.template get_access<sycl::access::mode::write>(*dev.m_cgh);
        auto srcAxr = src.template get_access<sycl::access::mode::read>(*dev.m_cgh);
        dev.parallelFor<__partial_memcpy_kernel<T>, 1>(n, [=] (size_t id) {
            dstAxr[id] = srcAxr[id];
        });
    });
}

template <class T>
class __fully_memcpy_kernel;

template <class T, size_t Dim>
static void __fully_memcpy
        ( sycl::buffer<T, Dim> &dst
        , sycl::buffer<T, Dim> &src
        , vec<Dim, size_t> shape
        ) {
    enqueue([&] (fdb::DeviceHandler dev) {
        auto dstAxr = dst.template get_access<sycl::access::mode::discard_write>(*dev.m_cgh);
        auto srcAxr = src.template get_access<sycl::access::mode::read>(*dev.m_cgh);
        dev.parallelFor<__fully_memcpy_kernel<T>, Dim>(shape, [=] (vec<Dim, size_t> idx) {
            auto id = vec_to_other<sycl::id<Dim>>(idx);
            dstAxr[id] = srcAxr[id];
        });
    });
}


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

    NDArray clone() const {
        NDArray ret(m_shape);
        __fully_memcpy<T, Dim>(ret.m_buffer, const_cast<
                sycl::buffer<T, Dim> &>(m_buffer), m_shape);
        return ret;
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


template <class T>
struct Vector {
    NDArray<T> m_arr;
    size_t m_size = 0;

    Vector(NDArray<T> &&arr)
        : m_arr(std::move(arr))
        , m_size(m_arr.shape())
    {}

    Vector(size_t n, T *data = nullptr)
        : m_arr(n, data)
        , m_size(n)
    {}

    Vector clone() const {
        Vector ret(m_arr.clone());
        return ret;
    }

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

    void __recapacity(size_t n) {
        auto old_buffer = std::move(m_arr.m_buffer);
        m_arr.reshape(n);
        __partial_memcpy(m_arr.m_buffer, old_buffer, m_size);
    }

    void reserve(size_t n) {
        if (n > capacity()) {
            __recapacity(n);
        }
    }

    void shrink_to_fit() {
        if (capacity() > m_size) {
            __recapacity(m_size);
        }
    }

    void resize(size_t n) {
        reserve(n);
        m_size = n;
    }

    void clear() {
        m_size = 0;
    }
};


}
