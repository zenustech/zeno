#pragma once

#include "common.h"
#include <CL/sycl.hpp>
#include <optional>


namespace fdb::__ImplSycl {


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

    template <class Key, class Kernel>
    void singleTask(Kernel kernel) const {
        m_cgh->single_task<Key>([=] {
            kernel();
        });
    }
};


template <class Functor>
void enqueue(Functor const &functor,
        sycl::queue q = sycl::queue{}, bool blocked = true) {
    auto event = q.submit([&] (sycl::handler &cgh) {
        DeviceHandler dev(cgh);
        functor(std::as_const(dev));
    });
    if (blocked) {
        event.wait();
    }
}


inline static constexpr sycl::access::mode __to_sycl_mode(Access mode) {
    switch (mode) {
        case Access::read: return sycl::access::mode::read;
        case Access::write: return sycl::access::mode::write;
        case Access::read_write: return sycl::access::mode::read_write;
        case Access::discard_write: return sycl::access::mode::discard_write;
        case Access::discard_read_write: return sycl::access::mode::discard_read_write;
        case Access::atomic: return sycl::access::mode::atomic;
    }
    return sycl::access::mode::read_write;
}


template <class T>
class __partial_memcpy_kernel;

template <class T>
static void __partial_memcpy
        ( sycl::buffer<T, 1> &dst
        , sycl::buffer<T, 1> &src
        , size_t n
        ) {
    enqueue([&] (DeviceHandler dev) {
        auto dstAxr = dst.template get_access<sycl::access::mode::write>(*dev.m_cgh);
        auto srcAxr = src.template get_access<sycl::access::mode::read>(*dev.m_cgh);
        dev.parallelFor<__partial_memcpy_kernel<T>, 1>(n, [=] (size_t id) {
            dstAxr[id] = srcAxr[id];
        });
    });
}

template <class T, size_t Dim>
class __fully_memcpy_kernel;

template <class T, size_t Dim>
static void __fully_memcpy
        ( sycl::buffer<T, Dim> &dst
        , sycl::buffer<T, Dim> &src
        , vec<Dim, size_t> shape
        ) {
    enqueue([&] (DeviceHandler dev) {
        auto dstAxr = dst.template get_access<sycl::access::mode::discard_write>(*dev.m_cgh);
        auto srcAxr = src.template get_access<sycl::access::mode::read>(*dev.m_cgh);
        dev.parallelFor<__fully_memcpy_kernel<T, Dim>, Dim>(shape, [=] (vec<Dim, size_t> idx) {
            auto id = vec_to_other<sycl::id<Dim>>(idx);
            dstAxr[id] = srcAxr[id];
        });
    });
}

template <class T, size_t Dim, class ...Args>
class __fully_meminit_kernel;

template <class T, size_t Dim, class ...Args>
static void __fully_meminit
        ( sycl::buffer<T, Dim> &dst
        , vec<Dim, size_t> shape
        , Args ...args
        ) {
    enqueue([&] (DeviceHandler dev) {
        auto dstAxr = dst.template get_access<sycl::access::mode::discard_write>(*dev.m_cgh);
        dev.parallelFor<__fully_meminit_kernel<T, Dim, Args...>, Dim>(shape, [=] (vec<Dim, size_t> idx) {
            auto id = vec_to_other<sycl::id<Dim>>(idx);
            new (&dstAxr[id]) T(args...);
        });
    });
}

template <class T, size_t Dim>
class __fully_memdeinit_kernel;

template <class T, size_t Dim>
static void __fully_memdeinit
        ( sycl::buffer<T, Dim> &dst
        , vec<Dim, size_t> shape
        ) {
    if constexpr (!std::is_trivially_destructible<T>::value) {
        enqueue([&] (DeviceHandler dev) {
            auto dstAxr = dst.template get_access<sycl::access::mode::discard_read_write>(*dev.m_cgh);
            dev.parallelFor<__fully_memdeinit_kernel<T, Dim>, Dim>(shape, [=] (vec<Dim, size_t> idx) {
                auto id = vec_to_other<sycl::id<Dim>>(idx);
                dstAxr[id].~T();
            });
        });
    }
}

template <class T, class ...Args>
class __partial_meminit_kernel;

template <class T, class ...Args>
static void __partial_meminit
        ( sycl::buffer<T, 1> &dst
        , size_t nbeg
        , size_t nend
        , Args ...args
        ) {
    enqueue([&] (DeviceHandler dev) {
        auto dstAxr = dst.template get_access<sycl::access::mode::write>(*dev.m_cgh);
        dev.parallelFor<__partial_meminit_kernel<T, Args...>, 1>(nend - nbeg, [=] (size_t id) {
            new (&dstAxr[nbeg + id]) T(args...);
        });
    });
}

template <class T>
class __partial_memdeinit_kernel;

template <class T>
static void __partial_memdeinit
        ( sycl::buffer<T, 1> &dst
        , size_t nbeg
        , size_t nend
        ) {
    if constexpr (!std::is_trivially_destructible<T>::value) {
        enqueue([&] (DeviceHandler dev) {
            auto dstAxr = dst.template get_access<sycl::access::mode::read_write>(*dev.m_cgh);
            dev.parallelFor<__partial_memdeinit_kernel<T>, 1>(nend - nbeg, [=] (size_t id) {
                dstAxr[nbeg + id].~T();
            });
        });
    }
}


template <class T, size_t Dim = 1>
struct __NDBuffer {
    static_assert(Dim > 0);

    sycl::buffer<T, Dim> m_buffer;
    vec<Dim, size_t> m_shape;

    __NDBuffer() = default;
    __NDBuffer(__NDBuffer const &) = delete;
    __NDBuffer &operator=(__NDBuffer const &) = delete;
    __NDBuffer(__NDBuffer &&) = default;
    __NDBuffer &operator=(__NDBuffer &&) = default;

    explicit __NDBuffer(vec<Dim, size_t> shape = {0})
        : m_buffer((T *)nullptr, vec_to_other<sycl::range<Dim>>(shape))
        , m_shape(shape)
    {}

    template <class ...Args>
    void construct(Args const &...args) {
        __fully_meminit<T, Dim>(m_buffer, m_shape, args...);
    }

    void destroy() {
        __fully_memdeinit<T, Dim>(m_buffer, m_shape);
    }

    auto const &shape() const {
        return m_shape;
    }

    void reshape(vec<Dim, size_t> shape) {
        m_buffer = sycl::buffer<T, Dim>(
                (T *)nullptr, vec_to_other<sycl::range<Dim>>(shape));
        m_shape = shape;
    }

    __NDBuffer clone() const {
        __NDBuffer ret(m_shape);
        __fully_memcpy<T, Dim>(ret.m_buffer, const_cast<
                sycl::buffer<T, Dim> &>(m_buffer), m_shape);
        return ret;
    }

    template
        < auto Mode = sycl::access::mode::read_write
        , auto Target = sycl::access::target::global_buffer>
    struct __Accessor {
        sycl::accessor<T, Dim, Mode, Target> m_axr;

        template <class ...Args>
        __Accessor(__NDBuffer &parent, Args &&...args)
            : m_axr(parent.m_buffer.template get_access<Mode>(
                        std::forward<Args>(args)...))
        {}

        struct __Proxy {
            sycl::accessor<T, Dim, Mode, Target> const &m_axr;
            sycl::id<Dim> m_id;

            using RetType = std::conditional_t<Mode == sycl::access::mode::atomic,
                decltype(std::declval<sycl::accessor<T, Dim, Mode, Target> &>()
                        [std::declval<sycl::id<Dim>>()]),
                std::conditional_t<Mode == sycl::access::mode::read,
                T const &, T &>>;

            RetType operator()() const {
                return m_axr[m_id];
            }

            __Proxy &operator=(T const &val) const {
                m_axr[m_id] = val;
                return *this;
            }
        };

        inline __Proxy operator()(vec<Dim, size_t> indices) const {
            auto id = vec_to_other<sycl::id<Dim>>(indices);
            return __Proxy{m_axr, id};
        }
    };

    template <auto Mode = Access::read_write>
    auto accessor(HostHandler hand) {
        return __Accessor<__to_sycl_mode(Mode),
               sycl::access::target::host_buffer>(*this);
    }

    template <auto Mode = Access::read_write>
    auto accessor(DeviceHandler hand) {
        return __Accessor<__to_sycl_mode(Mode),
               sycl::access::target::global_buffer>(*this, *hand.m_cgh);
    }
};


#if 0
template <class T, size_t ...Ns>
struct NDArray {
    inline static constexpr size_t Dim = sizeof...(Ns);
    static_assert(Dim > 0);

    __NDBuffer<T, Dim> m_buf;

    template <class ...Args>
    explicit NDArray(Args const &...args)
        : m_buf(vec<Dim, size_t>(Ns...))
    {
        m_buf.construct(args...);
    }

    NDArray(NDArray &&) = default;

    NDArray &operator=(NDArray const &other) const {
        m_buf = other.m_buf.clone();
        return *this;
    }

    NDArray &operator=(NDArray &&) = default;

    ~NDArray() {
        m_buf.destroy();
    }

    static inline constexpr auto size() {
        return vec<Dim, size_t>(Ns...);
    }

    template <auto Mode = Access::read_write, class Handler>
    auto accessor(Handler hand) {
        auto bufAxr = m_buf.template accessor<Mode>(hand);
        return [=] (vec<Dim, size_t> index) -> T * {
            return bufAxr(index);
        };
    }
};
#endif

template <class T>
struct Vector {
    __NDBuffer<T> m_buf;
    size_t m_size = 0;

    explicit Vector(__NDBuffer<T> &&buf)
        : m_buf(std::move(buf))
        , m_size(m_buf.shape())
    {}

    Vector()
        : Vector(0)
    {}

    template <class ...Args>
    explicit Vector(size_t n, Args const &...args)
        : m_buf(std::max((size_t)1, n))
        , m_size(n)
    {
        m_buf.construct(args...);
    }

    Vector(Vector const &other)
        : m_buf(other.clone())
        , m_size(other.m_size)
    {}

    Vector(Vector &&) = default;

    Vector &operator=(Vector const &other) const {
        m_buf = other.m_buf.clone();
        return *this;
    }

    Vector &operator=(Vector &&) = default;

    ~Vector() {
        m_buf.destroy();
    }

    template <auto Mode = Access::read_write, class Handler>
    auto accessor(Handler hand) {
        auto bufAxr = m_buf.template accessor<Mode>(hand);
        return [=] (size_t index) {
            return bufAxr(vec1S(index));
        };
    }

    size_t size() const {
        return m_size;
    }

    size_t capacity() const {
        return m_buf.shape();
    }

    void __recapacity(size_t n) {
        auto old_buffer = std::move(m_buf.m_buffer);
        m_buf.reshape(std::max((size_t)1, n));
        __partial_memcpy<T>(m_buf.m_buffer, old_buffer, m_size);
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

    template <class ...Args>
    void resize(size_t n, Args const &...args) {
        if (n == 0)
            return clear();
        reserve(n);
        if (m_size < n) {
            if (m_size == 0)
                __fully_meminit<T>(m_buf.m_buffer, n, args...);
            else
                __partial_meminit<T>(m_buf.m_buffer, m_size, n, args...);
        } else if (m_size > n) {
            __partial_memdeinit<T>(m_buf.m_buffer, n, m_size);
        }
        m_size = n;
    }

    void clear() {
        __fully_memdeinit<T>(m_buf.m_buffer, m_size);
        m_size = 0;
    }
};


}


namespace fdb {

using namespace __ImplSycl;

}
