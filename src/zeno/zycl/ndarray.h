#pragma once

#include <zeno/zycl/core.h>
#include <zeno/zycl/instance.h>
#include <zeno/zycl/helper.h>


#ifndef ZENO_WITH_SYCL

ZENO_NAMESPACE_BEGIN
namespace zycl {

template <class T>
struct ndarray : std::vector<T> {
    using std::vector<T>::vector;

    template <access::mode mode>
    auto get_access(auto &&cgh) {
        return functor_accessor([this] (id<1> idx) -> decltype(auto) {
            return (*this)[idx];
        });
    }

    inline auto &as_vector() {
        return static_cast<std::vector<T> &>(*this);
    }

    inline auto const &to_vector() const {
        return static_cast<std::vector<T> const &>(*this);
    }
};

}
ZENO_NAMESPACE_END

#else

ZENO_NAMESPACE_BEGIN
namespace zycl {

template <int N>
inline constexpr range<N> _M_nozerosize(range<N> const &shape) {
    return [] <std::size_t ...Is> (range<N> const &shape, std::index_sequence<Is...>) {
        return range<N>(std::max((size_t)1, shape[Is])...);
    }(shape, std::make_index_sequence<N>{});
}

template <int N>
inline constexpr bool _M_isempty(range<N> const &shape) {
    return [] <std::size_t ...Is> (range<N> const &shape, std::index_sequence<Is...>) {
        return !shape[Is] || ... || false;
    }(shape, std::make_index_sequence<N>{});
}

template <class T, int N>
inline void _M_transfer(buffer<T, N> &buf_src, buffer<T, N> &buf_dst) {
    default_queue().submit([&] (handler &cgh) {
        auto src_acc = buf_src.template get_access<access::mode::read>(cgh);
        auto dst_acc = buf_dst.template get_access<access::mode::discard_write>(cgh);
        cgh.copy(src_acc, dst_acc);
    });
}

template <class T, int N>
inline void _M_fillwith(buffer<T, N> &buf_dst, T const &val) {
    default_queue().submit([&] (handler &cgh) {
        auto dst_acc = buf_dst.template get_access<access::mode::discard_write>(cgh);
        cgh.fill(dst_acc, val);
    });
}

template <class T, int N>
    requires (N >= 1 && std::is_trivially_copy_constructible_v<T> && std::is_trivially_destructible_v<T>)
struct ndarray {
    mutable buffer<T, N> _M_buf;
    range<N> _M_shape;

    ndarray(ndarray const &) = default;
    ndarray &operator=(ndarray const &) = default;
    ndarray(ndarray &&) = default;
    ndarray &operator=(ndarray &&) = default;

    void reshape(range<N> const &shape, T const &val = {}) {
        _M_buf = buffer<T, N>(_M_nozerosize(shape));
        if (!_M_isempty(shape))
            _M_fillwith(_M_buf, val);
        _M_shape = shape;
    }

    bool empty() const {
        return _M_isempty(_M_shape);
    }

    range<N> const &shape() const {
        return _M_shape;
    }

    void clear() {
        _M_buf = buffer<T, 1>(_M_nozerosize(range<N>{}));
        _M_shape = range<N>{};
    }

    ndarray() : _M_buf(_M_nozerosize(range<N>{})), _M_shape(range<N>{}) {
    }

    explicit ndarray(range<N> const &shape, T const &val = {})
        : _M_buf(_M_nozerosize(shape)), _M_shape(shape)
    {
        if (!_M_isempty(shape))
            _M_fillwith(_M_buf, val);
    }

    template <access::mode mode>
    auto get_access(auto &&cgh) const {
        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(cgh)>, host_handler>)
            return _M_buf.template get_access<mode>();
        else
            return _M_buf.template get_access<mode>(cgh);
    }

    buffer<T, N> &get_buffer() const {
        return _M_buf;
    }
};

}
ZENO_NAMESPACE_END

#endif
