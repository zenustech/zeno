#pragma once

#include <zeno/zycl/zycl.h>
#include <type_traits>
#include <optional>
#include <utility>


ZENO_NAMESPACE_BEGIN
namespace zycl {

struct host_handler {};

template <access::mode mode, class Cgh>
auto make_access(Cgh &&cgh, auto &&buf) {
    return buf.template get_access<mode>(cgh);
}

template <access::mode mode>
auto make_access(auto &buf) {
    return make_access<mode>(host_handler{}, buf);
}

template <class F>
struct functor_accessor {
    F f;

    constexpr functor_accessor(F &&f) : f(std::forward<F>(f)) {
    }

    constexpr decltype(auto) operator[](auto &&t) {
        return f(std::forward<decltype(t)>(t));
    }
};

template <class T, size_t N>
struct ndarray {
    std::optional<buffer<T, N>> _M_buf;

    ndarray() = default;
    ndarray(ndarray const &) = default;
    ndarray &operator=(ndarray const &) = default;
    ndarray(ndarray &&) = default;
    ndarray &operator=(ndarray &&) = default;

    range<N> shape() const {
        return _M_buf ? _M_buf->shape() : range<N>(0);
    }

    size_t size() const {
        return _M_buf ? _M_buf->size() : 0;
    }

    explicit ndarray(range<N> size, T *base = nullptr) {
        reshape(size, base);
    }

    void reshape(range<N> size, T *base = nullptr) {
        for (int i = 0; i < N; i++) {
            if (!size[i]) {
                _M_buf = std::nullopt;
                return;
            }
        }
        _M_buf = buffer<T, N>(base, size);
    }

    template <access::mode mode>
    auto get_access(auto &&cgh) {
        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(cgh)>, host_handler>)
            return _M_buf->template get_access<mode>();
        else
            return _M_buf->template get_access<mode>(cgh);
    }
};

}
ZENO_NAMESPACE_END
