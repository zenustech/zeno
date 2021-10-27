#pragma once

#include <zeno/zycl/zycl.h>
#include <type_traits>
#include <utility>


ZENO_NAMESPACE_BEGIN
namespace zycl {

struct host_handler {};

template <access::mode mode = access::mode::read_write, class Cgh>
auto make_accessor(Cgh &&cgh, auto &&buf) {
    if constexpr (std::is_same_v<std::remove_cvref_t<Cgh>, host_handler>)
        return buf.template get_accessor<mode>();
    else
        return buf.template get_accessor<mode>(cgh);
}

template <class F>
struct functor_accessor {
    F f;

    constexpr functor_accessor(F &&f) : f(std::forward<F>(f)) {
    }

    constexpr decltype(auto) operator[](auto &&ts) {
        return f(std::forward<decltype(t)>(t));
    }
};

template <class T>
struct vector {
    buffer<T, 1> _M_buf;

    template <access::mode mode>
    functor_accessor get_accessor(auto &&cgh) {
        auto a_buf = make_accessor(cgh, _M_buf);
        return [=] (id<1> idx) {
            return a_buf[idx];
        };
    }
};

}
ZENO_NAMESPACE_END
