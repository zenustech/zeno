#pragma once

#include <zeno/zycl/zycl.h>
#include <type_traits>
#include <utility>


ZENO_NAMESPACE_BEGIN
namespace zycl {

struct host_handler {};

template <access::mode mode = access::mode::read_write, class Cgh>
auto make_access(Cgh &&cgh, auto &&buf) {
    if constexpr (std::is_same_v<std::remove_cvref_t<Cgh>, host_handler>)
        return buf.template get_access<mode>();
    else
        return buf.template get_access<mode>(cgh);
}

template <access::mode mode = access::mode::read_write>
auto make_access(auto &&buf) {
    return make_access(host_handler{}, std::forward<decltype(buf)>(buf));
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

}
ZENO_NAMESPACE_END
