#pragma once

#include <zeno/zycl/zycl.h>
#include <type_traits>
#include <optional>
#include <utility>


ZENO_NAMESPACE_BEGIN
namespace zycl {

struct host_handler {};

template <access::mode mode>
auto make_access(auto &&cgh, auto &&buf) {
    return buf.template get_access<mode>(std::forward<decltype(cgh)>(cgh));
}

template <access::mode mode, class T, size_t N>
auto local_access(auto &&cgh, range<N> size) {
    return accessor<T, N, mode, access::target::local>(size, std::forward<decltype(cgh)>(cgh));
}

template <class F>
struct functor_accessor {
    F f;

    constexpr functor_accessor(F &&f) : f(std::move(f)) {
    }

    constexpr decltype(auto) operator[](auto &&t) const {
        return f(std::forward<decltype(t)>(t));
    }
};

}
ZENO_NAMESPACE_END
