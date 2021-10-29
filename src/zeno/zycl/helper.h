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
    return buf.template get_access<mode>(std::forward<Cgh>(cgh));
}

template <class F>
struct functor_accessor {
    F f;

    constexpr functor_accessor(F &&f) : f(std::forward<F>(f)) {
    }

    constexpr decltype(auto) operator[](auto &&t) const {
        return f(std::forward<decltype(t)>(t));
    }
};

}
ZENO_NAMESPACE_END
