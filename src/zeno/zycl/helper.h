#pragma once

#include <zeno/zycl/core.h>
#include <type_traits>
#include <optional>
#include <utility>


ZENO_NAMESPACE_BEGIN
namespace zycl {

struct host_handler {};


inline constexpr auto ro = access::mode::read;
inline constexpr auto wo = access::mode::write;
inline constexpr auto wd = access::mode::discard_write;
inline constexpr auto rw = access::mode::read_write;
inline constexpr auto rwd = access::mode::discard_read_write;


template <class F>
struct functor_accessor {
    F _M_f;

    constexpr functor_accessor(F &&f) : _M_f(std::move(f)) {
    }

    constexpr decltype(auto) operator[](auto &&t) const {
        return _M_f(std::forward<decltype(t)>(t));
    }
};


template <access::mode mode>
auto make_access(auto &&cgh, auto &&buf) {
    return buf.template get_access<mode>(std::forward<decltype(cgh)>(cgh));
}

#ifdef ZENO_WITH_SYCL
template <access::mode mode, class T, int N>
auto local_access(auto &&cgh, range<N> const &size) {
    return accessor<T, N, mode, access::target::local>(size, std::forward<decltype(cgh)>(cgh));
}
#endif

template <access::mode mode>
auto host_access(auto &&buf) {
    return make_access<mode>(host_handler{}, buf);
}

}
ZENO_NAMESPACE_END
