#pragma once

#include <zeno/zycl/zycl.h>
#include <type_traits>
#include <optional>
#include <utility>


ZENO_NAMESPACE_BEGIN
namespace zycl {

struct host_handler {};


template <class F>
struct functor_accessor {
    F _M_f;

    constexpr functor_accessor(F &&f) : _M_f(std::move(f)) {
    }

    constexpr decltype(auto) operator[](auto &&t) const {
        return _M_f(std::forward<decltype(t)>(t));
    }
};


template <class Base, class Range>
struct span {
    Base _M_base;
    Range _M_size;

    constexpr span(Base &&base, Range const &size)
        : _M_base(std::move(base)), _M_size(std::move(size)) {
    }

    constexpr Range size() const {
        return _M_size;
    }

    constexpr decltype(auto) operator[](auto &&t) const {
        return _M_base[std::forward<decltype(t)>(t)];
    }
};


template <class Base, class Range>
span(Base &&base, Range &&size) -> span<std::remove_cvref_t<Base>, std::remove_cvref_t<Range>>;


template <access::mode mode>
auto make_access(auto &&cgh, auto &&buf) {
    return span{buf.template get_access<mode>(std::forward<decltype(cgh)>(cgh)), buf.size()};
}

#ifndef ZENO_SYCL_IS_EMULATED
template <access::mode mode, class T, int N>
auto local_access(auto &&cgh, range<N> const &size) {
    return span{accessor<T, N, mode, access::target::local>(size, std::forward<decltype(cgh)>(cgh)), size};
}
#endif

template <access::mode mode>
auto host_access(auto &&buf) {
    return make_access<mode>(host_handler{}, buf);
}

#ifndef ZENO_SYCL_IS_EMULATED
auto make_reduction(auto &&buf, auto ident, auto &&binop) {
    return reduction(buf, ident, binop, {});
}
#endif

}
ZENO_NAMESPACE_END
