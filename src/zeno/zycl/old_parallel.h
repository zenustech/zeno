#pragma once

#include <zeno/zycl/core.h>
#include <type_traits>
#include <optional>
#include <utility>

// TODO: this header has been deprecated


ZENO_NAMESPACE_BEGIN
namespace zycl {

template <class T, class Alt>
using _M_void_or = std::conditional_t<std::is_void_v<T>, std::remove_cvref_t<Alt>, T>;

template <int N>
auto parallel_for
    ( handler &cgh
    , range<N> const &dim
    , auto &&body
    ) {
    return cgh.parallel_for(dim, body);
}

#ifdef ZENO_WITH_SYCL

template <int N>
auto parallel_for
    ( handler &cgh
    , range<N> const &dim
    , range<N> const &blkdim
    , auto &&body
    , auto &&...args
    ) {
    return cgh.parallel_for
    ( _M_calc_nd_range(dim, blkdim)
    , std::forward<decltype(args)>(args)...
    , [dim, body = std::move(body)] (nd_item<N> const &it, auto &...args) {
        for (int i = 0; i < N; i++) {
            [[unlikely]] if (it.get_global_id(i) >= dim[i])
                return;
        }
        body(it, args...);
    });
}

#else

template <int N>
auto parallel_for
    ( handler &cgh
    , range<N> const &dim
    , range<N> const &blkdim
    , auto &&body
    , auto &&...args
    ) {
    return parallel_for(
        cgh, dim, std::move(body),
        std::forward<decltype(args)>(args)...);
}

#endif

#ifdef ZENO_WITH_SYCL

template <int N>
auto parallel_reduce
    ( handler &cgh
    , range<N> const &dim
    , range<N> const &blkdim
    , auto &&buf
    , auto ident
    , auto &&binop
    , auto &&body
    ) {
    return parallel_for(
                 cgh, dim, blkdim, std::move(body),
                 reduction(buf, ident, binop));
}

#else

template <class T, class BinOp>
struct _M_parallel_reducer {
    T _M_had;
    BinOp _M_binop;

    _M_parallel_reducer(T const &ident, BinOp &&binop)
        : _M_had(ident), _M_binop(std::move(binop)) {}

    constexpr void combine(T const &t) const {
        _M_binop(_M_had, t);
    }
};

template <int N>
auto parallel_reduce
    ( handler &cgh
    , range<N> const &dim
    , range<N> const &blkdim
    , auto &&buf
    , auto ident
    , auto &&binop
    , auto &&body
    ) {
    auto current = ident;
    auto e = parallel_for
    ( cgh
    , dim
    , [&current, binop = std::move(binop), body = std::move(body)] (item<1> const &it) {
        _M_parallel_reducer reducer(current, std::move(binop));
        body(it, reducer);
    });
    buf[0] = current;
    return e;
}

#endif


}
ZENO_NAMESPACE_END
