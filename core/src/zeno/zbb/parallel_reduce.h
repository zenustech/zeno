#pragma once


#include <zeno/zbb/parallel_for.h>


ZENO_NAMESPACE_BEGIN
namespace zbb {


template <class T, class Ret>
static Ret parallel_reduce(blocked_range<T> const &r, Ret const &ident, auto const &binop, auto const &body) {
}


template <class T, class Ret>
static Ret parallel_reduce(T const &i0, T const &i1, Ret const &ident, auto const &binop, auto const &body) {
    return parallel_reduce(blocked_range<T>(i0, i1), ident, binop, body);
}


}
ZENO_NAMESPACE_END
