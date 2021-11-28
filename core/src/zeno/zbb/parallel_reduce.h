#pragma once


#include <zeno/zbb/parallel_for.h>


ZENO_NAMESPACE_BEGIN
namespace zbb {


template <class T, class Ret>
static Ret parallel_reduce(T i0, T i1, Ret ident, auto const &binop, auto const &body) {
}


}
ZENO_NAMESPACE_END
