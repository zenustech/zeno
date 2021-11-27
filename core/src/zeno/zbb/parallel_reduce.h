#pragma once


#include <zeno/common.h>
#include <utility>


ZENO_NAMESPACE_BEGIN
namespace zbb {


template <class T, class Ret, class BinOp, class Body>
static Ret parallel_reduce(T i0, T i1, Ret ident, BinOp const &binop, Body const &body) {
    struct Wrapped {
        T val;
        BinOp const &binop;
    } res = {ident, binop};
    #pragma omp declare reduction(zbb_reduce: Wrapped: omp_out.val = omp_out.binop(std::move(omp_out.val), std::as_const(omp_in.val))) initializer(omp_priv = omp_orig)
    #pragma omp parallel for reduction(zbb_reduce: res)
    for (T i = i0; i != i1; i++) {
        body(res.val, std::as_const(i));
    }
    return res.val;
}


}
ZENO_NAMESPACE_END
