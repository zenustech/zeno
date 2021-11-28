#pragma once


#include <zeno/zbb/parallel_arena.h>
#include <vector>


ZENO_NAMESPACE_BEGIN
namespace zbb {


template <class T, class Ret>
static Ret parallel_reduce(blocked_range<T> const &r, Ret const &ident, auto const &binop, auto const &body) {
    std::vector<Ret> partial(r.num_procs());
    parallel_arena(r, [&] (auto const &engine, std::size_t tid) {
        Ret tls = ident;
        engine([&] (blocked_range<T> const &r) {
            body(tls, r);
        });
        partial[tid] = tls;
    });
    Ret res = ident;
    for (Ret const &x: partial) {
        res = binop(res, x);
    }
    return res;
}


template <class T, class Ret>
static Ret parallel_reduce(T const &i0, T const &i1, Ret const &ident, auto const &binop, auto const &body) {
    return parallel_reduce(make_blocked_range(i0, i1), ident, binop, body);
}


}
ZENO_NAMESPACE_END
