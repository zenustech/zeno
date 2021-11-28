#pragma once


#include <zeno/zbb/parallel_arena.h>


ZENO_NAMESPACE_BEGIN
namespace zbb {


template <class T>
static void parallel_for(blocked_range<T> const &r, auto const &body) {
    parallel_arena(r, [&] (auto const &engine, std::size_t tid) {
        engine(body);
    });
}


template <class T>
static void parallel_for(T const &i0, T const &i1, auto const &body) {
    parallel_for(make_blocked_range(i0, i1), [&] (blocked_range<T> const &r) {
        for (T it = r.begin(); it != r.end(); ++it) {
            body(std::as_const(it));
        }
    });
}


}
ZENO_NAMESPACE_END
