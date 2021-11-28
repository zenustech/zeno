#pragma once


#include <zeno/zbb/parallel_for.h>


ZENO_NAMESPACE_BEGIN
namespace zbb {


template <class T>
static void parallel_for_each(blocked_range<T> const &r, auto const &body) {
    parallel_for(r, [&] (blocked_range<T> const &r) {
        for (T it = r.begin(); it != r.end(); ++it) {
            body(*it);
        }
    });
}


template <class T>
static void parallel_for_each(T const &i0, T const &i1, auto const &body) {
    parallel_for_each(make_blocked_range(i0, i1), body);
}


}
ZENO_NAMESPACE_END
