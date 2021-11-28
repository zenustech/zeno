#pragma once


#include <zeno/zbb/parallel_for.h>


ZENO_NAMESPACE_BEGIN
namespace zbb {


template <class T>
static void parallel_for_each(T i0, T i1, auto const &body) {
    parallel_for(i0, i1, [&] (T const &it) {
        body(*it);
    });
}


}
ZENO_NAMESPACE_END
