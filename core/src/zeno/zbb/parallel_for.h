#pragma once


#include <zeno/common.h>


ZENO_NAMESPACE_BEGIN
namespace zbb {


template <class T, class Body>
static void parallel_for(T i0, T i1, Body const &body) {
    #pragma omp parallel for
    for (T i = i0; i != i1; i++) {
        body(static_cast<T const &>(i));
    }
}


}
ZENO_NAMESPACE_END
