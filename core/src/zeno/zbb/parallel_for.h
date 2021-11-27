#pragma once


#include <zeno/common.h>
#include <utility>


ZENO_NAMESPACE_BEGIN
namespace zbb {


template <class T>
static void parallel_for(T i0, T i1, auto &&body) {
    #pragma omp parallel for
    for (T i = i0; i < i1; i++) {
        body(std::as_const(i));
    }
}


}
ZENO_NAMESPACE_END
