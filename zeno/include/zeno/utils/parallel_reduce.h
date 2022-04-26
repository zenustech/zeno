#pragma once

#include <cstdint>

namespace zeno {

template <class ValT, class GetF, class SumF>
inline ValT parallel_reduce_array(size_t num, ValT init, GetF const &get, SumF const &sum) {
    size_t nproc =
#ifdef _OPENMP
        std::max(128, (int)(num / 250000))
#else
        1
#endif
        ;
    std::vector<ValT> tls(nproc, init);
#pragma omp parallel for
    for (intptr_t p = 0; p < nproc; p++) {
        size_t i0 = num / nproc * p;
        size_t i1 = p == nproc - 1 ? num : num / nproc * (p + 1);
        ValT val = init;
        for (size_t i = i0; i < i1; i++) {
            val = sum(val, get(i));
        }
        tls[p] = val;
    }
    ValT ret = init;
    for (size_t p = 0; p < nproc; p++) {
        ret = sum(ret, tls[p]);
    }
    return ret;
}

}
