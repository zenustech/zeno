#pragma once

#include <utility>

namespace fdb {

namespace policy {

    struct Serial {
        template <class F, class T>
        void range_for(T start, T stop, F const &func) const {
            for (T i = start; i < stop; i++) {
                func(i);
            }
        }
    };

    struct Parallel {
        template <class F, class T>
        void range_for(T start, T stop, F const &func) const {
            #pragma omp parallel for
            for (T i = start; i < stop; i++) {
                func(i);
            }
        }
    };
}

}
