#include <iostream>
#include <zeno/utils/vec.h>

namespace zeno {

template <size_t N, class T>
std::ostream &operator<<(std::ostream& o, vec<N,T> v) {
    for (int i = 0; i < N; i++) {
        if (i != 0) {
            o << ", ";
        }
        o << v[i];
    }
    return o;
}
}