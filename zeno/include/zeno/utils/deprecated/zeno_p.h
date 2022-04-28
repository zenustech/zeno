#pragma once

#include <iostream>

namespace zeno {

template <class T0 = float, class T>
static void _zeno_p(const char *msg, T const &t) {
    std::cout << msg << ':';
    T0 const *begin = reinterpret_cast<T0 const *>(&t);
    T0 const *end = reinterpret_cast<T0 const *>(&t + 1);
    for (auto p = begin; p < end; p++) {
        std::cout << ' ' << std::to_string(*p);
    }
    std::cout << std::endl;
}

}

#define ZENO_P(x) ::zeno::_zeno_p(#x, x)
