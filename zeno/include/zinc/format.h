#pragma once

#include <cstdio>
#include <cstring>
#include <string>

namespace zinc {

template <class ...Ts>
std::string format(const char *fmt, Ts &&...ts) {
#if 0
    int n = sprintf(nullptr, fmt, std::forward<Ts>(ts)...);
    if (n < 0) return {};
#else
    int n = strlen(fmt) * 2 + 4096;
#endif
    std::string res(n);
    sprintf(res.data(), fmt, std::forward<Ts>(ts)...);
    return res;
}

}
