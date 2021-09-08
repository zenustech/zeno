#pragma once

#include <cstdio>
#include <cstring>
#include <string>

namespace zinc {

template <class ...Ts>
std::string format(const char *fmt, Ts &&...ts) {
#if 0
    int n = snprintf(nullptr, 0, fmt, std::forward<Ts>(ts)...);
    if (n < 0) return {};
#else
    int n = strlen(fmt) * 2 + 4096;
#endif
    std::string res(n + 2);
    sprintf(res.data(), n + 1, fmt, std::forward<Ts>(ts)...);
    res.resize(n);
    return res;
}

}
