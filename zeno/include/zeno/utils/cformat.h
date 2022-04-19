#pragma once

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <stdarg.h>
#include <string>

namespace zeno {

template <class ...Ts>
std::string cformat(const char *fmt, Ts &&...ts) {
    int n = snprintf(nullptr, 0, fmt, std::forward<Ts>(ts)...);
    if (n < 0) return {};
    std::string res;
    res.resize(n + 2);
    n = snprintf(res.data(), n + 1, fmt, std::forward<Ts>(ts)...);
    res.resize(n);
    return res;
}

}
