#pragma once

#include <cstdio>
#include <cstring>
#include <string>

namespace zinc {

template <class ...Ts>
std::string format(const char *fmt, Ts &&...ts) {
    std::string res;
    int n = sprintf(nullptr, fmt, std::forward<Ts>(ts)...);
    if (n < 0) return {};
    res.resize(n);
    sprintf(res.data(), fmt, std::forward<Ts>(ts)...);
    return res;
}

}
