#pragma once

#include <cstdio>
#include <string>
#include <sstream>

namespace ztd {

template <class ...Ts>
std::string format(const char *fmt, Ts &&...ts) {
    int n = snprintf(nullptr, 0, fmt, std::forward<Ts>(ts)...);
    if (n < 0) return {};
    std::string res;
    res.resize(n + 2);
    n = snprintf(res.data(), n + 1, fmt, std::forward<Ts>(ts)...);
    res.resize(n);
    return res;
}

template <class ...Ts>
std::string toString(Ts const &...ts) {
    std::ostringstream ss;
    ss << ... << ts;
    return ss.str();
}

}
