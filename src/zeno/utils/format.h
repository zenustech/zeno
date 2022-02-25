#pragma once

#include <string>
#include <algorithm>
#include <zeno/utils/to_string.h>

namespace zeno {

template <class It>
void __format(std::string &s, It fb, It fe) {
    if (fb == fe) return;
    s += std::string_view(fb, fe - fb);
}

template <class It, class Arg0, class ...Args>
void __format(std::string &s, It fb, It fe, Arg0 &&arg0, Args &&...args) {
    if (fb == fe) return;
    auto it = std::find(fb, fe, '{');
    for (auto p = fb; p != it; ++p) {
        s += *p;
    }
    s += to_string(std::forward<Arg0>(arg0));
    __format(s, it, fe, std::forward<Args>(args)...);
}

template <class ...Args>
std::string format(std::string_view fmt, Args &&...args) {
    std::string s;
    __format(s, fmt.begin(), fmt.end(), std::forward<Args>(args)...);
    return s;

}

}
