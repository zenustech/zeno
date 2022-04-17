#pragma once

#include <string>
#include <sstream>
#include <algorithm>
#include <zeno/utils/to_string.h>

namespace zeno {

template <int curr, class Os, class It, class ...Args>
bool __format(Os &os, It fb, It fe, std::tuple<Args const &...> const &args) {
    if constexpr (curr >= sizeof...(Args)) {
        if (fb != fe)
            os << std::string_view(fb, fe - fb);
        return curr > sizeof...(Args);
    } else {
        if (fb == fe) return false;
        auto ib = std::find(fb, fe, '{');
        if (ib == fe) return false;
        os << std::string_view(fb, ib - fb);
        ++ib;
        auto ie = std::find(ib, fe, '}');
        if (ie == fe) return false;
        ++ie;
        auto fms = std::string_view(ib, ie - ib);
        if (auto i = fms.find(':'); i != std::string_view::npos) {
            fms = fms.substr(i + 1);
        }
        to_stream(os, std::get<curr>(args), fms);
        return __format<curr + 1>(os, ie, fe, args);
    }
}

template <class Os, class ...Args>
bool format_to(Os &os, std::string_view fmt, Args const &...args) {
    return __format<0>(os, fmt.begin(), fmt.end(), std::tuple<Args const &...>(args...));
}

template <class ...Args>
std::string format(std::string_view fmt, Args const &...args) {
    std::stringstream ss;
    format_to(ss, fmt, args...);
    return ss.str();
}

}
