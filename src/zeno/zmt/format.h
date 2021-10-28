#pragma once


#include <zeno/common.h>
#include <zeno/ztd/meta_tools.h>
#include <string_view>
#include <sstream>
#include <string>
#include <tuple>


ZENO_NAMESPACE_BEGIN
namespace zmt {

template <class ...Args>
using format_string = std::string_view;

template <class ...Args>
inline void _format_impl(auto &os, std::string_view fmt, size_t idx, std::tuple<Args...> const &args) {
    if constexpr (sizeof...(Args) != 0) {
        size_t beg = fmt.find('{', idx);
        size_t end = fmt.find('}', beg);
        os << fmt.substr(idx, beg);
        os << std::get<0>(args);
        return _format_impl(os, fmt, end + 1, ztd::tuple_pop_front(args));
    } else {
        os << fmt.substr(idx);
    }
}

template <class ...Args>
inline void vformat_to(auto &os, std::string_view fmt, Args &&...args) {
    return _format_impl<Args...>(os, fmt, 0, std::forward_as_tuple(std::forward<Args>(args)...));
}

template <class ...Args>
inline void format_to(auto &os, format_string<Args...> fmt, Args &&...args) {
    return vformat_to<Args...>(os, fmt, std::forward<Args>(args)...);
}

template <class ...Args>
inline std::string vformat(std::string_view fmt, Args &&...args) {
    std::ostringstream ss;
    vformat_to<Args...>(ss, fmt, std::forward<Args>(args)...);
    return ss.str();
}

template <class ...Args>
inline std::string format(format_string<Args...> fmt, Args &&...args) {
    return vformat<Args...>(fmt, std::forward<Args>(args)...);
}

}
ZENO_NAMESPACE_END
