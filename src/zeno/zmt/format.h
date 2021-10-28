#pragma once


#include <zeno/common.h>
#include <string_view>
#include <tuple>


ZENO_NAMESPACE_BEGIN
namespace zmt {

template <class ...Args>
using format_string = std::string_view;

template <class ...Args>
inline void format_impl(auto begin, auto end, std::tuple<Args...> &&args) {
    for (auto p = begin; p != end; p++) {
        if (*p == '{') {
            if (auto q = std::find(p, end, '}'); q != end) {
                return format_impl(q, end, args);
            }
        }
    }
}

template <class ...Args>
inline void format(format_string<Args...> fmt, Args &&...args) {
    format_impl<Args...>(fmt.cbegin(), fmt.cend(), std::forward_as_tuple<Args...>(args...));
}

}
ZENO_NAMESPACE_END
