#pragma once

#include <algorithm>
#include <zeno/ztd/error.h>

ZENO_NAMESPACE_BEGIN
namespace ztd {

inline size_t try_find_index(auto &&range, auto const &value) {
    auto beg = std::begin(range);
    auto end = std::end(range);
    return std::find(beg, end, value) - beg;
}

inline size_t find_index(auto &&range, auto const &value) {
    auto beg = std::begin(range);
    auto end = std::end(range);
    auto it = std::find(beg, end, value);
    [[unlikely]] if (it == end)
        throw ztd::format_error("ValueError: {} is not in the list", value);
    return it - beg;
}

}
ZENO_NAMESPACE_END
