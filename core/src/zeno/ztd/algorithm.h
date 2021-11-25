#pragma once

#include <algorithm>
#include <zeno/ztd/error.h>

ZENO_NAMESPACE_BEGIN
namespace ztd {

inline size_t try_find_index(auto &&range, auto const &value) {
    return std::find(range.begin(), range.end(), value) - range.begin();
}

inline size_t find_index(auto &&range, auto const &value) {
    auto it = std::find(range.begin(), range.end(), value);
    [[unlikely]] if (it == range.end())
        throw ztd::format_error("ValueError: {} is not in the list", value);
    return it - range.begin();
}

}
ZENO_NAMESPACE_END
