#pragma once

#include <zs/ztd/error.h>

namespace zs::ztd {

#define __ZS_ZTD_ASSERT_PRED(y, ...) y
#define __ZS_ZTD_ASSERT_EXT(xs, y, ...) __VA_OPT__(": {}"), xs __VA_ARGS__
#define ZS_ZTD_ASSERT(x, ...) (([&] (auto &&__assert_val) -> decltype(auto) { \
    [[unlikely]] if (!((__assert_val) __ZS_ZTD_ASSERT_PRED(__VA_ARGS__))) \
        throw ::zs::ztd::format_error("AssertionError: {}" __ZS_ZTD_ASSERT_EXT(#x, __VA_ARGS__)); \
    return std::forward<decltype(__assert_val)>(__assert_val); \
})((x)))

}
