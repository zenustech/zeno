#pragma once

#include <zs/zeno/ztd/error.h>

namespace zeno2::ztd {

#define __ZENO2_ZTD_ASSERT_PRED(y, ...) y
#define __ZENO2_ZTD_ASSERT_EXT(xs, y, ...) __VA_OPT__(": {}"), xs __VA_ARGS__
#define ZENO2_ZTD_ASSERT(x, ...) (([&] (auto &&__assert_val) -> decltype(auto) { \
    [[unlikely]] if (!((__assert_val) __ZENO2_ZTD_ASSERT_PRED(__VA_ARGS__))) \
        throw ztd::format_error("AssertionError: {}" __ZENO2_ZTD_ASSERT_EXT(#x, __VA_ARGS__)); \
    return std::forward<decltype(__assert_val)>(__assert_val); \
})((x)))

}
