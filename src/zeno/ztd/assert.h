#pragma once

#include <zeno/ztd/error.h>

ZENO_NAMESPACE_BEGIN
namespace ztd {

#define __ZENO_ZTD_ASSERT_TST(y, ...) y
#define __ZENO_ZTD_ASSERT_PRE(y, ...) __VA_OPT__(": {}")
#define __ZENO_ZTD_ASSERT_MSG(y, ...) __VA_OPT__(, __VA_ARGS__)
#define __ZENO_ZTD_ASSERT_EXT(xs, ...) __VA_OPT__(__ZENO_ZTD_ASSERT_PRE(__VA_ARGS__)), xs __VA_OPT__(__ZENO_ZTD_ASSERT_MSG(__VA_ARGS__))
#define ZENO_ZTD_ASSERT(x, ...) (([&] (auto &&__assert_val) -> decltype(auto) { \
    [[unlikely]] if (!((__assert_val) __ZENO_ZTD_ASSERT_TST(__VA_ARGS__))) \
        throw ZENO_NAMESPACE::ztd::format_error("AssertionError: {}" __ZENO_ZTD_ASSERT_EXT(#x, __VA_ARGS__)); \
    return std::forward<decltype(__assert_val)>(__assert_val); \
})((x)))

#define ZENO_ZTD_ASSERT_EQ(x, y) ZENO_ZTD_ASSERT((x), == (y), fmt::format("unsatisfied: {} == {}", (x), (y)))
#define ZENO_ZTD_ASSERT_NE(x, y) ZENO_ZTD_ASSERT((x), != (y), fmt::format("unsatisfied: {} != {}", (x), (y)))
#define ZENO_ZTD_ASSERT_GE(x, y) ZENO_ZTD_ASSERT((x), >= (y), fmt::format("unsatisfied: {} >= {}", (x), (y)))
#define ZENO_ZTD_ASSERT_LE(x, y) ZENO_ZTD_ASSERT((x), <= (y), fmt::format("unsatisfied: {} <= {}", (x), (y)))
#define ZENO_ZTD_ASSERT_GT(x, y) ZENO_ZTD_ASSERT((x), > (y), fmt::format("unsatisfied: {} > {}", (x), (y)))
#define ZENO_ZTD_ASSERT_LT(x, y) ZENO_ZTD_ASSERT((x), < (y), fmt::format("unsatisfied: {} < {}", (x), (y)))
#define ZENO_ZTD_ASSERT_FALSE(x) ZENO_ZTD_ASSERT((x), == false, fmt::format("unsatisfied: {} == false", (x)))
#define ZENO_ZTD_ASSERT_TRUE(x) ZENO_ZTD_ASSERT((x), == true, fmt::format("unsatisfied: {} == true", (x)))
#define ZENO_ZTD_ASSERT_NULL(x) ZENO_ZTD_ASSERT((x), == nullptr, fmt::format("unsatisfied: {} == nullptr", (void *)(x)))
#define ZENO_ZTD_ASSERT_NOTNULL(x) ZENO_ZTD_ASSERT((x), != nullptr, fmt::format("unsatisfied: {} != nullptr", (void *)(x)))

}
ZENO_NAMESPACE_END
