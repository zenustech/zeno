#pragma once

#include <zeno/common.h>
#include <cmath>

ZENO_NAMESPACE_BEGIN
namespace math {
inline namespace _math_h {

static auto divup(auto x, auto y) {
    return (x + y - 1) / y;
}

static auto pymod(auto x, auto y) {
    auto z = x / y;
    z -= std::floor(z);
    return z * y;
}

static auto &augmin(auto &x, auto y) {
    x = std::min(x, y);
    return x;
}

static auto &augmax(auto &x, auto y) {
    x = std::max(x, y);
    return x;
}

static auto &augpymod(auto &x, auto y) {
    x = pymod(x, y);
    return x;
}

static auto &augdivup(auto &x, auto y) {
    x = divup(x, y);
    return x;
}

}
}
ZENO_NAMESPACE_END
