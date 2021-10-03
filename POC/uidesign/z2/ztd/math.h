#pragma once

#include <cmath>

namespace z2::ztd {

static auto pymod(auto x, auto y) {
    auto z = x / y;
    z -= std::floor(z);
    return z * y;
}

}
