#pragma once

#include <cstddef>

struct Dim3 {
    size_t x, y, z;

    Dim3(size_t x = 1, size_t y = 1, size_t z = 1)
        : x(x), y(y), z(z)
    {}
};
