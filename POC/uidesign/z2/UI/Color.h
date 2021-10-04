#pragma once


#include <z2/UI/Color.h>


namespace z2::UI {


struct Color {
    float r, g, b;

    Color(float r = 0, float g = 0, float b = 0)
        : r(r), g(g), b(b) {}

    float *data() {
        return &r;
    }

    float const *data() const {
        return &r;
    }
};


}  // namespace z2::UI
