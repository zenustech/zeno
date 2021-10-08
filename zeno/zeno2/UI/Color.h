#pragma once


#include <zeno2/UI/Color.h>


namespace zeno2::UI {


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


}  // namespace zeno2::UI
