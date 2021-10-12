#pragma once


#include <zs/editor/UI/Color.h>


namespace zs::editor::UI {


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


}  // namespace zs::editor::UI
