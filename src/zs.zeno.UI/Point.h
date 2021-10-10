#pragma once


#include <zeno2/UI/stdafx.h>


namespace zeno2::UI {


struct Point {
    float x, y;

    constexpr Point(float x = 0, float y = 0)
        : x(x), y(y) {}

    constexpr Point operator+(Point const &o) const {
        return {x + o.x, y + o.y};
    }

    constexpr Point operator-(Point const &o) const {
        return {x - o.x, y - o.y};
    }

    constexpr Point operator*(float o) const {
        return {x * o, y * o};
    }
};


}  // namespace zeno2::UI
