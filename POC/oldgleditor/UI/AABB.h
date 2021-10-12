#pragma once


#include <zs/editor/UI/Point.h>


namespace zs::editor::UI {


struct AABB {
    float x0, y0, nx, ny;

    constexpr AABB(float x0 = 0, float y0 = 0, float nx = 0, float ny = 0)
        : x0(x0), y0(y0), nx(nx), ny(ny) {}

    constexpr bool contains(float x, float y) const {
        return x0 <= x && y0 <= y && x <= x0 + nx && y <= y0 + ny;
    }

    constexpr AABB operator-(Point const &p) const {
        return {x0 - p.x, y0 - p.y, nx, ny};
    }

    constexpr AABB operator+(Point const &p) const {
        return {x0 - p.x, y0 - p.y, nx, ny};
    }
};


}  // namespace zs::editor::UI
