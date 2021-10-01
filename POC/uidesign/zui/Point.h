#pragma once


#include "stdafx.h"


struct Point {
    float x, y;

    Point(float x = 0, float y = 0)
        : x(x), y(y) {}

    Point operator+(Point const &o) const {
        return {x + o.x, y + o.y};
    }

    Point operator-(Point const &o) const {
        return {x - o.x, y - o.y};
    }

    Point operator*(float o) const {
        return {x * o, y * o};
    }
};
