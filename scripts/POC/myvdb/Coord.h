#pragma once

#include "vec.h"

using Coord = vec3i;

template <int L>
static Coord combineCoord(Coord const &leafCoord, Coord const &subCoord) {
    return {
        leafCoord[0] << L | subCoord[0],
        leafCoord[1] << L | subCoord[1],
        leafCoord[2] << L | subCoord[2],
    };
}

template <int L>
static Coord staggerCoord(Coord const &coord) {
    int offset = 1 << (L - 1);
    return {
        coord[0] + offset,
        coord[1] + offset,
        coord[2] + offset,
    };
}
