#pragma once

struct Coord {
    int x, y, z;
};

template <int L>
static Coord combineCoord(Coord const &leafCoord, Coord const &subCoord) {
    return {
        leafCoord.x << L | subCoord.x,
        leafCoord.y << L | subCoord.y,
        leafCoord.z << L | subCoord.z,
    };
}

template <int L>
static Coord staggerCoord(Coord const &coord) {
    int offset = 1 << (L - 1);
    return {
        coord.x + offset,
        coord.y + offset,
        coord.z + offset,
    };
}
