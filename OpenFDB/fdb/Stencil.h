#pragma once

#include "VDBGrid.h"

namespace fdb {

template <class Pol, class Grid, class F>
void foreachVert(Pol const &pol, Grid &grid, F const &func) const {
    return grid->foreach(pol, [&] (Quint3 coor1, auto &leaf) {
        leaf->foreach([&] (Quint3 coor2, auto &value) {
            auto coor = coor1 << 3 | coor2;
            func(coor, value);
        });
    });
}

template <class Pol, class Grid, class F>
void foreachCell(Pol const &pol, Grid &grid, F const &func) const {
    return grid->foreach(pol, [&] (Quint3 coor1, auto &leaf) {
        leaf->foreach([&] (Quint3 coor2, auto &value) {
            auto coor = coor1 << 3 | coor2;
            func(coor, value);
        });
    });
}

}
