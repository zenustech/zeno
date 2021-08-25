#pragma once

#include "VDBGrid.h"
#include "policy.h"

namespace fdb {

template <class Pol, class Grid, class F>
void foreach(Pol const &pol, Grid &grid, F const &func) {
    return grid.foreach(pol, [&] (Quint3 coor1, auto &leaf) {
        leaf->foreach(pol, [&] (Quint3 coor2, auto &value) {
            auto coor = coor1 << 3 | coor2;
            func(coor, value);
        });
    });
}

template <class Pol, class Grid, class F>
void foreach_cell(Pol const &pol, Grid &grid, F const &func) {
    return grid.foreach(pol, [&] (Quint3 coor1, auto &leaf) {
        ndrange_for(policy::Serial{}, Quint3(0), Quint3(7), [&] (Quint3 coor, auto &value) {
            func(coor1 << 3 | coor, value
            , leaf->at(coor + Quint3(1, 0, 0))
            , leaf->at(coor + Quint3(0, 1, 0))
            , leaf->at(coor + Quint3(1, 1, 0))
            , leaf->at(coor + Quint3(0, 0, 1))
            , leaf->at(coor + Quint3(1, 0, 1))
            , leaf->at(coor + Quint3(0, 1, 1))
            , leaf->at(coor + Quint3(1, 1, 1))
            );
        });
        auto xleaf = grid.get(coor1 + Quint(1, 0, 0));
        auto yleaf = grid.get(coor1 + Quint(0, 1, 0));
        auto zleaf = grid.get(coor1 + Quint(0, 0, 1));
        ndrange_for(policy::Serial{}, Quint2(0), Quint2(7), [&] (Quint2 coor2) {
            if (xleaf) {
                auto coor = Quint3(7, coor2[0], coor2[1]);
                func(coor1 << 3 | coor, leaf(coor)
                , xleaf->at(coor + Quint3(-7, 0, 0))
                , leaf->at(coor + Quint3(0, 1, 0))
                , xleaf->at(coor + Quint3(-7, 1, 0))
                , leaf->at(coor + Quint3(0, 0, 1))
                , xleaf->at(coor + Quint3(-7, 0, 1))
                , leaf->at(coor + Quint3(0, 1, 1))
                , xleaf->at(coor + Quint3(-7, 1, 1))
                );
            }
            if (yleaf) {
                auto coor = Quint3(coor2[0], 7, coor2[1]);
                func(coor1 << 3 | coor, leaf(coor)
                , leaf->at(coor + Quint3(1, 0, 0))
                , yleaf->at(coor + Quint3(0, -7, 0))
                , yleaf->at(coor + Quint3(1, -7, 0))
                , leaf->at(coor + Quint3(0, 0, 1))
                , leaf->at(coor + Quint3(1, 0, 1))
                , yleaf->at(coor + Quint3(0, -7, 1))
                , yleaf->at(coor + Quint3(1, -7, 1))
                );
            }
            if (zleaf) {
                auto coor = Quint3(coor2[0], coor2[1], 7);
                func(coor1 << 3 | coor, leaf->at(coor)
                , leaf->at(coor + Quint3(1, 0, 0))
                , leaf->at(coor + Quint3(0, 1, 0))
                , leaf->at(coor + Quint3(1, 1, 0))
                , zleaf->at(coor + Quint3(0, 0, -7))
                , zleaf->at(coor + Quint3(1, 0, -7))
                , zleaf->at(coor + Quint3(0, 1, -7))
                , zleaf->at(coor + Quint3(1, 1, -7))
                );
            }
        });
        auto xyleaf = grid.get(coor1 + Quint(1, 1, 0));
        auto yzleaf = grid.get(coor1 + Quint(0, 1, 1));
        auto zxleaf = grid.get(coor1 + Quint(1, 0, 1));
        range_for(policy::Serial{}, Quint(0), Quint(7), [&] (Quint coor2) {
            if (xleaf && yleaf && xyleaf) {
                auto coor = Quint3(7, 7, coor2);
                func(coor1 << 3 | coor, leaf->at(coor)
                , xleaf->at(coor + Quint3(-7, 0, 0))
                , yleaf->at(coor + Quint3(0, -7, 0))
                , xyleaf->at(coor + Quint3(-7, -7, 0))
                , leaf->at(coor + Quint3(0, 0, 1))
                , xleaf->at(coor + Quint3(-7, 0, 1))
                , yleaf->at(coor + Quint3(0, -7, 1))
                , xyleaf->at(coor + Quint3(-7, -7, 1))
                );
            }
            if (yleaf && zleaf && yzleaf) {
                auto coor = Quint3(coor2, 7, 7);
                func(coor1 << 3 | coor, leaf->at(coor)
                , leaf->at(coor + Quint3(1, 0, 0))
                , yleaf->at(coor + Quint3(0, -7, 0))
                , yleaf->at(coor + Quint3(1, -7, 0))
                , zleaf->at(coor + Quint3(0, 0, -7))
                , zleaf->at(coor + Quint3(1, 0, -7))
                , yzleaf->at(coor + Quint3(0, -7, -7))
                , yzleaf->at(coor + Quint3(1, -7, -7))
                );
            }
            if (zleaf && xleaf && zxleaf) {
                auto coor = Quint3(7, coor2, 7);
                func(coor1 << 3 | coor, leaf->at(coor)
                , xleaf->at(coor + Quint3(-7, 0, 0))
                , leaf->at(coor + Quint3(0, 1, 0))
                , xleaf->at(coor + Quint3(-7, 1, 0))
                , zleaf->at(coor + Quint3(0, 0, -7))
                , zxleaf->at(coor + Quint3(-7, 0, -7))
                , zleaf->at(coor + Quint3(0, 1, -7))
                , zxleaf->at(coor + Quint3(-7, 1, -7))
                );
            }
        });
        auto xyzleaf = grid.get(coor1 + Quint(1, 1, 1));
        if (xyzleaf) {
            auto coor = Quint3(7, 7, 7);
            func(coor1 << 3 | coor, leaf->at(coor)
            , xleaf->at(coor + Quint3(-7, 0, 0))
            , yleaf->at(coor + Quint3(0, -7, 0))
            , xyleaf->at(coor + Quint3(-7, -7, 0))
            , zleaf->at(coor + Quint3(0, 0, -7))
            , zxleaf->at(coor + Quint3(-7, 0, -7))
            , yzleaf->at(coor + Quint3(0, -7, -7))
            , xyzleaf->at(coor + Quint3(-7, -7, -7))
            );
        }
    });
}

}
