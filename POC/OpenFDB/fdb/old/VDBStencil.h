#pragma once

#include "VDBGrid.h"
#include "policy.h"

namespace fdb {

template <class Grid>
struct Stencil {
    Grid &grid;

    Stencil(Grid &grid) : grid(grid) {}

    template <class Pol, class F>
    void foreach(Pol const &pol, F const &func) {
        return grid.foreach(pol, [&] (Quint3 leafCoor, auto *leaf) {
            func(leafCoor, leaf, [&] (auto const &func) {
                leaf->foreach(pol, [&] (Quint3 subCoor, auto &value) {
                    auto coor = leafCoor << 3 | subCoor;
                    func(coor, value);
                });
            });
        });
    }

    template <class Pol, class F>
    void foreach_2x2x2_cube(Pol const &pol, F const &func) {
        return grid.foreach(pol, [&] (Quint3 leafCoor, auto *leaf) {
            func(leafCoor, leaf, [&] (auto const &func) {
                ndrange_for(policy::Serial{}, Quint3(0), Quint3(7), [&] (Quint3 coor) {
                    func(leafCoor << 3 | coor, leaf->at(coor)
                    , leaf->at(coor + Quint3(1, 0, 0))
                    , leaf->at(coor + Quint3(0, 1, 0))
                    , leaf->at(coor + Quint3(1, 1, 0))
                    , leaf->at(coor + Quint3(0, 0, 1))
                    , leaf->at(coor + Quint3(1, 0, 1))
                    , leaf->at(coor + Quint3(0, 1, 1))
                    , leaf->at(coor + Quint3(1, 1, 1))
                    );
                });
                auto xleaf = grid.get(leafCoor + Quint3(1, 0, 0));
                auto yleaf = grid.get(leafCoor + Quint3(0, 1, 0));
                auto zleaf = grid.get(leafCoor + Quint3(0, 0, 1));
                ndrange_for(policy::Serial{}, Quint2(0), Quint2(7), [&] (Quint2 subCoor) {
                    if (xleaf) {
                        auto coor = Quint3(7, subCoor[0], subCoor[1]);
                        func(leafCoor << 3 | coor, leaf->at(coor)
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
                        auto coor = Quint3(subCoor[0], 7, subCoor[1]);
                        func(leafCoor << 3 | coor, leaf->at(coor)
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
                        auto coor = Quint3(subCoor[0], subCoor[1], 7);
                        func(leafCoor << 3 | coor, leaf->at(coor)
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
                auto xyleaf = grid.get(leafCoor + Quint3(1, 1, 0));
                auto yzleaf = grid.get(leafCoor + Quint3(0, 1, 1));
                auto zxleaf = grid.get(leafCoor + Quint3(1, 0, 1));
                range_for(policy::Serial{}, Quint(0), Quint(7), [&] (Quint subCoor) {
                    if (xleaf && yleaf && xyleaf) {
                        auto coor = Quint3(7, 7, subCoor);
                        func(leafCoor << 3 | coor, leaf->at(coor)
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
                        auto coor = Quint3(subCoor, 7, 7);
                        func(leafCoor << 3 | coor, leaf->at(coor)
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
                        auto coor = Quint3(7, subCoor, 7);
                        func(leafCoor << 3 | coor, leaf->at(coor)
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
                auto xyzleaf = grid.get(leafCoor + Quint3(1, 1, 1));
                if (xleaf && yleaf && zleaf && xyleaf && yzleaf && zxleaf && xyzleaf) {
                    auto coor = Quint3(7, 7, 7);
                    func(leafCoor << 3 | coor, leaf->at(coor)
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
        });
    }

    template <class Pol, class F>
    void foreach_2x2x2_star(Pol const &pol, F const &func) {
        return grid.foreach(pol, [&] (Quint3 leafCoor, auto *leaf) {
            func(leafCoor, leaf, [&] (auto const &func) {
                ndrange_for(policy::Serial{}, Quint3(0), Quint3(7), [&] (Quint3 coor) {
                    func(leafCoor << 3 | coor, leaf->at(coor)
                    , leaf->at(coor + Quint3(1, 0, 0))
                    , leaf->at(coor + Quint3(0, 1, 0))
                    , leaf->at(coor + Quint3(0, 0, 1))
                    );
                });
                auto xleaf = grid.get(leafCoor + Quint3(1, 0, 0));
                auto yleaf = grid.get(leafCoor + Quint3(0, 1, 0));
                auto zleaf = grid.get(leafCoor + Quint3(0, 0, 1));
                ndrange_for(policy::Serial{}, Quint2(0), Quint2(7), [&] (Quint2 subCoor) {
                    if (xleaf) {
                        auto coor = Quint3(7, subCoor[0], subCoor[1]);
                        func(leafCoor << 3 | coor, leaf->at(coor)
                        , xleaf->at(coor + Quint3(-7, 0, 0))
                        , leaf->at(coor + Quint3(0, 1, 0))
                        , leaf->at(coor + Quint3(0, 0, 1))
                        );
                    }
                    if (yleaf) {
                        auto coor = Quint3(subCoor[0], 7, subCoor[1]);
                        func(leafCoor << 3 | coor, leaf->at(coor)
                        , leaf->at(coor + Quint3(1, 0, 0))
                        , yleaf->at(coor + Quint3(0, -7, 0))
                        , leaf->at(coor + Quint3(0, 0, 1))
                        );
                    }
                    if (zleaf) {
                        auto coor = Quint3(subCoor[0], subCoor[1], 7);
                        func(leafCoor << 3 | coor, leaf->at(coor)
                        , leaf->at(coor + Quint3(1, 0, 0))
                        , leaf->at(coor + Quint3(0, 1, 0))
                        , zleaf->at(coor + Quint3(0, 0, -7))
                        );
                    }
                });
                auto xyleaf = grid.get(leafCoor + Quint3(1, 1, 0));
                auto yzleaf = grid.get(leafCoor + Quint3(0, 1, 1));
                auto zxleaf = grid.get(leafCoor + Quint3(1, 0, 1));
                range_for(policy::Serial{}, Quint(0), Quint(7), [&] (Quint subCoor) {
                    if (xleaf && yleaf) {
                        auto coor = Quint3(7, 7, subCoor);
                        func(leafCoor << 3 | coor, leaf->at(coor)
                        , xleaf->at(coor + Quint3(-7, 0, 0))
                        , yleaf->at(coor + Quint3(0, -7, 0))
                        , leaf->at(coor + Quint3(0, 0, 1))
                        );
                    }
                    if (yleaf && zleaf) {
                        auto coor = Quint3(subCoor, 7, 7);
                        func(leafCoor << 3 | coor, leaf->at(coor)
                        , leaf->at(coor + Quint3(1, 0, 0))
                        , yleaf->at(coor + Quint3(0, -7, 0))
                        , zleaf->at(coor + Quint3(0, 0, -7))
                        );
                    }
                    if (zleaf && xleaf) {
                        auto coor = Quint3(7, subCoor, 7);
                        func(leafCoor << 3 | coor, leaf->at(coor)
                        , xleaf->at(coor + Quint3(-7, 0, 0))
                        , leaf->at(coor + Quint3(0, 1, 0))
                        , zleaf->at(coor + Quint3(0, 0, -7))
                        );
                    }
                });
                auto xyzleaf = grid.get(leafCoor + Quint3(1, 1, 1));
                if (xleaf && yleaf && zleaf) {
                    auto coor = Quint3(7, 7, 7);
                    func(leafCoor << 3 | coor, leaf->at(coor)
                    , xleaf->at(coor + Quint3(-7, 0, 0))
                    , yleaf->at(coor + Quint3(0, -7, 0))
                    , zleaf->at(coor + Quint3(0, 0, -7))
                    );
                }
            });
        });
    }
};

}
