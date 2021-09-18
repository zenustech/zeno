#pragma once

#include "VDBGrid.h"
#include "Stencil.h"
#include <unordered_set>


template <class T>
struct MarchingCube {
    VDBGrid<T> m_sdf;

    std::unordered_set<Quint4> m_Js;
    std::unordered_map<Quint4, Quint> m_Jtab;
    std::vector<Quint3> m_faces;

    void march() {
        Stencil(m_sdf).foreach_2x2x2_cube(policy::Serial{},
        [&] (auto leafCoor, auto *leaf, auto callback) {
            auto *leaf2 = grid2.add(leafCoor);
            callback([&] (auto coor
                , auto &value000
                , auto &value100
                , auto &value010
                , auto &value110
                , auto &value001
                , auto &value101
                , auto &value011
                , auto &value111
                ) {
                Quint id = 0;
                if (value000 > 0) id |= 1;
                if (value100 > 0) id |= 2;
                if (value010 > 0) id |= 4;
                if (value110 > 0) id |= 8;
                if (value001 > 0) id |= 16;
                if (value101 > 0) id |= 32;
                if (value011 > 0) id |= 64;
                if (value111 > 0) id |= 128;

                for (Qint l = 0; l < 3; l++) {
                    auto e = edgeTable[l];
                    Quint4 J = coor;
                    Quint Jw = 0;
                    if (e == 1 || e == 3 || e == 5 || e == 7) Jw = 1;
                    else if (e == 8 || e == 9 || e == 10 || e == 11) Jw = 2;
                    if (e == 1 || e == 5 || e == 9 || e == 10) J[0]++;
                    if (e == 2 || e == 6 || e == 10 || e == 11) J[1]++;
                    if (e == 4 || e == 5 || e == 6 || e == 7) J[2]++;
                    m_Js.insert(J);
                }
            });
        });
    }
};
