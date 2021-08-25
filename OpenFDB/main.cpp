#include <cstdio>
#include <fdb/types.h>
#include <fdb/policy.h>
#include <fdb/VDBGrid.h>
#include <fdb/Stencil.h>
#include <fdb/openvdb.h>
#include <fdb/MarchingCube.h>

using namespace fdb;

int main() {
    VDBGrid<float> grid;

    ndrange_for(policy::Serial{}, Quint3(0), Quint3(8), [&] (auto coor) {
        grid.add(coor);
    });

    Stencil(grid).foreach(policy::Parallel{}, [&] (auto leafCoor, auto *leaf, auto callback) {
        callback([&] (auto coor, auto &value) {
            value = length(coor - 32.f) < 32.f ? 1.0f : 0.0f;
        });
    });

    Stencil(m_sdf).foreach_2x2x2_star(policy::Serial{},
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

    write_dense_vdb("/tmp/a.vdb", [&] (Quint3 coor) {
        return grid.read_at(coor);
    }, Quint3(64, 64, 64));
}
