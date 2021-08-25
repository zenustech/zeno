#include <cstdio>
#include <fdb/types.h>
#include <fdb/policy.h>
#include <fdb/VDBGrid.h>
#include <fdb/stencil.h>
#include <fdb/openvdb.h>

using namespace fdb;

int main() {
    VDBGrid<float> grid;

    grid.add({0, 0, 0});
    grid.add({1, 0, 0});
    grid.add({0, 1, 0});
    grid.add({1, 1, 0});
    grid.add({0, 0, 1});
    grid.add({1, 0, 1});
    grid.add({0, 1, 1});
    grid.add({1, 1, 1});

    fdb::foreach(policy::Serial{}, grid, [&] (auto coor, auto &value) {
        value = length(coor - 8.f) < 8.f ? 1.f : 0.f;
        printf("%d %d %d: %f\n", coor[0], coor[1], coor[2], value);
    });

    fdb::write_dense_vdb("/tmp/a.vdb", [&] (Quint3 coor) {
        return grid.sample(coor);
        //return length(coor - 8.f) < 8.f ? 1.f : 0.f;
    }, Quint3(16, 16, 16));

    /*fdb::foreach_cell(policy::Serial{}, grid, [&] (auto coor
        , auto &value000
        , auto &value100
        , auto &value010
        , auto &value110
        , auto &value001
        , auto &value101
        , auto &value011
        , auto &value111
        ) {
    });*/
}
