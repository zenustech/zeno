#include <fdb/types.h>
#include <fdb/policy.h>
#include <fdb/VDBGrid.h>
#include <fdb/stencil.h>
#include <fdb/openvdb.h>
#include <cstdio>

using namespace fdb;

int main() {
    VDBGrid grid;

    grid.add({4, 4, 4});

    fdb::foreach(policy::Serial{}, grid, [&] (auto coor, auto &value) {
        value = length(coor - 36) - 4;
        printf("%d %d %d: %f\n", coor[0], coor[1], coor[2], value);
    });

    fdb::write_dense_vdb("/tmp/a.vdb", [&] (Quint3 coor) {
        return grid.get({4, 4, 4})->at(coor);
    }, Quint3(8, 8, 8));

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
