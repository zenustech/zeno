#include <cstdio>
#include <fdb/types.h>
#include <fdb/policy.h>
#include <fdb/VDBGrid.h>
#include <fdb/stencil.h>
#include <fdb/openvdb.h>

using namespace fdb;

int main() {
    VDBGrid<float> grid, grid2;

    int recount = 0;
    ndrange_for(policy::Serial{}, Quint3(0), Quint3(8), [&] (auto coor) {
        recount++;
        grid.add(coor);
    });

    int count = 0;
    fdb::foreach(policy::Serial{}, grid, [&] (auto leafCoor, auto *leaf, auto callback) {
        count++;
        callback([&] (auto coor, auto &value) {
            value = 1.0f;
        });
    });
    printf("%d %d\n", count, recount);

    /*fdb::foreach_cell(policy::Serial{}, grid, [&] (auto leafCoor, auto *leaf, auto callback) {
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
            leaf2->at(coor) = value000;//fabsf(value100 - value000);
        });
    });*/

    fdb::write_dense_vdb("/tmp/a.vdb", [&] (Quint3 coor) {
        return grid.read_at(coor);
    }, Quint3(64, 64, 64));
}
