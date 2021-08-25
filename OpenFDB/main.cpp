#include <cstdio>
#include <fdb/types.h>
#include <fdb/policy.h>
#include <fdb/VDBGrid.h>
#include <fdb/stencil.h>
#include <fdb/openvdb.h>

using namespace fdb;

int main() {
    VDBGrid<float> grid, grid2;

    ndrange_for(policy::Serial{}, Quint3(0), Quint3(4), [&] (auto coor) {
        grid.add(coor);
    });

    fdb::foreach(policy::Serial{}, grid, [&] (auto leafCoor, auto *leaf, auto callback) {
        callback([&] (auto coor, auto &value) {
            value = coor[0] > 16.f ? 1.0f : 0.0f;
        });
    });

    fdb::foreach_cell(policy::Serial{}, grid, [&] (auto leafCoor, auto *leaf, auto callback) {
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
            leaf2->at(coor) = fabsf(value100 - value000);
        });
    });

    fdb::write_dense_vdb("/tmp/a.vdb", [&] (Quint3 coor) {
        return grid2.read_at(coor);
    }, Quint3(32, 32, 32));
}
