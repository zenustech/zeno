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

    fdb::Stencil(grid).foreach(policy::Parallel{}, [&] (auto leafCoor, auto *leaf, auto callback) {
        callback([&] (auto coor, auto &value) {
            value = length(coor - 32.f) < 32.f ? 1.0f : 0.0f;
        });
    });

    fdb::write_dense_vdb("/tmp/a.vdb", [&] (Quint3 coor) {
        return grid.read_at(coor);
    }, Quint3(64, 64, 64));
}
