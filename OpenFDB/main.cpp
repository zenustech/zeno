#include <cstdio>
#include <fdb/types.h>
#include <fdb/policy.h>
#include <fdb/VDBGrid.h>
#include <fdb/Stencil.h>
#include <fdb/openvdb.h>

using namespace fdb;

int main() {
    VDBGrid<Qfloat> g_pressure;
    VDBGrid<Qfloat3> g_velocity;

    ndrange_for(policy::Serial{}, Quint3(0), Quint3(8), [&] (auto coor) {
        g_pressure.add(coor);
    });

    Stencil(g_pressure).foreach(policy::Parallel{}, [&] (auto leafCoor, auto *leaf, auto callback) {
        callback([&] (auto coor, auto &value) {
            value = length(coor - 32.f) < 32.f ? 1.0f : 0.0f;
        });
    });

    Stencil(g_pressure).foreach_2x2x2_star(policy::Serial{},
    [&] (auto leafCoor, auto *leaf, auto callback) {
        auto *vel_leaf = g_velocity.add(leafCoor);
        callback([&] (auto coor
            , auto &value000
            , auto &value100
            , auto &value010
            , auto &value001
            ) {
            vel_leaf->at(coor) = Qfloat3(
                    value000 - value100,
                    value000 - value010,
                    value000 - value001);
        });
    });

    write_dense_vdb("/tmp/a.vdb", [&] (Quint3 coor) {
        return g_velocity.read_at(coor);
    }, Quint3(64, 64, 64));
    return 0;
}
