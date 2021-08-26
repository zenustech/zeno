#include <cstdio>
#include <fdb/policy.h>
#include <fdb/SPGrid.h>
#include <fdb/openvdb.h>

using namespace fdb;

int main() {
    spgrid::SPFloatGrid<12> g_pre;
    spgrid::SPFloat3Grid<12> g_vel;

    ndrange_for(policy::Serial{},
    vec3i(-64), vec3i(64), [&] (auto idx) {
        float value = max(0.f, 40.f - length(tofloat(idx)));
        g_pre.set(idx, value);
    });

    /*ndrange_for(policy::Serial{},
    vec3i(1), vec3i(127), [&] (auto idx) {
        float c = g_pre.get(idx);
        g_vel.set(idx, vec<float, 3>(
                c - g_pre.get(i+1, j, k),
                c - g_pre.get(i, j+1, k),
                c - g_pre.get(i, j, k+1)));
    });*/

    write_dense_vdb("/tmp/a.vdb", [&] (auto idx) {
        return abs(g_pre.get(idx));
    }, vec3i(-64), vec3i(64));

    return 0;
}
