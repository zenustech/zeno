#include <cstdio>
#include <cassert>
#include <fdb/schedule.h>
#include <fdb/VDBGrid.h>
#include <fdb/openvdb.h>
#include "marching.h"

using namespace fdb;

int main() {
    vdbgrid::VDBGrid<float> g_sdf;

    ndrange_for(Serial{}, vec3i(0), vec3i(65), [&] (auto idx) {
        float value = 16.f - length(tofloat(idx));
        g_sdf.set(idx, value);
    });

    volumemesh::MarchingTetra mc(g_sdf);

    for (auto f: mc.triangles()) { f += 1;
        printf("f %d %d %d\n", f[0], f[1], f[2]);
    }
    for (auto v: mc.vertices()) {
        printf("v %f %f %f\n", v[0], v[1], v[2]);
    }

    write_dense_vdb("/tmp/a.vdb", [&] (auto idx) {
        return g_sdf.get(idx);
    }, vec3i(0), vec3i(64));

    return 0;
}
