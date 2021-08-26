#include <cstdio>
#include <cassert>
#include <fdb/schedule.h>
#include <fdb/VDBGrid.h>
#include <fdb/openvdb.h>
#include <map>

using namespace fdb;

int main() {
    ndrange_for(Serial{}, vec3i(0), vec3i(64), [&] (auto idx) {
        float value = 16.f - length(tofloat(idx));
        g_sdf.set(idx, value);
    });

    marching_tetra();

    for (auto f: g_triangles) { f += 1;
        printf("f %d %d %d\n", f[0], f[1], f[2]);
    }
    for (auto v: g_vertices) {
        printf("v %f %f %f\n", v[0], v[1], v[2]);
    }

    write_dense_vdb("/tmp/a.vdb", [&] (auto idx) {
        return g_sdf.get(idx);
    }, vec3i(0), vec3i(64));

    return 0;
}
