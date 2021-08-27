#include <cstdio>
#include <cassert>
#include <fdb/schedule.h>
#include <fdb/VDBGrid.h>
#include <fdb/openvdb.h>
#include <fdb/levelsetToMesh.h>
#include <set>
#include <vector>
#include <tuple>
#include <map>

using namespace fdb;

int main() {
    vdbgrid::VDBGrid<float> sdf;
    ndrange_for(Serial{}, vec3i(0), vec3i(64), [&] (auto idx) {
        float value = max(-4.0f, length(idx - 32.f) - 10.9f);
        sdf.set(idx, value);
    });

    fdb::levelsetToMesh::MarchingTetra mt(sdf);
    mt.march();

    FILE *fp = fopen("/tmp/a.obj", "w");
    for (auto f: mt.triangles()) { f += 1;
        fprintf(fp, "f %d %d %d\n", f[0], f[1], f[2]);
    }
    for (auto v: mt.vertices()) {
        fprintf(fp, "v %f %f %f\n", v[0], v[1], v[2]);
    }
    fclose(fp);

    /*write_dense_vdb("/tmp/a.vdb", [&] (auto idx) {
        return g_sdf.get(idx);
    }, vec3i(0), vec3i(64));*/

    return 0;
}
