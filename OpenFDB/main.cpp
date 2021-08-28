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
        float value = max(0.f, 10.9f - length(idx - 48.f));
        sdf.set(idx, value);
    });

    std::vector<vec3f> vertices;
    std::vector<vec3I> triangles;
    fdb::levelsetToMesh::marching_tetra(sdf, vertices, triangles,
            /*isovalue=*/1.0f);

    FILE *fp = fopen("/tmp/a.obj", "w");
    for (auto f: triangles) { f += 1;
        fprintf(fp, "f %d %d %d\n", f[0], f[1], f[2]);
    }
    for (auto v: vertices) {
        fprintf(fp, "v %f %f %f\n", v[0], v[1], v[2]);
    }
    fclose(fp);

    write_dense_vdb("/tmp/a.vdb", [&] (auto idx) {
        return sdf.get(idx);
    }, vec3i(0), vec3i(64));

    return 0;
}
