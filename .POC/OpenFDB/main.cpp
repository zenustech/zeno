#include <cstdio>
#include <cassert>
#include <fdb/converter.h>
#include <fdb/levelsetToMesh.h>
#include <fdb/PPGrid.h>
#include <set>
#include <vector>
#include <tuple>
#include <map>

using namespace fdb;

template <typename GridT>
typename GridT::Ptr readVdbGrid(const std::string &fn) {
  openvdb::io::File file(fn);
  file.open();
  openvdb::GridPtrVecPtr my_grids = file.getGrids();
  file.close();
  for (openvdb::GridPtrVec::iterator iter = my_grids->begin();
       iter != my_grids->end(); ++iter) {
    openvdb::GridBase::Ptr it = *iter;
    if ((*iter)->isType<GridT>()) {
      return openvdb::gridPtrCast<GridT>(*iter);
    }
  }
  throw "cannot readVdbGrid";
}

template <typename GridT>
void writeVdbGrid(const std::string &fn, typename GridT::Ptr grid) {
  openvdb::io::File(fn).write({grid});
}

int main() {
    ppgrid::PPGrid<float> sdf;

    openvdb::initialize();
    auto vdb = readVdbGrid<openvdb::FloatGrid>("/tmp/origin.vdb");

    converter::from_vdb_grid(sdf, *vdb);

#if 0
    sdf.foreach(Serial{}, [&] (auto ijk, auto &value) {
        value += 0.1f;
    });
#endif

#if 1
    std::vector<vec3f> vertices;
    std::vector<vec3I> triangles;
    fdb::levelsetToMesh::marching_tetra(sdf, vertices, triangles,
            /*isovalue=*/-0.001f);

    FILE *fp = fopen("/tmp/a.obj", "w");
    for (auto f: triangles) { f += 1;
        fprintf(fp, "f %d %d %d\n", f[0], f[1], f[2]);
    }
    for (auto v: vertices) { v *= vdb->transform().voxelSize()[0];
        fprintf(fp, "v %f %f %f\n", v[0], v[1], v[2]);
    }
    fclose(fp);
#endif

#if 1
    auto oldtrans = vdb->transform();
    vdb = openvdb::FloatGrid::create();
    vdb->setTransform(std::make_shared<openvdb::math::Transform>(oldtrans));
    converter::to_vdb_grid(sdf, *vdb);

    writeVdbGrid<openvdb::FloatGrid>("/tmp/a.vdb", vdb);
#endif

#if 0
    ppgrid::PPGrid<float> sdf;
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
#endif
}
