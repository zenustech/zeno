#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/StringObject.h>
#include <zeno/VDBGrid.h>
#include <zeno/utils/vec.h>
#include <zeno/zeno.h>
#include <zeno/ZenoInc.h>

namespace zeno {


struct GetVDBBound : INode {
  virtual void apply() override {
    auto grid = safe_dynamic_cast<VDBGrid>(get_input("vdbGrid"));
    auto bbmin = std::make_unique<zeno::NumericObject>();
    auto bbmax = std::make_unique<zeno::NumericObject>();

    zeno::vec3f bmin, bmax;
    openvdb::CoordBBox box = grid->evalActiveVoxelBoundingBox();
    auto corner = box.min();
    auto length = box.max() - box.min();
    auto world_min = grid->indexToWorld(box.min());
    auto world_max = grid->indexToWorld(box.max());

    for (size_t d = 0; d < 3; d++) {
      bmin[d] = world_min[d];
      bmax[d] = world_max[d];
    }

    for (int dx = 0; dx < 2; dx++)
      for (int dy = 0; dy < 2; dy++)
        for (int dz = 0; dz < 2; dz++) {
          auto coord =
              corner + decltype(length){dx ? length[0] : 0, dy ? length[1] : 0,
                                        dz ? length[2] : 0};

          auto pos = grid->indexToWorld(coord);

          for (int d = 0; d < 3; d++) {
            bmin[d] = pos[d] < bmin[d] ? pos[d] : bmin[d];
            bmax[d] = pos[d] > bmax[d] ? pos[d] : bmax[d];
          }
        }

    bbmin->set<zeno::vec3f>(bmin);
    bbmax->set<zeno::vec3f>(bmax);
    set_output("bmin", std::move(bbmin));
    set_output("bmax", std::move(bbmax));
  }
};

ZENDEFNODE(GetVDBBound, {
                            {
                                {gParamType_VDBGrid,"vdbGrid", "", zeno::Socket_ReadOnly},
                            },
                            {{gParamType_Vec3f, "bmin"}, {gParamType_Vec3f, "bmax"}},
                            {},
                            {"openvdb"},
                        });

struct GetVDBVoxelSize : INode {
  virtual void apply() override {
    auto grid = safe_dynamic_cast<VDBGrid>(get_input("vdbGrid"));
    auto dx = std::make_unique<zeno::NumericObject>();
    auto dy = std::make_unique<zeno::NumericObject>();
    auto dz = std::make_unique<zeno::NumericObject>();
    auto dxyz = std::make_unique<zeno::NumericObject>();
    vec3f del = grid->getVoxelSize();
    dx->set(del[0]);
    dy->set(del[1]);
    dz->set(del[2]);
    dxyz->set(del);
    set_output("dx", std::move(dx));
    set_output("dy", std::move(dy));
    set_output("dz", std::move(dz));
    set_output("dxyz", std::move(dxyz));
  }
};

ZENDEFNODE(GetVDBVoxelSize, {
                            {
                                {gParamType_VDBGrid,"vdbGrid", "", zeno::Socket_ReadOnly},
                            },
                            {{gParamType_Float,"dx"}, {gParamType_Float,"dy"}, {gParamType_Float,"dz"}, {gParamType_Vec3f,"dxyz"}},
                            {},
                            {"openvdb"},
                        });

}
