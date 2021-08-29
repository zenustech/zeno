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
    auto grid = get_input<VDBGrid>("vdbGrid");
    auto bbmin = zeno::IObject::make<zeno::NumericObject>();
    auto bbmax = zeno::IObject::make<zeno::NumericObject>();

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
    set_output("bmin", bbmin);
    set_output("bmax", bbmax);
  }
};

ZENDEFNODE(GetVDBBound, {
                            {"vdbGrid"},
                            {"bmin", "bmax"},
                            {},
                            {"openvdb"},
                        });

struct GetVDBVoxelSize : INode {
  virtual void apply() override {
    auto grid = get_input<VDBGrid>("vdbGrid");
    auto dx = zeno::IObject::make<zeno::NumericObject>();
    auto dy = zeno::IObject::make<zeno::NumericObject>();
    auto dz = zeno::IObject::make<zeno::NumericObject>();
    auto dxyz = zeno::IObject::make<zeno::NumericObject>();
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
                            {"vdbGrid"},
                            {"dx", "dy", "dz", "dxyz"},
                            {},
                            {"openvdb"},
                        });

struct VDBVoxelAsParticles : INode {
    virtual void apply() override {
        auto ingrid = get_input<VDBFloatGrid>("vdbGrid");
        auto const &grid = ingrid->m_grid;

        auto hasInactive = get_param<bool>("hasInactive");
        tbb::concurrent_vector<vec3i> pos;
        tbb::concurrent_vector<float> sdf;
        auto wrangler = [&](auto &leaf, openvdb::Index leafpos) {
            for (auto iter = leaf.beginValueOn(); iter != leaf.endValueOn(); ++iter) {
                auto coord = iter.getCoord();
                auto value = iter.getValue();
                auto p = grid->transform().indexToWorld(coord);
                pos.emplace_back(p[0], p[1], p[2]);
                sdf.emplace_back(value);
            }
            if (hasInactive) {
                for (auto iter = leaf.beginValueOff(); iter != leaf.endValueOff(); ++iter) {
                    auto coord = iter.getCoord();
                    auto value = iter.getValue();
                    auto p = grid->transform().indexToWorld(coord);
                    pos.emplace_back(p[0], p[1], p[2]);
                    sdf.emplace_back(value);
                }
            }
        };
        openvdb::tree::LeafManager<std::decay_t<decltype(grid->tree())>> leafman(grid->tree());
        leafman.foreach(wrangler);

        auto prim = std::make_shared<zeno::PrimitiveObject>();
        prim->resize(pos.size());
        auto &primPos = prim->add_attr<zeno::vec3f>("pos");
        auto &primSdf = prim->add_attr<float>("sdf");
        for (int i = 0; i < pos.size(); i++) {
            primPos[i] = pos[i];
        }
        for (int i = 0; i < sdf.size(); i++) {
            primSdf[i] = sdf[i];
        }
    }
}

ZENDEFNODE(VDBVoxelAsParticles, {
                            {"vdbGrid"},
                            {"primPars"},
                            {{"bool", "hasInactive", "0"}},
                            {"openvdb"},
                        });

}
