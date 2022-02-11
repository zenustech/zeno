#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/StringObject.h>
#include <zeno/VDBGrid.h>
#include <zeno/utils/vec.h>
#include <zeno/zeno.h>
#include <zeno/ZenoInc.h>

namespace zeno {


struct VDBVoxelAsParticles : INode {
    virtual void apply() override {
        auto ingrid = get_input<VDBFloatGrid>("vdbGrid");
        auto const &grid = ingrid->m_grid;

        auto hasInactive = get_param<bool>("hasInactive");
        tbb::concurrent_vector<vec3f> pos;
        tbb::concurrent_vector<float> sdf;
        auto wrangler = [&](auto &leaf, openvdb::Index leafpos) {
            for (auto iter = leaf.cbeginValueOn(); iter != leaf.cendValueOn(); ++iter) {
                auto coord = iter.getCoord();
                auto value = iter.getValue();
                auto p = grid->transform().indexToWorld(coord.asVec3d());
                pos.emplace_back(p[0], p[1], p[2]);
                sdf.emplace_back(value);
            }
            if (hasInactive) {
                for (auto iter = leaf.cbeginValueOff(); iter != leaf.cendValueOff(); ++iter) {
                    auto coord = iter.getCoord();
                    auto value = iter.getValue();
                    auto p = grid->transform().indexToWorld(coord.asVec3d());
                    pos.emplace_back(p[0], p[1], p[2]);
                    sdf.emplace_back(value);
                }
            }
        };
        openvdb::tree::LeafManager<std::decay_t<decltype(grid->tree())>> leafman(grid->tree());
        leafman.foreach(wrangler);

        auto prim = std::make_shared<zeno::PrimitiveObject>();
        prim->resize(pos.size());
        auto &primPos = prim->add_attr<vec3f>("pos");
        for (int i = 0; i < pos.size(); i++) {
            primPos[i] = pos[i];
        }
        if (auto attr = get_param<std::string>("valToAttr"); attr.size()) {
            auto &primSdf = prim->add_attr<float>(attr);
            for (int i = 0; i < sdf.size(); i++) {
                primSdf[i] = sdf[i];
            }
        }
        set_output("primPars", std::move(prim));
    }
};

ZENDEFNODE(VDBVoxelAsParticles, {
                            {"vdbGrid"},
                            {"primPars"},
                            {
                             {"bool", "hasInactive", "0"},
                             {"string", "valToAttr", "sdf"},
                            },
                            {"visualize"},
                        });


struct VDBLeafAsParticles : INode {
    virtual void apply() override {
        auto ingrid = get_input<VDBFloatGrid>("vdbGrid");
        auto const &grid = ingrid->m_grid;
        auto h = grid->voxelSize()[0];
        tbb::concurrent_vector<vec3f> pos;
        auto wrangler = [&](auto &leaf, openvdb::Index leafpos) {
            auto coord = leaf.origin();
            auto p = grid->transform().indexToWorld(coord + decltype(coord)(8/2));
            //auto bbox = leaf.getNodeBoundingBox();
            //auto p = grid->transform().indexToWorld(bbox.min() + bbox.max());
            pos.emplace_back(p[0]-0.5*h, p[1]-0.5*h, p[2]-0.5*h);
        };
        openvdb::tree::LeafManager<std::decay_t<decltype(grid->tree())>> leafman(grid->tree());
        leafman.foreach(wrangler);

        auto prim = std::make_shared<zeno::PrimitiveObject>();
        prim->resize(pos.size());
        auto &primPos = prim->add_attr<vec3f>("pos");
        for (int i = 0; i < pos.size(); i++) {
            primPos[i] = pos[i];
        }
        set_output("primPars", std::move(prim));
    }
};

ZENDEFNODE(VDBLeafAsParticles, {
                            {"vdbGrid"},
                            {"primPars"},
                            {},
                            {"visualize"},
                        });

}
