#include "zeno/types/PrimitiveObject.h"
#include "zeno/types/StringObject.h"
#include <mutex>
#include <openvdb/Types.h>
#include <thread>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/StringObject.h>
#include <zeno/VDBGrid.h>
#include <zeno/utils/vec.h>
#include <zeno/zeno.h>
#include <zeno/ZenoInc.h>

namespace zeno {
struct ParticleAsVoxels : INode{
    virtual void apply() override{
        auto type = get_input<VDBGrid>("vdbGrid")->getType();
        
        if(type=="FloatGrid"){
            auto ingrid = get_input<VDBFloatGrid>("vdbGrid");
            auto const &grid = ingrid->m_grid;
            auto inparticles = get_input<PrimitiveObject>("particles");
            auto attrName = get_input<StringObject>("Attr")->value;
            

            inparticles->attr_visit(attrName, [&](auto &arr) {
            #pragma omp parallel for
                for (int i = 0; i < arr.size(); i++) {
                    if constexpr (is_decay_same_v<decltype(arr[i]), vec3f>) {
                    } else {
                        auto accessor = grid->getUnsafeAccessor();
                        openvdb::Vec3d p(inparticles->verts[i][0], inparticles->verts[i][1], inparticles->verts[i][2]);
                        openvdb::Coord coord(grid->worldToIndex(p).x(),grid->worldToIndex(p).y(),grid->worldToIndex(p).z());
                        accessor.setValue(coord, arr[i]);
                    }
                }
            });
            set_output("oGrid", std::move(ingrid));
        }
        if(type=="Vec3fGrid") {
            auto ingrid = get_input<VDBFloat3Grid>("vdbGrid");
            auto const &grid = ingrid->m_grid;
            auto inparticles = get_input<PrimitiveObject>("particles");
            auto attrName = get_input<StringObject>("Attr")->value;
            
            inparticles->attr_visit(attrName, [&](auto &arr) {
            #pragma omp parallel for
                for (int i = 0; i < arr.size(); i++) {
                    if constexpr (is_decay_same_v<decltype(arr[i]), vec3f>) {

                        auto accessor = grid->getUnsafeAccessor();
                        openvdb::Vec3d p(inparticles->verts[i][0], inparticles->verts[i][1], inparticles->verts[i][2]);
                        openvdb::Coord coord(grid->worldToIndex(p).x(),grid->worldToIndex(p).y(),grid->worldToIndex(p).z());
                        accessor.setValue(coord, openvdb::Vec3f(arr[i][0], arr[i][1], arr[i][2]));
                    } else {
                    }
                }
            });
            set_output("oGrid", std::move(ingrid));
        }
        
        
    }
};
ZENDEFNODE(ParticleAsVoxels, {
                            {{"VDBGrid", "vdbGrid"}, 
                             {"string", "Attr"},
                             {"particles"},
                            },
                            {"oGrid"},
                            
                            {
                             
                            },
                            {"visualize"},
                        });
struct VDBVoxelAsParticles : INode {
    virtual void apply() override {
        auto type = get_input<VDBGrid>("vdbGrid")->getType();
        if(type == "FloatGrid"){
            auto ingrid = get_input<VDBFloatGrid>("vdbGrid");
            auto const &grid = ingrid->m_grid;

            auto hasInactive = get_param<bool>("hasInactive");
            // tbb::concurrent_vector<vec3f> pos;
            // wxl
#if 1
            using MapT = std::map<std::thread::id, std::vector<vec3f>>;
            using IterT = typename MapT::iterator;
            std::map<std::thread::id, std::vector<vec3f>> poses;
            std::mutex mutex;
#endif

            //tbb::concurrent_vector<float> sdf;
            auto wrangler = [&](auto &leaf, openvdb::Index leafpos) {
                // wxl
#if 1
                IterT iter;
                {
                    std::lock_guard<std::mutex> lk(mutex);
                    bool tag;
                    std::tie(iter, tag) = poses.insert(std::make_pair(std::this_thread::get_id(), std::vector<vec3f>{}));
                }
                auto &pos_ = iter->second;
#endif
                for (auto iter = leaf.cbeginValueOn(); iter != leaf.cendValueOn(); ++iter) {
                    auto coord = iter.getCoord();
                    auto value = iter.getValue();
                    auto p = grid->transform().indexToWorld(coord.asVec3d());
                    // pos.emplace_back(p[0], p[1], p[2]);
                    // wxl
                    pos_.emplace_back(p[0], p[1], p[2]);
                    //sdf.emplace_back(value);
                }
                if (hasInactive) {
                    for (auto iter = leaf.cbeginValueOff(); iter != leaf.cendValueOff(); ++iter) {
                        auto coord = iter.getCoord();
                        auto value = iter.getValue();
                        auto p = grid->transform().indexToWorld(coord.asVec3d());
                        // pos.emplace_back(p[0], p[1], p[2]);
                        // wxl
                        pos_.emplace_back(p[0], p[1], p[2]);
                        //sdf.emplace_back(value);
                    }
                }
            };
            openvdb::tree::LeafManager<std::decay_t<decltype(grid->tree())>> leafman(grid->tree());
            leafman.foreach(wrangler);

#if 1
            std::vector<vec3f> zspos;
            for (const auto &[_, pos] : poses) {
                zspos.insert(std::end(zspos), std::begin(pos), std::end(pos));
            }
            // printf("concurrent vec of size %d, zs pos size %d\n", pos.size(), zspos.size());
#endif

            auto prim = std::make_shared<zeno::PrimitiveObject>();
            prim->resize(zspos.size());
            auto &primPos = prim->add_attr<vec3f>("pos");
            // wxl
            primPos = zspos;
#if 0
            for (int i = 0; i < zspos.size(); i++) {
                primPos[i] = zspos[i];
            }
#endif
            set_output("primPars", std::move(prim));
        }
        else if(type == "Vec3fGrid")
        {
            auto ingrid = get_input<VDBFloat3Grid>("vdbGrid");
            auto const &grid = ingrid->m_grid;

            auto hasInactive = get_param<bool>("hasInactive");
            // tbb::concurrent_vector<vec3f> pos;
            //tbb::concurrent_vector<float> sdf;
            // wxl
#if 1
            using MapT = std::map<std::thread::id, std::vector<vec3f>>;
            using IterT = typename MapT::iterator;
            std::map<std::thread::id, std::vector<vec3f>> poses;
            std::mutex mutex;
#endif
            auto wrangler = [&](auto &leaf, openvdb::Index leafpos) {
                // wxl
#if 1
                IterT iter;
                {
                    std::lock_guard<std::mutex> lk(mutex);
                    bool tag;
                    std::tie(iter, tag) = poses.insert(std::make_pair(std::this_thread::get_id(), std::vector<vec3f>{}));
                }
                auto &pos_ = iter->second;
#endif
                for (auto iter = leaf.cbeginValueOn(); iter != leaf.cendValueOn(); ++iter) {
                    auto coord = iter.getCoord();
                    auto value = iter.getValue();
                    auto p = grid->transform().indexToWorld(coord.asVec3d() - openvdb::Vec3d(0.5, 0, 0));
                    // pos.emplace_back(p[0], p[1], p[2]);
                    pos_.emplace_back(p[0], p[1], p[2]);
                    p = grid->transform().indexToWorld(coord.asVec3d() - openvdb::Vec3d(0, 0.5, 0));
                    // pos.emplace_back(p[0], p[1], p[2]);
                    pos_.emplace_back(p[0], p[1], p[2]);
                    p = grid->transform().indexToWorld(coord.asVec3d() - openvdb::Vec3d(0, 0, 0.5));
                    // pos.emplace_back(p[0], p[1], p[2]);
                    pos_.emplace_back(p[0], p[1], p[2]);
                    //sdf.emplace_back(value);
                }
                if (hasInactive) {
                    for (auto iter = leaf.cbeginValueOff(); iter != leaf.cendValueOff(); ++iter) {
                        auto coord = iter.getCoord();
                        auto value = iter.getValue();
                        auto p = grid->transform().indexToWorld(coord.asVec3d() - openvdb::Vec3d(0.5, 0, 0));
                        // pos.emplace_back(p[0], p[1], p[2]);
                        pos_.emplace_back(p[0], p[1], p[2]);
                        p = grid->transform().indexToWorld(coord.asVec3d() - openvdb::Vec3d(0, 0.5, 0));
                        // pos.emplace_back(p[0], p[1], p[2]);
                        pos_.emplace_back(p[0], p[1], p[2]);
                        p = grid->transform().indexToWorld(coord.asVec3d() - openvdb::Vec3d(0, 0, 0.5));
                        // pos.emplace_back(p[0], p[1], p[2]);
                        pos_.emplace_back(p[0], p[1], p[2]);
                        //sdf.emplace_back(value);
                    }
                }
            };
            openvdb::tree::LeafManager<std::decay_t<decltype(grid->tree())>> leafman(grid->tree());
            leafman.foreach(wrangler);

#if 1
            std::vector<vec3f> zspos;
            for (const auto &[_, pos] : poses) {
                zspos.insert(std::end(zspos), std::begin(pos), std::end(pos));
            }
            // printf("concurrent vec of size %d, zs pos size %d\n", pos.size(), zspos.size());
#endif

            auto prim = std::make_shared<zeno::PrimitiveObject>();
            prim->resize(zspos.size());
            auto &primPos = prim->add_attr<vec3f>("pos");
            // wxl
            primPos = zspos;
#if 0
            for (int i = 0; i < pos.size(); i++) {
                primPos[i] = pos[i];
            }
#endif
            set_output("primPars", std::move(prim));
        }
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
