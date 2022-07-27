#include <zeno/zeno.h>
#include <zeno/VDBGrid.h>
#include <zeno/types/NumericObject.h>
#include <openvdb/openvdb.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/points/PointCount.h>
#include <openvdb/points/PointAdvect.h>
#include <openvdb/tools/Interpolation.h>
#include <zeno/utils/variantswitch.h>
#include <zeno/utils/arrayindex.h>
#include <zeno/utils/perlin.h>
#include <zeno/utils/vec.h>
#include <zeno/utils/log.h>
#include <zeno/utils/zeno_p.h>

namespace {
using namespace zeno;

template <class T>
struct fuck_openvdb_vec {
    using type = T;
};

template <>
struct fuck_openvdb_vec<openvdb::Vec3f> {
    using type = vec3f;
};

struct VDBPerlinNoise : INode {
  virtual void apply() override {
    auto inoutSDF = get_input<VDBFloatGrid>("inoutSDF");
        auto scale = get_input2<float>("scale");
        auto scale3d = get_input2<vec3f>("scale3d");
        auto detail = get_input2<float>("detail");
        auto roughness = get_input2<float>("roughness");
        auto disortion = get_input2<float>("disortion");
        auto offset = get_input2<vec3f>("offset");
        auto average = get_input2<float>("average");
        auto strength = get_input2<float>("strength");

    auto grid = inoutSDF->m_grid;
    float dx = grid->voxelSize()[0];
    strength *= dx;
    scale3d *= scale * dx;

    auto wrangler = [&](auto &leaf, openvdb::Index leafpos) {
        for (auto iter = leaf.beginValueOn(); iter != leaf.endValueOn(); ++iter) {
            auto coord = iter.getCoord();
            using OutT = typename fuck_openvdb_vec<std::decay_t<
                typename std::decay_t<decltype(leaf)>::ValueType>>::type;
            OutT noise;
            {
                vec3f p(coord[0], coord[1], coord[2]);
                p = scale3d * (p - offset);
                OutT o;
                if constexpr (std::is_same_v<OutT, float>) {
                    o = PerlinNoise::perlin(p, roughness, detail);
                } else if constexpr (std::is_same_v<OutT, vec3f>) {
                    o = OutT(
                        PerlinNoise::perlin(vec3f(p[0], p[1], p[2]), roughness, detail),
                        PerlinNoise::perlin(vec3f(p[1], p[2], p[0]), roughness, detail),
                        PerlinNoise::perlin(vec3f(p[2], p[0], p[1]), roughness, detail));
                } else {
                    throw makeError<TypeError>(typeid(vec3f), typeid(OutT), "outType");
                }
                noise = average + o * strength;
            }
            iter.modifyValue([&] (auto &v) {
                v += noise;
            });
        }
    };
    auto velman = openvdb::tree::LeafManager<std::decay_t<decltype(grid->tree())>>(grid->tree());
    velman.foreach(wrangler);

    set_output("inoutSDF", get_input("inoutSDF"));
  }
};

ZENO_DEFNODE(VDBPerlinNoise)(
     { /* inputs: */ {
     "inoutSDF",
    {"float", "scale", "5"},
    {"vec3f", "scale3d", "1,1,1"},
    {"float", "detail", "2"},
    {"float", "roughness", "0.5"},
    {"float", "disortion", "0"},
    {"vec3f", "offset", "0,0,0"},
    {"float", "average", "0"},
    {"float", "strength", "1"},
     }, /* outputs: */ {
       "inoutSDF",
     }, /* params: */ {
     }, /* category: */ {
     "openvdb",
     }});

}
