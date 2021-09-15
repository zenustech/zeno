#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/parallel.h>

namespace zeno {


// This is an area method...
// TODO: use this volume method instead:
// https://github.com/bulletphysics/bullet3/blob/master/examples/VoronoiFracture/VoronoiFractureDemo.cpp#L362
struct PrimitiveCalcCentroid : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto &pos = prim->attr<vec3f>("pos");

        vec4f acc;
        if (prim->tris.size()) {
            acc = parallel_reduce_array(prim->tris.size(), vec4f(0), [&] (size_t i) {
                auto ind = prim->tris[i];
                auto a = pos[ind[0]], b = pos[ind[1]], c = pos[ind[2]];
                auto weight = length(cross(b - a, c - a));
                auto center = weight / 3.0f * (a + b + c);
                return vec4f(center[0], center[1], center[2], weight);
            }, [&] (auto a, auto b) { return a + b; });
        } else {
            acc = parallel_reduce_array(prim->size(), vec4f(0), [&] (size_t i) {
                auto pos = prim->verts[i];
                return vec4f(pos[0], pos[1], pos[2], 1.0f);
            }, [&] (auto a, auto b) { return a + b; });
        }

        auto centroid = acc[3] == 0 ? vec3f(0) :
            vec3f(acc[0], acc[1], acc[2]) / acc[3];

        set_output2("centroid", centroid);
        set_output2("totalArea", acc[3]);
    }
};

ZENDEFNODE(PrimitiveCalcCentroid, {
    {
    {"PrimitiveObject", "prim"},
    },
    {
    {"vec3f", "centroid"},
    {"float", "totalArea"},
    },
    {},
    {"primitive"},
});


}
