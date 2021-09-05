#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/parallel.h>

namespace zeno {


struct PrimitiveCalcCentroid : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto &pos = prim->attr<vec3f>("pos");

        vec4f acc = parallel_reduce_array(prim->tris.size(), vec4f(0), [&] (size_t i) {
            auto ind = prim->tris[i];
            auto a = pos[ind[0]], b = pos[ind[1]], c = pos[ind[2]];
            auto weight = length(cross(b - a, c - a));
            auto center = weight * (a + b + c);
            return vec4f(center[0], center[1], center[2], weight);
        }, [&] (auto a, auto b) { return a + b; });
        auto centroid = acc[3] == 0 ? vec3f(0) :
            vec3f(acc[0], acc[1], acc[2]) / (3.f * acc[3]);

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
