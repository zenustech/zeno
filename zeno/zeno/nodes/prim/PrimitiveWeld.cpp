#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/tuple_hash.h>
#include <zeno/utils/value_type.h>
#include <unordered_map>

namespace zeno {

struct PrimitiveWeld : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto tagAttr = get_input<StringObject>("ind")->get();

        float dx = 0.00001f;
        float inv_dx = 1.0f / dx;

        std::unordered_map<vec3i, int, tuple_hash, tuple_equal> lut;
        lut.reserve(prim->verts.size());
        for (int i = 0; i < prim->verts.size(); i++) {
            vec3f pos = prim->verts[i];
            vec3i posi = vec3i(pos * inv_dx);
            lut.emplace(posi, i);
        }

        auto &indices = prim->verts.add_attr<int>(tagAttr);
        int curr = 0;
        std::fill(indices.begin(), indices.end(), 0);
        for (auto const &[key, idx]: lut) {
            indices[idx] = 1;
        }

        set_output("prim", std::move(prim));
    }
};


ZENDEFNODE(PrimitiveWeld, {
    {
    {"PrimitiveObject", "prim"},
    {"string", "tagAttr", "tag"},
    },
    {
    {"PrimitiveObject", "prim"},
    },
    {
    },
    {"primitive"},
});

}

