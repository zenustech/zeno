#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveUtils.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/tuple_hash.h>
#include <unordered_map>

namespace zeno {
namespace {

struct PrimMarkClose : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto tagAttr = get_input<StringObject>("tagAttr")->get();
        float distance = get_input<NumericObject>("distance")->get<float>();

        float factor = 1.0f / distance;
        std::unordered_multimap<vec3i, int, tuple_hash, tuple_equal> lut;
        lut.reserve(prim->verts.size());
        for (int i = 0; i < prim->verts.size(); i++) {
            vec3f pos = prim->verts[i];
            vec3i posi = vec3i(pos * factor);
            lut.emplace(posi, i);
        }

        auto &tag = prim->verts.add_attr<int>(tagAttr);
        if (lut.size()) {
            int cnt = 0;
            int last_idx = lut.begin()->second;
            for (auto const &[key, idx]: lut) {
                if (last_idx != idx) {
                    ++cnt;
                    last_idx = idx;
                }
                tag[idx] = cnt;
            }
        }

        set_output("prim", std::move(prim));
    }
};


ZENDEFNODE(PrimMarkClose, {
    {
    {"PrimitiveObject", "prim"},
    {"float", "distance", "0.00005"},
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
}
