#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveUtils.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/tuple_hash.h>
#include <unordered_map>

namespace zeno {
namespace {

struct PrimWeldClose : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto tagAttr = get_input<StringObject>("tagAttr")->get();
        float distance = get_input<NumericObject>("distance")->get<float>();

        float factor = 1.0f / distance;
        std::unordered_map<vec3i, int, tuple_hash, tuple_equal> lut;
        lut.reserve(prim->verts.size());
        for (int i = 0; i < prim->verts.size(); i++) {
            vec3f pos = prim->verts[i];
            vec3i posi = vec3i(pos * factor);
            auto [it, succ] = lut.emplace(posi, i);
            if (!succ) {
            }
        }

        if (tagAttr.size()) {
            auto &tag = prim->verts.add_attr<int>(tagAttr);
            std::fill(tag.begin(), tag.end(), 0);
            for (auto const &[key, idx]: lut) {
                tag[idx] = 1;
            }
        } else {
            std::vector<int> revamp;
            revamp.reserve(lut.size());
            for (auto const &[key, idx]: lut) {
                revamp.push_back(idx);
            }
            primRevampVerts(prim.get(), revamp);
        }

        set_output("prim", std::move(prim));
    }
};


ZENDEFNODE(PrimWeldClose, {
    {
    {"PrimitiveObject", "prim"},
    {"float", "distance", "0.00005"},
    {"string", "tagAttr", ""},
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
