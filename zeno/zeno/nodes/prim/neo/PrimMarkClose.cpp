#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveUtils.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/tuple_hash.h>
#include <map>

namespace zeno {
namespace {

static void primMarkClose(PrimitiveObject *prim, std::string tagAttr, float distance = 0.0001f) {
    float factor = 1.0f / distance;
    std::multimap<vec3i, int, tuple_less> lut;
    for (int i = 0; i < prim->verts.size(); i++) {
        vec3f pos = prim->verts[i];
        vec3i posi = vec3i(floor(pos * factor + 0.5f));
        lut.emplace(posi, i);
    }

    auto &tag = prim->verts.add_attr<int>(tagAttr);
    if (lut.size()) {
        int cnt = 0;
        vec3i last_key = lut.begin()->first;
        for (auto const &[key, idx]: lut) {
            if (!tuple_equal{}(last_key, key)) {
                ++cnt;
                last_key = key;
            }
            printf("%d %d\n", idx, cnt);
            tag[idx] = cnt;
        }
    }
}

struct PrimMarkClose : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto tagAttr = get_input<StringObject>("tagAttr")->get();
        float distance = get_input<NumericObject>("distance")->get<float>();

        primMarkClose(prim.get(), tagAttr, distance);

        set_output("prim", std::move(prim));
    }
};


ZENDEFNODE(PrimMarkClose, {
    {
    {"PrimitiveObject", "prim"},
    {"float", "distance", "0.0001"},
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
