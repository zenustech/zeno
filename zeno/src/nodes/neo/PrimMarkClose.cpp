#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveUtils.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/tuple_hash.h>
#include <zeno/para/parallel_for.h>
#include <zeno/utils/log.h>
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
            vec3i posi = vec3i(floor(pos * factor));
            lut.emplace(posi, i);
        }

        auto &tag = prim->verts.add_attr<int>(tagAttr);
        if (lut.size()) {
            int cnt = 0;
            auto last_key = lut.begin()->first;
            for (auto const &[key, idx]: lut) {
                if (!tuple_equal{}(last_key, key)) {
                    ++cnt;
                    last_key = key;
                }
                tag[idx] = cnt;
            }
            zeno::log_info("PrimMarkClose: collapse from {} to {}", prim->verts.size(), cnt + 1);
        }

        set_output("prim", std::move(prim));
    }
};


ZENDEFNODE(PrimMarkClose, {
    {
    {"PrimitiveObject", "prim"},
    {"float", "distance", "0.00001"},
    {"string", "tagAttr", "tag"},
    },
    {
    {"PrimitiveObject", "prim"},
    },
    {
    },
    {"primitive"},
});


struct PrimMarkIndex : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto tagAttr = get_input<StringObject>("tagAttr")->get();
        int base = get_input<NumericObject>("base")->get<int>();
        int step = get_input<NumericObject>("step")->get<int>();
        auto type = get_input<StringObject>("type")->get();

        if (type == "float") {
            auto &tag = prim->verts.add_attr<float>(tagAttr);
            parallel_for((size_t)0, tag.size(), [&] (size_t i) {
                tag[i] = float(base + i * step);
            });
        } else {
            auto &tag = prim->verts.add_attr<int>(tagAttr);
            parallel_for((size_t)0, tag.size(), [&] (size_t i) {
                tag[i] = base + i * step;
            });
        }

        set_output("prim", std::move(prim));
    }
};


ZENDEFNODE(PrimMarkIndex, {
    {
    {"PrimitiveObject", "prim"},
    {"string", "tagAttr", "tag"},
    {"enum int float", "type", "int"},
    {"int", "base", "0"},
    {"int", "step", "1"},
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
