#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveUtils.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/variantswitch.h>
#include <zeno/utils/logger.h>
#include <unordered_map>
#include <cassert>
#include <limits>

namespace zeno {

/* AWAK, NIKOLA TESLA'S JOB IS DJ, I.E. DJTESLA */
ZENO_API void primLineDistance(PrimitiveObject *prim, std::string resAttr, int start) {
    if (!prim->lines.size()) return;

    const float kFloatMax = std::numeric_limits<float>::max();

    std::unordered_multimap<int, std::pair<int, float>> neigh;
    for (int i = 0; i < prim->lines.size(); i++) {
        auto line = prim->lines[i];
        auto dist = distance(prim->verts[line[0]], prim->verts[line[1]]);
        neigh.emplace(line[0], std::pair{line[1], dist});
        neigh.emplace(line[1], std::pair{line[0], dist});
    }

    auto &result = prim->add_attr<float>(resAttr);
    std::fill(result.begin(), result.end(), kFloatMax);
    result[start] = 0;

    std::vector<float> table(prim->verts.size(), kFloatMax);
    {
        auto [b, e] = neigh.equal_range(start);
        for (auto it = b; it != e; ++it) {
            table[it->second.first] = it->second.second;
        }
    }
    table[start] = kFloatMax;

    for (int i = 0; i < prim->verts.size(); i++) {
        float minValue = kFloatMax;
        int minIndex = -1;
        for (int j = 0; j < table.size(); j++) {
            if (table[j] < minValue) {
                minValue = table[j];
                minIndex = j;
            }
        }
        assert(minIndex != -1);
        result[minIndex] = minValue;
        table[minIndex] = kFloatMax;
        for (int j = 0; j < table.size(); j++) {
            auto [b, e] = neigh.equal_range(minIndex);
            auto jit = std::find_if(b, e, [&] (auto const &pa) {
                return pa.first == j;
            });
            if (jit != e && result[j] < kFloatMax) {
                float newDist = result[minIndex] + jit->second.second;
                if (newDist < table[j]) {
                    table[j] = newDist;
                }
            }
        }
    }
}

struct PrimitiveLineDistance : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        primLineDistance(prim.get(),
                         get_input2<std::string>("resAttr"),
                         get_input2<int>("start"));
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimitiveLineDistance, {
    {
    {"PrimitiveObject", "prim"},
    {"string", "resAttr", "len"},
    {"int", "start", "0"},
    },
    {
    {"PrimitiveObject", "prim"},
    },
    {
    },
    {"primitive"},
});

}
