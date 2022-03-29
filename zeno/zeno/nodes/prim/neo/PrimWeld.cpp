#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveUtils.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <unordered_map>

namespace zeno {
namespace {

struct PrimWeld : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto tagAttr = get_input<StringObject>("tagAttr")->get();

        std::unordered_multimap<int, int> lut;
        auto &tag = prim->verts.attr<int>(tagAttr);
        for (int i = 0; i < prim->size(); i++) {
            lut.insert({tag[i], i});
        }
        for (auto it = lut.begin(); it != lut.end();) {
            auto nit = std::find_if(std::next(it), lut.end(), [val = it->first] (auto const &p) {
                return p.first != val;
            });
            for (; it != lut.end(); ++it) {
                int idx = it->second;
            }
        }

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimWeld, {
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
}
