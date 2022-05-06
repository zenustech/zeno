#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveUtils.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/tuple_hash.h>
#include <unordered_map>

namespace zeno {
namespace {

static void primMarkIsland(PrimitiveObject *prim, std::string tagAttr) {
    auto const &tris = prim->tris;
    auto n = tris.size();
    auto &tagarr = prim->add_attr<int>(tagAttr);
    auto m = tagarr.size();
    static std::vector<int> found(m);
    for (int i = 0; i < m; i++) {
        found[i] = i;
    }
    auto find = [&] (int i) {
        while (i != found[i])
            i = found[i];
        return i;
    };
    for (int i = 0; i < n; i++) {
        auto const &tri = tris[i];
        int e0 = find(tri[0]);
        int e1 = find(tri[1]);
        int e2 = find(tri[2]);
        found[e0] = e2;
        found[e0] = e1;
    }
    for (int i = 0; i < m; i++) {
        tagarr[i] = find(i);
    }
}

struct PrimMarkIsland : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto tagAttr = get_input<StringObject>("tagAttr")->get();

        primMarkIsland(prim.get(), tagAttr);

        set_output("prim", std::move(prim));
    }
};


ZENDEFNODE(PrimMarkIsland, {
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
