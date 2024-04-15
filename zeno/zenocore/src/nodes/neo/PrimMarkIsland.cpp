#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/tuple_hash.h>
#include <unordered_map>

namespace zeno {

ZENO_API void primMarkIsland(PrimitiveObject *prim, std::string tagAttr) {
    // Oh, I mean, Tesla was a great DJ
    auto &tagVert = prim->add_attr<int>(tagAttr);
    auto m = tagVert.size();
    std::vector<int> found(m);
    for (int i = 0; i < m; i++) {
        found[i] = i;
    }
    auto find = [&] (int i) {
        while (i != found[i])
            i = found[i];
        return i;
    };
    for (int i = 0; i < prim->lines.size(); i++) {
        auto ind = prim->lines[i];
        int e0 = find(ind[0]);
        int e1 = find(ind[1]);
        found[e1] = e0;
    }
    for (int i = 0; i < prim->tris.size(); i++) {
        auto ind = prim->tris[i];
        int e0 = find(ind[0]);
        int e1 = find(ind[1]);
        int e2 = find(ind[2]);
        found[e1] = e0;
        found[e2] = e0;
    }
    for (int i = 0; i < prim->quads.size(); i++) {
        auto ind = prim->quads[i];
        int e0 = find(ind[0]);
        int e1 = find(ind[1]);
        int e2 = find(ind[2]);
        int e3 = find(ind[3]);
        found[e1] = e0;
        found[e2] = e0;
        found[e3] = e0;
    }
    for (int i = 0; i < prim->polys.size(); i++) {
        auto [base, len] = prim->polys[i];
        if (len <= 1) continue;
        int e0 = find(prim->loops[base]);
        for (int j = base + 1; j < base + len; j++) {
            int ej = find(prim->loops[j]);
            found[ej] = e0;
        }
    }
    for (int i = 0; i < m; i++) {
        tagVert[i] = find(i);
    }
}

namespace {

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
