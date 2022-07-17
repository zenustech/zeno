#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/para/parallel_for.h>
#include <zeno/utils/log.h>
#include <limits>

namespace zeno {

namespace {

struct PrimMatchUVLine : INode {
    virtual void apply() override {
        std::vector<float>::iterator x;
        auto prim = get_input<PrimitiveObject>("prim");
        auto prim2 = get_input<PrimitiveObject>("prim2");
        auto uvAttr = get_input2<std::string>("uvAttr");
        auto posAttr = get_input2<std::string>("posAttr");
        auto uvAttr2 = get_input2<std::string>("uvAttr2");
        auto posAttr2 = get_input2<std::string>("posAttr2");

        if (!prim2->lines.size() || !prim2->verts.size())
            throw makeError("no lines connectivity found in prim2");

        auto const &uv = prim->verts.attr<float>(uvAttr);
        auto const &uv2 = prim2->verts.attr<float>(uvAttr2);
        prim2->verts.attr_visit(posAttr2, [&] (auto const &pos2) {

            using PosType = std::decay_t<decltype(pos2[0])>;
            auto &pos = prim->verts.add_attr<PosType>(posAttr);

            std::vector<std::vector<int>> neigh(prim2->verts.size());
            for (auto ind: prim2->lines) {
                neigh[ind[0]].push_back(ind[1]);
                neigh[ind[1]].push_back(ind[0]);
            }

            std::vector<float> uvs = uv2;
            std::sort(uvs.begin(), uvs.end());

            parallel_for(uv.size(), [&] (size_t i) {
                float val2 = uv[i]; // WARN: `uv` and `pos` may be the same array in some cases?
                size_t index = std::upper_bound(uvs.begin(), uvs.end(), val2) - uvs.begin();
                int idx1 = index == uvs.size() ? uvs.size() - 1 : index;
                auto const &neilst = neigh[idx1];
                int idx0 = idx1;
                for (auto neidx: neilst) {
                    if (uvs[neidx] <= uvs[idx1]) {
                        idx0 = neidx;
                        break;
                    }
                }
                float val1 = uvs[idx1], val0 = uvs[idx0];
                float fac = val2 - val0, eps = std::numeric_limits<float>::epsilon();
                fac /= std::max(std::abs(val1 - val0), eps);
                fac = std::clamp(fac, 0.f, 1.f);
                pos[i] = mix(pos2[idx0], pos2[idx1], fac);
            });
        });

        set_output("prim", std::move(prim));
    }
};

ZENO_DEFNODE(PrimMatchUVLine)({
    {
        "prim",
        "prim2",
        {"string", "uvAttr", "uv"},
        {"string", "posAttr", "pos"},
        {"string", "uvAttr2", "uv"},
        {"string", "posAttr2", "pos"},
    },
    {
        "prim",
    },
    {},
    {"primitive"},
});

}

}
