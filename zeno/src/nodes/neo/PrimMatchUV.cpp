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
        auto prim = get_input<PrimitiveObject>("prim");
        auto prim2 = get_input<PrimitiveObject>("prim2");
        auto uvAttr = get_input2<std::string>("uvAttr");
        auto posAttr = get_input2<std::string>("posAttr");
        auto uvAttr2 = get_input2<std::string>("uvAttr2");
        auto posAttr2 = get_input2<std::string>("posAttr2");

        auto const &uv = prim->verts.attr<float>(uvAttr);
        auto const &uv2 = prim2->verts.attr<float>(uvAttr2);
        prim2->verts.attr_visit(posAttr2, [&] (auto const &pos2) {

            using PosType = std::decay_t<decltype(pos2[0])>;
            auto &pos = prim->verts.add_attr<PosType>(posAttr);

            std::map<int, std::vector<int>> neigh;
            for (auto ind: prim2->lines) {
                neigh[ind[0]].push_back(ind[1]);
                neigh[ind[1]].push_back(ind[0]);
            }

            std::vector<float> uvs;
            std::vector<int> inds;
            for (auto const &[k, v]: neigh) {
                uvs.push_back(uv2[k]);
                inds.push_back(k);
            }
            if (inds.empty())
                throw makeError("no lines connectivity in prim2");

            for (int i = 0; i < uv.size(); i++) {
                float val2 = uv[i]; // WARN: `uv` and `pos` may be the same array in some cases
                size_t index = std::lower_bound(uvs.begin(), uvs.end(), val2) - uvs.begin();
                int idx0 = index == uvs.size() ? inds.back() : inds[index];
                auto const &neilst = neigh.at(idx0);
                int idx1 = idx0;
                for (auto neidx: neilst) {
                    if (uv2[neidx] > uv2[idx0]) {
                        idx1 = neidx;
                        break;
                    }
                }
                float val1 = uv2[idx1], val0 = uv2[idx0];
                float fac = val2 - val0, eps = std::numeric_limits<float>::epsilon();
                fac /= std::max(std::abs(val1 - val0), eps);
                pos[i] = mix(pos2[idx0], pos2[idx1], fac);
            }
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
