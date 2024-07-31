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
        auto uvAttr2 = get_input2<std::string>("uvAttr2");
        auto copyOtherAttrs = get_input2<bool>("copyOtherAttrs");
        //auto dirAttrOut = get_input2<std::string>("dirAttrOut");

        if (!prim2->lines.size() || !prim2->verts.size())
            throw makeError("no lines connectivity found in prim2");

        auto const &uv = prim->verts.attr<float>(uvAttr);
        auto const &uv2 = prim2->verts.attr<float>(uvAttr2);

        auto &pos = prim->verts.values;
        auto &pos2 = prim2->verts.values;

        std::vector<std::vector<int>> neigh(prim2->verts.size());
        for (auto ind: prim2->lines) {
            neigh[ind[0]].push_back(ind[1]);
            neigh[ind[1]].push_back(ind[0]);
        }

        std::vector<float> uvs = uv2;
        std::sort(uvs.begin(), uvs.end());

        prim2->verts.foreach_attr([&] (auto const &key, auto &arr) {
            using T = std::decay_t<decltype(arr[0])>;
            prim->verts.add_attr<T>(key);
        });

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
            fac /= std::max(val1 - val0, eps);
            fac = std::clamp(fac, 0.f, 1.f);
            pos[i] = mix(pos2[idx0], pos2[idx1], fac);

            if (copyOtherAttrs) {
                prim2->verts.foreach_attr([&] (auto const &key, auto &arr) {
                    using T = std::decay_t<decltype(arr[0])>;
                    auto &arr1 = prim->verts.attr<T>(key);
                    arr1[i] = mix(arr[idx0], arr[idx1], fac);
                });
            }
        });

        set_output("prim", std::move(prim));
    }
};

ZENO_DEFNODE(PrimMatchUVLine)({
    {
        {"prim", "prim", "", zeno::Socket_ReadOnly},
        {"prim", "prim2", "", zeno::Socket_ReadOnly},
        {"string", "uvAttr", "tmp"},
        {"string", "uvAttr2", "tmp"},
        {"bool", "copyOtherAttrs", "1"},
    },
    {
        "prim",
    },
    {},
    {"primitive"},
});

}

}
