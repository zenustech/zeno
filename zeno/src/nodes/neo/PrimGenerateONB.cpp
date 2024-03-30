#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/para/parallel_for.h>
#include <zeno/utils/orthonormal.h>
#include <zeno/utils/zeno_p.h>

namespace zeno {

namespace {

struct PrimGenerateONB : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto dirAttr = get_input2<std::string>("dirAttr");
        auto tanAttrOut = get_input2<std::string>("tanAttrOut");
        auto bitanAttrOut = get_input2<std::string>("bitanAttrOut");
        auto writebackDir = get_input2<bool>("doNormalize");

        auto &dir = prim->verts.attr<vec3f>(dirAttr);
        auto &tan = prim->verts.add_attr<vec3f>(tanAttrOut);
        auto &bitan = prim->verts.add_attr<vec3f>(bitanAttrOut);

        parallel_for(prim->verts.size(), [&] (size_t i) {
            auto d = normalizeSafe(dir[i]);
            pixarONB(d, tan[i], bitan[i]);
            if (writebackDir)
                dir[i] = d;
        });

        set_output("prim", std::move(prim));
    }
};

ZENO_DEFNODE(PrimGenerateONB)({
    {
        {"", "prim", "", PrimarySocket},
        {"string", "dirAttr", "nrm"},
        {"string", "tanAttrOut", "tang"},
        {"string", "bitanAttrOut", "bitang"},
        {"bool", "writebackDir", "1"},
    },
    {
        "prim",
    },
    {},
    {"primitive"},
});

struct PrimLineGenerateONB : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto lineSort = get_input2<bool>("lineSort");
        auto dirAttrOut = get_input2<std::string>("dirAttrOut");
        auto tanAttrOut = get_input2<std::string>("tanAttrOut");
        auto bitanAttrOut = get_input2<std::string>("bitanAttrOut");

        size_t n = prim->verts.size();
        if (lineSort) primLineSort(prim.get());

        auto &dirs = prim->verts.add_attr<vec3f>(dirAttrOut);
        auto &tans = prim->verts.add_attr<vec3f>(tanAttrOut);
        auto &bitans = prim->verts.add_attr<vec3f>(bitanAttrOut);

        if (n >= 2) {
            parallel_for((size_t)1, n - 1, [&] (size_t i) {
                auto lastpos = prim->verts[i - 1];
                //auto currpos = prim->verts[i];
                auto nextpos = prim->verts[i + 1];
                auto direction = normalizeSafe(nextpos - lastpos);
                dirs[i] = direction;
            });
            dirs[0] = normalizeSafe(prim->verts[1] - prim->verts[0]);
            dirs[n - 1] = normalizeSafe(prim->verts[n - 1] - prim->verts[n - 2]);

            pixarONB(dirs[0], tans[0], bitans[0]);
            vec3f last_tangent = tans[0];
                //ZENO_P(dirs[0]);
                //ZENO_P(tans[0]);
                //ZENO_P(bitans[0]);
            for (size_t i = 1; i < n; i++) {
                guidedPixarONB(dirs[i], last_tangent, bitans[i]);
                tans[i] = last_tangent;
                //ZENO_P(dirs[i]);
                //ZENO_P(tans[i]);
                //ZENO_P(bitans[i]);
            }
        }

        set_output("prim", std::move(prim));
    }
};

}

ZENO_DEFNODE(PrimLineGenerateONB)({
    {
        {"", "prim", "", PrimarySocket},
        {"string", "dirAttrOut", "dir"},
        {"string", "tanAttrOut", "tang"},
        {"string", "bitanAttrOut", "bitang"},
        {"bool", "lineSort", "1"},
    },
    {
        "prim",
    },
    {},
    {"primitive"},
});

}
