#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/para/parallel_for.h>
#include <zeno/utils/log.h>

namespace zeno {

// 'smart loop_uvs' to 'qianqiang loop.attr(uv)'
ZENO_API void primDecodeUVs(PrimitiveObject *prim) {
}

// 'smart loop_uvs' to 'veryqianqiang vert.attr(uv)'
ZENO_API void primLoopUVsToVerts(PrimitiveObject *prim) {
    if (prim->loops.size() && prim->has_attr("uvs")) {
        auto &loop_uvs = prim->loops.attr<int>("uvs");
        auto &vert_uv = prim->verts.add_attr<vec3f>("uv"); // todo: support vec2f in attr...
        /*attr_uv.resize(prim->loop_uvs.size());*/
        for (size_t i = 0; i < loop_uvs.size(); i++) {
            auto uv = prim->uvs[loop_uvs[i]];
            int vertid = prim->loops[i];
            vert_uv[vertid] = {uv[0], uv[1], 0};
            // uv may overlap and conflict at edges, but doesn't matter
            // this node is veryqianqiang after all, just to serve ZFX pw
        }
        prim->loops.erase_attr("uvs");
    }
}

namespace {

struct PrimDecodeUVs : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        primDecodeUVs(prim.get());
        set_output("prim", std::move(prim));
    }
};

ZENO_DEFNODE(PrimDecodeUVs)({
    {
        "prim",
    },
    {
        "prim",
    },
    {},
    {"deprecated"},
});

struct PrimLoopUVsToVerts : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        primLoopUVsToVerts(prim.get());
        set_output("prim", std::move(prim));
    }
};

ZENO_DEFNODE(PrimLoopUVsToVerts)({
    {
        "prim",
    },
    {
        "prim",
    },
    {},
    {"primitive"},
});

}

}
