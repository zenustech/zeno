#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/para/parallel_for.h>
#include <zeno/utils/log.h>

namespace zeno {

// 'smart loop_uvs' to 'qianqiang loop.attr(uv)'
ZENO_API void primDecodeUVs(PrimitiveObject *prim) {
    if (prim->loop_uvs.size()) {
        auto &attr_uv = prim->loops.add_attr<vec3f>("uv"); // todo: support vec2f in attr...
        /*attr_uv.resize(prim->loop_uvs.size());*/
        parallel_for(prim->loop_uvs.size(), [&] (size_t i) {
            auto uv = prim->uvs[prim->loop_uvs[i]];
            attr_uv[i] = {uv[0], uv[1], 0};
        });
    }
    // for 'qianqiang loop.attr(uv)' to 'qianqiang tris.attr(uv0-3)'
    // please use primTriangulate (after calling primDecodeUVs)
}

// 'smart loop_uvs' to 'veryqianqiang vert.attr(uv)'
ZENO_API void primLoopUVsToVerts(PrimitiveObject *prim) {
    if (prim->loop_uvs.size()) {
        auto &vert_uvs = prim->verts.add_attr<vec3f>("uv"); // todo: support vec2f in attr...
        /*attr_uv.resize(prim->loop_uvs.size());*/
        for (size_t i = 0; i < prim->loop_uvs.size(); i++) {
            auto uv = prim->uvs[prim->loop_uvs[i]];
            int vertid = prim->loops[i];
            vert_uvs[vertid] = {uv[0], uv[1], 0};
            // uv may overlap and conflict at edges, but doesn't matter
            // this node is veryqianqiang after all, just to serve ZFX pw
        }
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
    {"primitive"},
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
