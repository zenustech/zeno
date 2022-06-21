#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/para/parallel_for.h>
#include <zeno/utils/log.h>

namespace zeno {

ZENO_API void primDecodeUVs(PrimitiveObject *prim) {
    if (prim->loop_uvs.size()) {
        auto &attr_uv = prim->loops.add_attr<vec3f>("uv"); // todo: support vec2f in attr...
        /*attr_uv.resize(prim->loop_uvs.size());*/
        parallel_for(prim->loop_uvs.size(), [&] (size_t i) {
            auto uv = prim->uvs[prim->loop_uvs[i]];
            attr_uv[i] = {uv[0], uv[1], 0};
        });
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

}

}
