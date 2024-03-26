#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/types/NumericObject.h>
#include <zeno/para/parallel_for.h>
#include <zeno/utils/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>

namespace zeno {

ZENO_API void primFlipFaces(PrimitiveObject *prim) {
    if (prim->lines.size())
        parallel_for_each(prim->lines.begin(), prim->lines.end(), [&] (auto &line) {
            std::swap(line[1], line[0]);
        });
    if (prim->tris.size()) {
        parallel_for_each(prim->tris.begin(), prim->tris.end(), [&] (auto &tri) {
            std::swap(tri[2], tri[0]);
        });
        if (prim->tris.attr_is<vec3f>("uv0")) {
            auto &uv1 = prim->tris.add_attr<vec3f>("uv1");
            auto &uv2 = prim->tris.add_attr<vec3f>("uv2");
            for (auto i = 0; i < prim->tris.size(); i++) {
                std::swap(uv1[i], uv2[i]);
            }
        }
    }
    if (prim->quads.size())
        parallel_for_each(prim->quads.begin(), prim->quads.end(), [&] (auto &quad) {
            std::swap(quad[3], quad[0]);
            std::swap(quad[2], quad[1]);
        });
    if (prim->polys.size()) {
        prim->loops.forall_attr<AttrAcceptAll>([&] (auto const &key, auto &arr) {
            parallel_for_each(prim->polys.begin(), prim->polys.end(), [&] (auto const &poly) {
                auto const &[start, len] = poly;
                for (int i = 0; i < (len >> 1); i++) {
                    std::swap(arr[start + i], arr[start + len - 1 - i]);
                }
            });
        });
    }
}

struct PrimFlipFaces : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        primFlipFaces(prim.get());
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimFlipFaces, {
    {"prim"},
    {"prim"},
    {},
    {"primitive"},
});

};
