#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveUV.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/types/NumericObject.h>
#include <zeno/para/parallel_for.h>
#include <zeno/utils/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>

namespace zeno {

ZENO_API void primFlipFaces(PrimitiveObject *prim, bool only_face) {
    if (!only_face && prim->lines.size())
        parallel_for_each(prim->lines.begin(), prim->lines.end(), [&] (auto &line) {
            std::swap(line[1], line[0]);
        });
    if (prim->tris.size()) {
        parallel_for_each(prim->tris.begin(), prim->tris.end(), [&] (auto &tri) {
            std::swap(tri[2], tri[0]);
        });
        if (prim->tris.attr_is<vec3f>("uv0")) {
            auto &uv0 = prim->tris.add_attr<zeno::vec3f>("uv0");
            auto &uv2 = prim->tris.add_attr<zeno::vec3f>("uv2");
            for (auto i = 0; i < prim->tris.size(); i++) {
                std::swap(uv0[i], uv2[i]);
            }
        }
        prim->tris.foreach_attr<AttrAcceptAll>([&](const auto &key, auto &values) {
            using T = std::decay_t<decltype(values[0])>;
            if constexpr (std::is_same_v<T, vec3i>) {
                if (key.rfind("auv_", 0) == 0) {
                    for (auto &value : values) {
                        std::swap(value[0], value[2]);
                    }
                }
            }
            const bool corner =
                key.rfind("__corner_", 0) == 0 && key.size() > 2 && key.substr(key.size() - 2) == "_0";
            bool oldUV = false;
            if constexpr (std::is_same_v<T, int>) {
                oldUV = key.rfind("uv", 0) == 0 && key.size() > 3 && key.back() == '0' &&
                        primitiveUVIndex("uvs_" + key.substr(2, key.size() - 3)) > 0;
            }
            if (corner || oldUV) {
                const auto other = key.substr(0, key.size() - 1) + "2";
                if (prim->tris.attr_is<T>(other)) {
                    values.swap(prim->tris.attr<T>(other));
                }
            }
        });
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
        primFlipFaces(prim.get(), false);
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
