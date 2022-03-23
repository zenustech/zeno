#pragma once

#include <zeno/utils/vec.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveUtils.h>
#include <spdlog/spdlog.h>

namespace zeno {

static void prim_triangulate(PrimitiveObject *prim) {
    prim->tris.clear();
    prim->tris.reserve(prim->polys.size());

    for (auto [start, len]: prim->polys) {
        if (len < 3) continue;
        for (int i = 2; i < len; i++) {
            prim->tris.emplace_back(
                    prim->loops[start],
                    prim->loops[start + i - 1],
                    prim->loops[start + i]);
        }
    }
    if (prim->loops.has_attr("uv_value")) {
        auto &uv0 = prim->tris.add_attr<zeno::vec3f>("uv0");
        auto &uv1 = prim->tris.add_attr<zeno::vec3f>("uv1");
        auto &uv2 = prim->tris.add_attr<zeno::vec3f>("uv2");

        auto &uv_value = prim->loops.attr<zeno::vec3f>("uv_value");
        auto &uv_loops = prim->loops.attr<int>("uv_loops");
        for (auto [start, len]: prim->polys) {
            if (len < 3) continue;
            for (int i = 2; i < len; i++) {
                int _0 = uv_loops[start];
                int _1 = uv_loops[start + i - 1];
                int _2 = uv_loops[start + i];
                prim->tris.attr<zeno::vec3f>("uv0").push_back(zeno::vec3f(0.5,0.5, 0));
                prim->tris.attr<zeno::vec3f>("uv1").push_back(zeno::vec3f(0.5,0.5, 0));
                prim->tris.attr<zeno::vec3f>("uv2").push_back(zeno::vec3f(0.5,0.5, 0));
            }
        }
    }
}

// makeXinxinVeryHappy
static auto primGetVal(PrimitiveObject *prim, size_t i) {
    std::map<std::string, std::variant<vec3f, float>> ret;
    prim->foreach_attr([&] (auto const &name, auto const &arr) {
        ret.emplace(name, arr[i]);
    });
    return ret;
}

// makeXinxinVeryHappy
static void primAppendVal(PrimitiveObject *prim, PrimitiveObject *primB, size_t i) {
    primB->foreach_attr([&] (auto const &name, auto const &arr) {
        using T = std::decay_t<decltype(arr[0])>;
        prim->attr<T>(name).push_back(arr[i]);
    });
}

ZENO_API void read_obj_file(
    std::vector<zeno::vec3f> &vertices,
    std::vector<zeno::vec3f> &uvs,
    std::vector<zeno::vec3f> &normals,
    std::vector<zeno::vec3i> &indices,
    const char *path
);

ZENO_API std::shared_ptr<PrimitiveObject>
primitive_merge(std::shared_ptr<ListObject> list, std::string tagAttr = {});

static void addIndividualPrimitive(PrimitiveObject* dst, const PrimitiveObject* src, size_t index)
        {
            for(auto key:src->attr_keys())
            {
                //using T = std::decay_t<decltype(src->attr(key)[0])>;
                if (key != "pos") {
                std::visit([index, &key, dst](auto &&src) {
                    using SrcT = std::remove_cv_t<std::remove_reference_t<decltype(src)>>;
                    std::get<SrcT>(dst->attr(key)).emplace_back(src[index]);
                }, src->attr(key));
                // dst->attr(key).emplace_back(src->attr(key)[index]);
                } else {
                    dst->attr<vec3f>(key).emplace_back(src->attr<vec3f>(key)[index]);
                }
            }
            dst->resize(dst->attr<zeno::vec3f>("pos").size());
        }

}
