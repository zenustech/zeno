#pragma once

#ifndef _MSC_VER
#warning "<zeno/utils/prim_ops.h> is deprecated, use <zeno/types/PrimitiveTools.h> instead"
#endif

<<<<<<< HEAD
namespace zeno {

// cihou zhxx
ZENO_API std::shared_ptr<PrimitiveObject>
primitive_merge(std::shared_ptr<ListObject> list);

ZENO_API void read_obj_file(
    std::vector<zeno::vec3f> &vertices,
    std::vector<zeno::vec3f> &uvs,
    std::vector<zeno::vec3f> &normals,
    std::vector<zeno::vec3i> &indices,
    const char *path
);

static void addIndividualPrimitive(PrimitiveObject* dst, const PrimitiveObject* src, size_t index)//cihou zhxx
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
=======
#include <zeno/types/PrimitiveTools.h>
>>>>>>> origin/master
