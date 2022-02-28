#pragma once
#include <memory>

#include <zeno/utils/vec.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/PrimitiveObject.h>

namespace zeno {

// cihou zhxx
ZENO_API std::shared_ptr<PrimitiveObject>
primitive_merge(std::shared_ptr<ListObject> list);

ZENO_API void read_obj_file(
    std::vector<zeno::vec3f> &vertices,
    std::vector<zeno::vec3i> &indices,
    const char *path
);

}
