#pragma once

#include <zeno/utils/vec.h>
#include <zeno/core/IObject.h>
#include <zeno/types/PrimitiveObject.h>

namespace zeno {

ZENO_API std::pair<vec3f, vec3f> primBoundingBox(PrimitiveObject* prim);
ZENO_API bool objectGetBoundingBox(IObject *ptr, vec3f &bmin, vec3f &bmax);
ZENO_API bool objectGetFocusCenterRadius(IObject *ptr, vec3f &center, float &radius);

}
