#pragma once

#include <zeno/utils/vec.h>
#include <zeno/core/IObject.h>

namespace zeno {

ZENO_API bool objectGetBoundingBox(IObject *ptr, vec3f &bmin, vec3f &bmax);
ZENO_API bool objectGetFocusCenterRadius(IObject *ptr, vec3f &center, float &radius);

}
