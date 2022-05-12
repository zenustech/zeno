#include <zeno/funcs/ObjectGeometryInfo.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveTools.h>
#include <zeno/types/UserData.h>

namespace zeno {

ZENO_API bool objectGetBoundingBox(IObject *ptr, vec3f &bmin, vec3f &bmax) {
    auto &ud = ptr->userData();
    if (ud.has("_bboxMin") && ud.has("_bboxMax")) {
        bmin = ud.getLiterial<vec3f>("_bboxMin");
        bmax = ud.getLiterial<vec3f>("_bboxMax");
        return true;
    } else {
        if (auto obj = dynamic_cast<PrimitiveObject *>(ptr)) {
            std::tie(bmin, bmax) = primBoundingBox(obj);
            ud.setLiterial("_bboxMin", bmin);
            ud.setLiterial("_bboxMax", bmax);
        }
        return true;
    }
    return false;
}


ZENO_API bool objectGetFocusCenterRadius(IObject *ptr, vec3f &center, float &radius) {
    vec3f bmin, bmax;
    if (!objectGetBoundingBox(ptr, bmin, bmax))
        return false;
    auto delta = bmax - bmin;
    radius = std::max({delta[0], delta[1], delta[2]}) * 0.5f;
    center = (bmin + bmax) * 0.5f;
    return true;
}

}
