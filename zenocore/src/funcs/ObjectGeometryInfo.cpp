#include <zeno/funcs/ObjectGeometryInfo.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveTools.h>
#include <zeno/types/UserData.h>
#include <zeno/para/parallel_reduce.h>

namespace zeno {

ZENO_API std::pair<vec3f, vec3f> primBoundingBox(PrimitiveObject* prim) {
    if (!prim->verts.size())
        return { {0, 0, 0}, {0, 0, 0} };
    return parallel_reduce_minmax(prim->verts.begin(), prim->verts.end());
}

ZENO_API bool objectGetBoundingBox(IObject *ptr, vec3f &bmin, vec3f &bmax) {
    auto ud = ptr->userData();
    if (ud->has("_bboxMin") && ud->has("_bboxMax")) {
        bmin = toVec3f(ud->get_vec3f("_bboxMin"));
        bmax = toVec3f(ud->get_vec3f("_bboxMax"));
        return true;
    } else {
        if (auto obj = dynamic_cast<PrimitiveObject *>(ptr)) {
            if (obj->verts.size() == 0) {
                 return false;
             }
            std::tie(bmin, bmax) = primBoundingBox(obj);
            ud->set_vec3f("_bboxMin", toAbiVec3f(bmin));
            ud->set_vec3f("_bboxMax", toAbiVec3f(bmax));
            return true;
        }
        else {
            return false;
        }
    }
}


ZENO_API bool objectGetFocusCenterRadius(IObject *ptr, vec3f &center, float &radius) {
    vec3f bmin, bmax;
    if (!objectGetBoundingBox(ptr, bmin, bmax))
        return false;
    auto delta = bmax - bmin;
    radius = std::max(std::max(delta[0], delta[1]), delta[2]) * 0.5f;
    center = (bmin + bmax) * 0.5f;
    return true;
}

}
