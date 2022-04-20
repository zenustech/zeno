#include <zeno/core/IObject.h>
#include <zeno/utils/vec.h>

namespace zeno {

struct LightObject : IObjectClone<LightObject> {
    vec3f pos{1, 1, 0};
    vec3f dir{-1, -1, 0};
    float intensity{100};
};

}
