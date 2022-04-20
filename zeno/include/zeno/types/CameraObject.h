#include <zeno/core/IObject.h>
#include <zeno/utils/vec.h>

namespace zeno {

struct CameraObject : IObjectClone<CameraObject> {
    vec3f pos{0, 0, -1};
    vec3f up{1, 1, 1};
    vec3f view{0, 0, 1};
    float fov{45.f};
    float dof{-1.f};
    float fnear{0.1f};
    float ffar{20000.f};
};

}
