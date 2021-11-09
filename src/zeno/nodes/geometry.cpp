#include <zeno/dop/dop.h>


ZENO_NAMESPACE_BEGIN
namespace {


ZENO_DOP_INTERFACE(Transform, {{
    "geometry", "transform an object (by translation, scaling, and rotation)",
}, {
    {"object", "Mesh"},
    {"translate", "vec3f"},
    {"scaling", "vec3f"},
    {"rotation", "vec4f"},
}, {
    {"object", "Mesh"},
}});


ZENO_DOP_INTERFACE(Reduction, {{
    "geometry", "perform reduction on object (calculate bounding box, mass center, etc.)",
}, {
    {"object", "Mesh"},
    {"type", "string"},
}, {
    {"res1", "vec3f"},
    {"res2", "vec3f"},
}});


}
ZENO_NAMESPACE_END
