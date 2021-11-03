#include <zeno/dop/dop.h>


ZENO_NAMESPACE_BEGIN
namespace {


ZENO_DOP_INTERFACE(Transform, {{
    "geometry", "transform an object (by translation, scaling, and rotation)",
}, {
    {"object"},
    {"translate"},
    {"scaling"},
    {"rotation"},
}, {
    {"object"},
}});


ZENO_DOP_INTERFACE(Reduction, {{
    "geometry", "perform reduction on object (calculate bounding box, mass center, etc.)",
}, {
    {"object"},
    {"type"},
}, {
    {"object"},
}});


}
ZENO_NAMESPACE_END
