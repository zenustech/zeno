#include <zeno/dop/dop.h>


ZENO_NAMESPACE_BEGIN
namespace {


ZENO_DOP_INTERFACE(Transform, {{
    "misc", "transform an object (by translation, scaling, and rotation)",
}, {
    {"object"},
    {"translate"},
    {"scaling"},
    {"rotation"},
}, {
    {"object"},
}});


}
ZENO_NAMESPACE_END
