#include <zeno/dop/dop.h>


ZENO_NAMESPACE_BEGIN
namespace {


struct Transform : dop::OverloadingNode {
};

ZENO_DOP_DEFINE(Transform, {{
    "misc", "transform an object (by translation, scaling, and rotation)",
}, {
    {"object"},
    {"translation"},
}, {
    {"object"},
}});


}
ZENO_NAMESPACE_END
