#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/LightObject.h>
#include <zeno/types/CameraObject.h>

namespace zeno {
namespace {

struct ForwardPass : INode {
    virtual void apply() override {
        auto objects = has_input("objects")
            ? get_input<ListObject>("objects")
            : std::make_shared<ListObject>();
        auto lights = has_input("lights")
            ? get_input<ListObject>("lights")
            : std::make_shared<ListObject>();
        auto materials = has_input("materials")
            ? get_input<ListObject>("materials")
            : std::make_shared<ListObject>();
        auto camera = get_input<CameraObject>();
    }
};

ZENO_DEFNODE(ForwardPass)({
    {
        {"list", "objects"},
        {"list", "lights"},
        {"list", "materials"},
    },
    {
        {"image", "image"},
    },
    {},
    {"pass"},
});

}
}
