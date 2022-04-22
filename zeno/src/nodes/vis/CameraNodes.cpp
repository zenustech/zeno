#include <zeno/zeno.h>
#include <zeno/types/CameraObject.h>

namespace zeno {

struct MakeCamera : INode {
    virtual void apply() override {
        auto camera = std::make_unique<CameraObject>();
        camera->pos = get_input2<vec3f>("pos");
        camera->up = get_input2<vec3f>("up");
        camera->view = get_input2<vec3f>("view");
        camera->dof = get_input2<float>("dof");
        camera->ffar = get_input2<float>("far");
        camera->fnear = get_input2<float>("near");
        camera->fov = get_input2<float>("fov");
        set_output("camera", std::move(camera));
    }
};

ZENO_DEFNODE(MakeCamera)({
    {
        {"vec3f", "pos", "0,0,1"},
        {"vec3f", "up", "0,1,0"},
        {"vec3f", "view", "0,0,-1"},
        {"float", "dof", "-1"},
        {"float", "far", "0.1"},
        {"float", "near", "20000"},
        {"float", "fov", "45"},
    },
    {
        {"CameraObject", "camera"},
    },
    {},
    {"scenevis"},
});

};
