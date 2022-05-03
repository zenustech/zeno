#include <zeno/zeno.h>
#include <zeno/types/CameraObject.h>
#include <zeno/types/LightObject.h>

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
        camera->nx = get_input2<int>("nx");
        camera->ny = get_input2<int>("ny");
        set_output("camera", std::move(camera));
    }
};

ZENO_DEFNODE(MakeCamera)({
    {
        {"vec3f", "pos", "0,0,1"},
        {"vec3f", "up", "0,1,0"},
        {"vec3f", "view", "0,0,-1"},
        {"float", "dof", "-1"},
        {"float", "near", "0.1"},
        {"float", "far", "20000"},
        {"float", "fov", "45"},
        {"int", "nx", "1920"},
        {"int", "ny", "1080"},
    },
    {
        {"CameraObject", "camera"},
    },
    {},
    {"scenevis"},
});


struct MakeLight : INode {
    virtual void apply() override {
        auto light = std::make_unique<LightObject>();
        light->lightDir = normalize(get_input2<vec3f>("lightDir"));
        light->intensity = get_input2<float>("intensity");
        light->shadowTint = get_input2<vec3f>("shadowTint");
        light->lightHight = get_input2<float>("lightHight");
        light->shadowSoftness = get_input2<float>("shadowSoftness");
        light->lightColor = get_input2<vec3f>("lightColor");
        light->lightScale = get_input2<float>("lightScale");
        light->isEnabled = get_input2<bool>("isEnabled");
        set_output("light", std::move(light));
    }
};

ZENO_DEFNODE(MakeLight)({
    {
        {"vec3f", "lightDir", "1,1,0"},
        {"float", "intensity", "10"},
        {"vec3f", "shadowTint", "0.2,0.2,0.2"},
        {"float", "lightHight", "1000.0"},
        {"float", "shadowSoftness", "1.0"},
        {"vec3f", "lightColor", "1,1,1"},
        {"float", "lightScale", "1"},
        {"bool", "isEnabled", "1"},
    },
    {
        {"LightObject", "light"},
    },
    {},
    {"scenevis"},
});

};
