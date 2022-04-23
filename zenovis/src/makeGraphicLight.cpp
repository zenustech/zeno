#include <zeno/utils/vec.h>
#include <zenovis/Camera.h>
#include <zenovis/IGraphic.h>
#include <zenovis/Scene.h>
#include <zenovis/ShaderManager.h>
#include <zenovis/makeGraphic.h>
#include <zenovis/opengl/buffer.h>
#include <zenovis/opengl/shader.h>
#include <zeno/types/LightObject.h>
#include <zenovis/Light.h>

namespace zenovis {
namespace {

using opengl::Buffer;
using opengl::Program;
using zeno::vec3f;

struct GraphicLight : IGraphic {
    Scene *scene;

    std::unique_ptr<Buffer> vbo;
    size_t vertex_count;

    Program *prog;

    explicit GraphicLight(Scene *scene_, std::shared_ptr<zeno::LightObject> cam) : scene(scene_) {
        // TODO: implement modify scene->light
    }

    virtual void draw(bool reflect, bool depthPass) override {
    }

    virtual void drawShadow(Light *light) override {
    }
};

}

std::unique_ptr<IGraphic> makeGraphicLight(Scene *scene, std::shared_ptr<zeno::IObject> obj) {
    if (auto lit = std::dynamic_pointer_cast<zeno::LightObject>(obj))
        return std::make_unique<GraphicLight>(scene, std::move(lit));
    return nullptr;
}

}
