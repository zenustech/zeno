#include <zeno/utils/vec.h>
#include <zeno/types/UserData.h>
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

struct GraphicLight : IGraphic {
    Scene *scene;

    explicit GraphicLight(Scene *scene_, zeno::LightObject *lit) : scene(scene_) {
        //auto nodeid = lit->userData().get("ident");
        scene->lightCluster->addLight(static_cast<zeno::LightData const &>(*lit));
        // TODO: implement modify scene->light
    }

    virtual void draw(bool reflect, bool depthPass) override {
    }

    virtual void drawShadow(Light *light) override {
    }
};

}

void ToGraphicVisitor::visit(zeno::LightObject *obj) {
     this->out_result = std::make_unique<GraphicLight>(this->in_scene, obj);
}

}
