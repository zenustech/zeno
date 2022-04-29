#include <zeno/utils/vec.h>
#include <zeno/types/UserData.h>
#include <zenovis/zhxx/ZhxxCamera.h>
#include <zenovis/zhxx/ZhxxIGraphic.h>
#include <zenovis/zhxx/ZhxxScene.h>
#include <zenovis/zhxx/ZhxxLight.h>
#include <zenovis/ShaderManager.h>
#include <zenovis/opengl/buffer.h>
#include <zenovis/opengl/shader.h>
#include <zeno/types/LightObject.h>

namespace zenovis::zhxx {
namespace {

struct ZhxxGraphicLight final : ZhxxIGraphicLight {
    ZhxxScene *scene;
    zeno::LightData lightData;

    explicit ZhxxGraphicLight(ZhxxScene *scene_, zeno::LightObject *lit) : scene(scene_) {
        //auto nodeid = lit->userData().get("ident");
        lightData = static_cast<zeno::LightData const &>(*lit);
    }

    virtual void addToScene() override {
        scene->lightCluster->addLight(this->lightData);
    }
};

}

void ZhxxMakeGraphicVisitor::visit(zeno::LightObject *obj) {
     this->out_result = std::make_unique<ZhxxGraphicLight>(this->in_scene, obj);
}

}
