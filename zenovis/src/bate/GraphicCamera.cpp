#include <zeno/utils/vec.h>
#include <zenovis/Camera.h>
#include <zenovis/Scene.h>
#include <zenovis/bate/IGraphic.h>
#include <zenovis/ShaderManager.h>
#include <zeno/types/CameraObject.h>

namespace zenovis {
namespace {

using zeno::vec3f;

struct GraphicCamera final : IGraphic {
    Scene *scene;

    explicit GraphicCamera(Scene *scene_, zeno::CameraObject *cam) : scene(scene_) {
        scene->camera->setCamera(cam->get());
    }
};

}

void MakeGraphicVisitor::visit(zeno::CameraObject *obj) {
     this->out_result = std::make_unique<GraphicCamera>(this->in_scene, obj);
}

}
