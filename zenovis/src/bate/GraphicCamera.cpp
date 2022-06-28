#include <zeno/utils/vec.h>
#include <zenovis/Camera.h>
#include <zenovis/Scene.h>
#include <zenovis/bate/IGraphic.h>
#include <zenovis/ShaderManager.h>
#include <zeno/types/CameraObject.h>
//#include <zeno/utils/scope_exit.h>

namespace zenovis {
namespace {

using zeno::vec3f;

struct GraphicCamera final : IGraphic {
    Scene *scene;

    explicit GraphicCamera(Scene *scene_, zeno::CameraObject *cam) : scene(scene_) {
        //zeno::scope_restore _1(scene->camera->m_nx);
        //zeno::scope_restore _2(scene->camera->m_ny);
        scene->camera->setCamera(cam->get());
    }
};

}

void MakeGraphicVisitor::visit(zeno::CameraObject *obj) {
     this->out_result = std::make_unique<GraphicCamera>(this->in_scene, obj);
}

}
