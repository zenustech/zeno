#include <zenovis/Scene.h>
#include <zeno/utils/vec.h>
#include <zeno/types/UserData.h>
#include <zenovis/bate/IGraphic.h>
#include <zeno/types/LightObject.h>

namespace zenovis {
namespace {

struct GraphicDummy final : IGraphic {
    Scene *scene;
    zeno::LightData lightData;

    explicit GraphicDummy(Scene *scene_, zeno::DummyObject *lit) : scene(scene_) {
    }
};

}

void MakeGraphicVisitor::visit(zeno::DummyObject *obj) {
     this->out_result = std::make_unique<GraphicDummy>(this->in_scene, obj);
}

}
