#include <zenovis/Scene.h>
#include <zeno/utils/vec.h>
#include <zeno/types/UserData.h>
#include <zenovis/bate/IGraphic.h>
#include <zeno/types/MaterialObject.h>

namespace zenovis {
namespace {

struct GraphicMaterial final : IGraphic {
    Scene *scene;

    explicit GraphicMaterial(Scene *scene_, zeno::MaterialObject *mtl) : scene(scene_) {
    }
};

}

void MakeGraphicVisitor::visit(zeno::MaterialObject *obj) {
     this->out_result = std::make_unique<GraphicMaterial>(this->in_scene, obj);
}

}
