#include <zenovis/bate/IGraphic.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/log.h>

namespace zenovis {
namespace {

struct GraphicString final : IGraphic {
    Scene *scene;

    explicit GraphicString(Scene *scene_, zeno::StringObject *str) : scene(scene_) {
        zeno::log_info("ToView got StringObject with content: {}", str->get());
    }
};

}

void MakeGraphicVisitor::visit(zeno::StringObject *obj) {
     this->out_result = std::make_unique<GraphicString>(this->in_scene, obj);
}

namespace {

struct GraphicNumeric final : IGraphic {
    Scene *scene;

    explicit GraphicNumeric(Scene *scene_, zeno::NumericObject *num) : scene(scene_) {
        std::visit([&] (auto &val) {
            zeno::log_info("ToView got NumericObject with content: {}", val);
        }, num->value);
    }
};

}

void MakeGraphicVisitor::visit(zeno::NumericObject *obj) {
     this->out_result = std::make_unique<GraphicNumeric>(this->in_scene, obj);
}

namespace {

struct GraphicDummy final : IGraphic {
    Scene *scene;

    explicit GraphicDummy(Scene *scene_, zeno::DummyObject *lit) : scene(scene_) {
    }
};

}

void MakeGraphicVisitor::visit(zeno::DummyObject *obj) {
     this->out_result = std::make_unique<GraphicDummy>(this->in_scene, obj);
}

namespace {

struct GraphicList final : IGraphic {
    Scene *scene;

    explicit GraphicList(Scene *scene_, zeno::ListObject *lst) : scene(scene_) {
        zeno::log_info("ToView got ListObject with size: {}", lst->arr.size());
    }
};

}

void MakeGraphicVisitor::visit(zeno::ListObject *obj) {
     this->out_result = std::make_unique<GraphicList>(this->in_scene, obj);
}

}
