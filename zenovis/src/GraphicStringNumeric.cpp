#include <zenovis/IGraphic.h>
#include <zenovis/makeGraphic.h>
#include <zeno/types/StringObject.h>

namespace zenovis {
namespace {

struct GraphicString : IGraphic {
    Scene *scene;

    explicit GraphicString(Scene *scene_, zeno::StringObject *str) : scene(scene_) {
        zeno::log_info("ToView got StringObject with content: {}", str->get());
    }

    virtual void draw(bool reflect, bool depthPass) override {
    }

    virtual void drawShadow(Light *light) override {
    }
};

}

void ToGraphicVisitor::visit(zeno::StringObject *obj) {
     this->out_result = std::make_unique<GraphicString>(this->in_scene, obj);
}

struct GraphicNumeric : IGraphic {
    Scene *scene;

    explicit GraphicNumeric(Scene *scene_, zeno::NumericObject *num) : scene(scene_) {
        std::visit([&] (auto &val) {
            zeno::log_info("ToView got NumericObject with content: {}", val);
        }, num->value);
    }

    virtual void draw(bool reflect, bool depthPass) override {
    }

    virtual void drawShadow(Light *light) override {
    }
};

}

void ToGraphicVisitor::visit(zeno::NumericObject *obj) {
     this->out_result = std::make_unique<GraphicNumeric>(this->in_scene, obj);
}

}
