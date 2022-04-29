#include <zenovis/zhxx/ZhxxIGraphic.h>
#include <zenovis/zhxx/ZhxxScene.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/log.h>

namespace zenovis::zhxx {
namespace {

struct ZhxxGraphicString final : ZhxxIGraphic {
    ZhxxScene *scene;

    explicit ZhxxGraphicString(ZhxxScene *scene_, zeno::StringObject *str) : scene(scene_) {
        zeno::log_info("ToView got StringObject with content: {}", str->get());
    }
};

}

void ZhxxMakeGraphicVisitor::visit(zeno::StringObject *obj) {
     this->out_result = std::make_unique<ZhxxGraphicString>(this->in_scene, obj);
}

namespace {

struct ZhxxGraphicNumeric final : ZhxxIGraphic {
    ZhxxScene *scene;

    explicit ZhxxGraphicNumeric(ZhxxScene *scene_, zeno::NumericObject *num) : scene(scene_) {
        std::visit([&] (auto &val) {
            zeno::log_info("ToView got NumericObject with content: {}", val);
        }, num->value);
    }
};

}

void ZhxxMakeGraphicVisitor::visit(zeno::NumericObject *obj) {
     this->out_result = std::make_unique<ZhxxGraphicNumeric>(this->in_scene, obj);
}

}
