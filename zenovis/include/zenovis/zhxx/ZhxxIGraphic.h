#pragma once

#include <memory>
#include <zeno/core/IObject.h>

namespace zenovis::zhxx {

struct ZhxxLight;
struct ZhxxScene;

struct ZhxxIGraphic {
    virtual ~ZhxxIGraphic() = default;
};

struct ZhxxIGraphicDraw : ZhxxIGraphic {
    virtual void draw() = 0;
    virtual void drawShadow(ZhxxLight *light) = 0;
    virtual bool hasMaterial() const { return false; }
};

struct ZhxxIGraphicLight : ZhxxIGraphic {
    virtual void addToScene() = 0;
};

struct ZhxxMakeGraphicVisitor : zeno::IObjectVisitor {
    ZhxxScene *in_scene{};
    std::unique_ptr<ZhxxIGraphic> out_result;

#define _ZENO_PER_XMACRO(TypeName, ...) \
    virtual void visit(zeno::TypeName *object) override;
ZENO_XMACRO_IObject(_ZENO_PER_XMACRO)
#undef _ZENO_PER_XMACRO
};

std::unique_ptr<ZhxxIGraphic> makeGraphic(ZhxxScene *scene, zeno::IObject *obj);
std::unique_ptr<ZhxxIGraphicDraw> makeGraphicAxis(ZhxxScene *scene);
std::unique_ptr<ZhxxIGraphicDraw> makeGraphicGrid(ZhxxScene *scene);

} // namespace zenovis
