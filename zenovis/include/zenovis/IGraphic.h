#pragma once

#include <memory>
#include <zeno/core/IObject.h>

namespace zenovis {

struct Light;
struct Scene;

struct IGraphic {
    virtual void draw(bool reflect, bool depthPass) = 0;
    virtual void drawShadow(Light *light) = 0;
    virtual bool hasMaterial() const { return false; }
    virtual ~IGraphic() = default;
};

struct ToGraphicVisitor : zeno::IObjectVisitor {
    Scene *in_scene{};
    std::unique_ptr<IGraphic> out_result;

#define _ZENO_PER_XMACRO(TypeName, ...) \
    virtual void visit(zeno::TypeName *object) override;
ZENO_XMACRO_IObject(_ZENO_PER_XMACRO)
#undef _ZENO_PER_XMACRO
};

std::unique_ptr<IGraphic> makeGraphic(Scene *scene, zeno::IObject *obj);
std::unique_ptr<IGraphic> makeGraphicAxis(Scene *scene);
std::unique_ptr<IGraphic> makeGraphicGrid(Scene *scene);

} // namespace zenovis
