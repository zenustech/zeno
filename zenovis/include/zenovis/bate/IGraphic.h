#pragma once

#include <memory>
#include <zeno/core/IObject.h>

namespace zenovis {

struct Scene;

struct IGraphic {
    virtual ~IGraphic() = default;
};

struct IGraphicDraw : IGraphic {
    virtual void draw() = 0;
};

struct MakeGraphicVisitor : zeno::IObjectVisitor {
    Scene *in_scene{};
    std::unique_ptr<IGraphic> out_result;

#define _ZENO_PER_XMACRO(TypeName, ...) \
    virtual void visit(zeno::TypeName *object) override;
ZENO_XMACRO_IObject(_ZENO_PER_XMACRO)
#undef _ZENO_PER_XMACRO
};

std::unique_ptr<IGraphic> makeGraphic(Scene *scene, zeno::IObject *obj);
std::unique_ptr<IGraphicDraw> makeGraphicAxis(Scene *scene);
std::unique_ptr<IGraphicDraw> makeGraphicGrid(Scene *scene);

} // namespace zenovis
