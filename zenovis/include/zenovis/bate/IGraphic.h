#pragma once

#include <memory>
#include <zeno/core/IObject.h>
#include <zeno/types/IObjectXMacro.h>
#include <zeno/utils/vec.h>


namespace zeno {
#define _ZENO_PER_XMACRO(TypeName, ...) \
struct TypeName;
ZENO_XMACRO_IObject(_ZENO_PER_XMACRO)
#undef _ZENO_PER_XMACRO
}

namespace zenovis {

struct Scene;

struct IGraphic {
    std::string nameid;
    std::shared_ptr<zeno::IObject> objholder;

    virtual ~IGraphic() = default;
};

struct IGraphicDraw : IGraphic {
    virtual void draw() = 0;
};

struct IGraphicInteractDraw : IGraphicDraw {
    virtual void setHovered(bool hovered) = 0;
};

struct MakeGraphicVisitor {
    Scene *in_scene{};
    std::unique_ptr<IGraphic> out_result;

#define _ZENO_PER_XMACRO(TypeName, ...) \
    void visit(zeno::TypeName *object);
ZENO_XMACRO_IObject(_ZENO_PER_XMACRO)
#undef _ZENO_PER_XMACRO
};

std::unique_ptr<IGraphic> makeGraphic(Scene *scene, zeno::IObject *obj);
std::unique_ptr<IGraphicDraw> makeGraphicAxis(Scene *scene);
std::unique_ptr<IGraphicDraw> makeGraphicGrid(Scene *scene);
std::unique_ptr<IGraphicDraw> makeGraphicSelectBox(Scene *scene);

std::unique_ptr<IGraphicInteractDraw> makeGraphicRing(Scene *scene, zeno::vec3f center);
std::unique_ptr<IGraphicInteractDraw> makeGraphicInteractAxis(Scene *scene, zeno::vec3f center) ;

std::unique_ptr<IGraphicDraw> makeGraphicInteractingAxis(Scene *scene, zeno::vec3f center, zeno::vec3i axis);
} // namespace zenovis
