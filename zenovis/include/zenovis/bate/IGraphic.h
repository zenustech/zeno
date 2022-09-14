#pragma once

#include <memory>
#include <zeno/core/IObject.h>
#include <zeno/types/IObjectXMacro.h>
#include <zeno/utils/vec.h>
#include <glm/glm.hpp>


namespace zeno {
#define _ZENO_PER_XMACRO(TypeName, ...) \
struct TypeName;
ZENO_XMACRO_IObject(_ZENO_PER_XMACRO)
#undef _ZENO_PER_XMACRO
}

namespace zenovis {

struct Scene;

enum {
    INTERACT_X,
    INTERACT_Y,
    INTERACT_Z,
    INTERACT_XY,
    INTERACT_YZ,
    INTERACT_XZ,
    INTERACT_XYZ,
    INTERACT_NONE,
};

struct IGraphic {
    std::string nameid;
    std::shared_ptr<zeno::IObject> objholder;

    virtual ~IGraphic() = default;
};

struct IGraphicDraw : IGraphic {
    virtual void draw() = 0;
};

struct IGraphicHandler : IGraphicDraw {
    virtual int collisionTest(glm::vec3 ori, glm::vec3 dir) = 0;
    virtual void setCenter(zeno::vec3f center) = 0;
    virtual void setMode(int mode) = 0;
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

std::shared_ptr<IGraphicHandler> makeGraphicInteractTrans(Scene *scene, zeno::vec3f center) ;
} // namespace zenovis
