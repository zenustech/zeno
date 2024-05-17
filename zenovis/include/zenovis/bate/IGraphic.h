#pragma once

#include <memory>
#include <optional>
#include <vector>
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

enum class OPERATION_MODE {
    INTERACT_X,
    INTERACT_Y,
    INTERACT_Z,
    INTERACT_XY,
    INTERACT_YZ,
    INTERACT_XZ,
    INTERACT_XYZ,
    INTERACT_NONE,
};

enum class COORD_SYS {
    WORLD_COORD_SYS,
    LOCAL_COORD_SYS,
//    VIEW_COORD_SYS
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
    OPERATION_MODE mode = OPERATION_MODE::INTERACT_NONE;
    OPERATION_MODE hover_mode = OPERATION_MODE::INTERACT_NONE;

    virtual OPERATION_MODE collisionTest(glm::vec3 ori, glm::vec3 dir) = 0;
    virtual void setCenter(zeno::vec3f center, zeno::vec3f localX, zeno::vec3f localY) = 0;
    virtual void resize(float scale) = 0;
    virtual std::optional<glm::vec3> getIntersect(glm::vec3 ori, glm::vec3 dir) = 0;
    virtual void setMode(OPERATION_MODE interact_mode);
    virtual OPERATION_MODE handleClick(glm::vec3 ori, glm::vec3 dir);
    virtual OPERATION_MODE handleHover(glm::vec3 ori, glm::vec3 dir);
};

struct IPicker : IGraphicDraw {
    virtual std::string getPicked(int x, int y) = 0;
    virtual std::string getPicked(int x0, int y0, int x1, int y1) = 0;
    virtual float getDepth(int x, int y) = 0;
    virtual void focus(const std::string& prim_name) = 0;
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

std::shared_ptr<IGraphicHandler> makeTransHandler(Scene *scene, zeno::vec3f center, zeno::vec3f localX_, zeno::vec3f localY_, float scale);
std::shared_ptr<IGraphicHandler> makeScaleHandler(Scene *scene, zeno::vec3f center, zeno::vec3f localX_, zeno::vec3f localY_, float scale);
std::shared_ptr<IGraphicHandler> makeRotateHandler(Scene *scene, zeno::vec3f center, zeno::vec3f localX_, zeno::vec3f localY_, float scale);

std::unique_ptr<IPicker> makeFrameBufferPicker(Scene *scene);
std::unique_ptr<IGraphicDraw> makePrimitiveHighlight(Scene* scene);
} // namespace zenovis
