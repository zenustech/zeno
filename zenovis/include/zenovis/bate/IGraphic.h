#pragma once

#include <memory>
#include <optional>
#include <vector>
#include <unordered_map>
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

enum {
    WORLD_COORD_SYS,
    LOCAL_COORD_SYS,
    VIEW_COORD_SYS
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
    int mode = INTERACT_NONE;
    int hover_mode = INTERACT_NONE;

    virtual int collisionTest(glm::vec3 ori, glm::vec3 dir) = 0;
    virtual void setCenter(zeno::vec3f center) = 0;
    virtual void setCoordSys(int coord_sys) = 0;
    virtual void resize(float scale) = 0;
    virtual std::optional<glm::vec3> getIntersect(glm::vec3 ori, glm::vec3 dir) = 0;
    virtual void setMode(int interact_mode);
    virtual int handleClick(glm::vec3 ori, glm::vec3 dir);
    virtual int handleHover(glm::vec3 ori, glm::vec3 dir);
};

struct IPicker : IGraphicDraw {
    virtual std::string getPicked(int x, int y) = 0;
    virtual std::string getPicked(int x0, int y0, int x1, int y1) = 0;
    virtual std::unordered_map<std::string , std::unordered_map<uint32_t, zeno::vec2f>> getPaintPicked(int x0, int y0, int x1, int y1) = 0;
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
std::unique_ptr<IGraphicDraw> makeGraphicPainterCursor(Scene *scene);

std::shared_ptr<IGraphicHandler> makeTransHandler(Scene *scene, zeno::vec3f center, float scale);
std::shared_ptr<IGraphicHandler> makeScaleHandler(Scene *scene, zeno::vec3f center, float scale);
std::shared_ptr<IGraphicHandler> makeRotateHandler(Scene *scene, zeno::vec3f center, float scale);

std::unique_ptr<IPicker> makeFrameBufferPicker(Scene *scene);
std::unique_ptr<IGraphicDraw> makePrimitiveHighlight(Scene* scene);
} // namespace zenovis
