#include <zenovis/Scene.h>
#include <zeno/core/IObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/CameraObject.h>
#include <zeno/types/LightObject.h>
#include <zeno/types/MaterialObject.h>
#include <zeno/types/DummyObject.h>
#include <zeno/utils/cppdemangle.h>
#include <zeno/utils/log.h>
#include <zenovis/bate/IGraphic.h>

namespace zenovis {

void IGraphicHandler::setMode(int interact_mode) 
{ 
    mode = interact_mode; 
}

int IGraphicHandler::handleClick(glm::vec3 ori, glm::vec3 dir) 
{ 
    mode = collisionTest(ori, dir); 
    return mode; 
}

int IGraphicHandler::handleHover(glm::vec3 ori, glm::vec3 dir) 
{ 
    hover_mode = collisionTest(ori, dir);
    return hover_mode;
}

std::unique_ptr<IGraphic> makeGraphic(Scene *scene, zeno::IObject *obj) {
    MakeGraphicVisitor visitor;
    visitor.in_scene = scene;

    if (0) {
#define _ZENO_PER_XMACRO(TypeName, ...) \
    } else if (auto p = dynamic_cast<zeno::TypeName *>(obj)) { \
        visitor.visit(p);
ZENO_XMACRO_IObject(_ZENO_PER_XMACRO)
#undef _ZENO_PER_XMACRO
    }

    auto res = std::move(visitor.out_result);

    if (!res) {
        zeno::log_error("load_object: unexpected view object {}",
                        zeno::cppdemangle(typeid(*obj)));
    }

    //printf("%s\n", ext.c_str());
    //assert(0 && "bad file extension name");
    return res;
}

} // namespace zenovis
