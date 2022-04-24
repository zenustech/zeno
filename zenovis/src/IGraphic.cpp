#include <zenovis/Scene.h>
#include <zeno/core/IObject.h>
#include <zeno/utils/cppdemangle.h>
#include <zeno/utils/log.h>
#include <zenovis/IGraphic.h>

namespace zenovis {

std::unique_ptr<IGraphic> makeGraphic(Scene *scene, zeno::IObject *obj) {
    ToGraphicVisitor visitor;
    visitor.in_scene = scene;
    obj->accept(&visitor);
    auto res = std::move(visitor.out_result);

    if (!res)
        zeno::log_debug("load_object: unexpected view object {}",
                        zeno::cppdemangle(typeid(*obj)));

    //printf("%s\n", ext.c_str());
    //assert(0 && "bad file extension name");
    return res;
}

} // namespace zenovis
