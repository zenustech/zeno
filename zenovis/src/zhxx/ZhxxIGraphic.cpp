#include <zenovis/zhxx/ZhxxScene.h>
#include <zeno/core/IObject.h>
#include <zeno/utils/cppdemangle.h>
#include <zeno/utils/log.h>
#include <zenovis/zhxx/ZhxxIGraphic.h>

namespace zenovis::zhxx {

std::unique_ptr<ZhxxIGraphic> makeGraphic(ZhxxScene *scene, zeno::IObject *obj) {
    ZhxxMakeGraphicVisitor visitor;
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
