#include <zeno/core/IObject.h>
#include <zeno/utils/cppdemangle.h>
#include <zeno/utils/log.h>
#include <zenovis/IGraphic.h>

namespace zenovis {

extern std::unique_ptr<IGraphic>
makeGraphicPrimitive(std::shared_ptr<zeno::IObject> obj);

std::unique_ptr<IGraphic> makeGraphic(std::shared_ptr<zeno::IObject> obj) {
    if (auto ig = makeGraphicPrimitive(obj)) {
        zeno::log_trace("load_object: primitive");
        return ig;
    }

    zeno::log_debug("load_object: unexpected view object {}",
                    zeno::cppdemangle(typeid(*obj)));

    //printf("%s\n", ext.c_str());
    //assert(0 && "bad file extension name");
    return nullptr;
}

} // namespace zenovis
