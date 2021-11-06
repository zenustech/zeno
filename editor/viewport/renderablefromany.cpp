#include "renderablefromany.h"
#include "renderablemesh.h"

ZENO_NAMESPACE_BEGIN

std::unique_ptr<Renderable> makeRenderableFromAny(ztd::any_ptr obj) {
    if (auto p = pointer_cast<types::Mesh>(obj)) {
        return renderableFromMesh(p);
    } else {
        return nullptr;
    }
}

ZENO_NAMESPACE_END
