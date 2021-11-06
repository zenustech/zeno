#include "renderablefromany.h"
#include "renderablemesh.h"
#include "renderabletesttriangle.h"

ZENO_NAMESPACE_BEGIN

std::unique_ptr<Renderable> makeRenderableFromAny(ztd::any_ptr obj) {
    if (auto p = pointer_cast<types::Mesh>(obj)) {
        return makeRenderableMesh(p);
    } else {
        return makeRenderableTestTriangle(p);
    }
}

ZENO_NAMESPACE_END
