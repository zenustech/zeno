#include "renderable.h"
#include "renderablemesh.h"
#include "renderabletesttriangle.h"

ZENO_NAMESPACE_BEGIN

Renderable::Renderable() = default;
Renderable::~Renderable() = default;

std::unique_ptr<Renderable> makeRenderableFromAny(ztd::any_ptr obj) {
    if (auto p = obj.pointer_cast<types::Mesh>()) {
        return makeRenderableMesh(p);
    } else {
        return makeRenderableTestTriangle();
    }
}

ZENO_NAMESPACE_END
