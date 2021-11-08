#pragma once

#include "qdmopenglviewport.h"
#include <zeno/ztd/any_ptr.h>

ZENO_NAMESPACE_BEGIN

class Renderable
{
public:
    Renderable();
    virtual ~Renderable();

    virtual void render(QDMOpenGLViewport *viewport) = 0;
};

std::unique_ptr<Renderable> makeRenderableFromAny(ztd::any_ptr obj);

ZENO_NAMESPACE_END
