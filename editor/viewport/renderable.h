#pragma once

#include "qdmopenglviewport.h"

ZENO_NAMESPACE_BEGIN

class Renderable
{
public:
    Renderable();
    virtual ~Renderable();

    virtual void render(QDMOpenGLViewport *viewport) = 0;
};

ZENO_NAMESPACE_END
