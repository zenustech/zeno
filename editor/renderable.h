#pragma once

#include "qdmopenglviewport.h"

class Renderable
{
public:
    Renderable();
    virtual ~Renderable();

    virtual void render(QDMOpenGLViewport *viewport) = 0;
};
