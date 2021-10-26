#ifndef RENDERABLE_H
#define RENDERABLE_H

#include "qdmopenglviewport.h"

class Renderable
{
public:
    Renderable();
    virtual ~Renderable();

    virtual void render(QDMOpenGLViewport *viewport) = 0;
};

#endif // RENDERABLE_H
