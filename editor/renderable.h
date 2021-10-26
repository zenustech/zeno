#ifndef RENDERABLE_H
#define RENDERABLE_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLVertexArrayObject>
#include <memory>

class Renderable
{
public:
    Renderable();
    virtual ~Renderable();

    virtual void render() = 0;
};

#endif // RENDERABLE_H
