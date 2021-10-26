#ifndef QDMOPENGLVIEWPORT_H
#define QDMOPENGLVIEWPORT_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLVertexArrayObject>
#include <memory>

class Renderable;

class QDMOpenGLViewport : public QOpenGLWidget, protected QOpenGLFunctions
{
    Q_OBJECT

    std::unique_ptr<QOpenGLVertexArrayObject> m_vao;
    std::vector<std::unique_ptr<Renderable>> m_renderables;

public:
    explicit QDMOpenGLViewport(QWidget *parent = nullptr);
    ~QDMOpenGLViewport();

    virtual QSize sizeHint() const override;
    virtual void initializeGL() override;
    virtual void resizeGL(int nx, int ny) override;
    virtual void paintGL() override;
};

#endif // QDMOPENGLVIEWPORT_H
