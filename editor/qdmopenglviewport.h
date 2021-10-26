#ifndef QDMOPENGLVIEWPORT_H
#define QDMOPENGLVIEWPORT_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <memory>

ZENO_NAMESPACE_BEGIN

class Renderable;

class QDMOpenGLViewport : public QOpenGLWidget, public QOpenGLFunctions
{
    Q_OBJECT

    std::vector<std::unique_ptr<Renderable>> m_renderables;

public:
    explicit QDMOpenGLViewport(QWidget *parent = nullptr);
    ~QDMOpenGLViewport();

    virtual QSize sizeHint() const override;
    virtual void initializeGL() override;
    virtual void resizeGL(int nx, int ny) override;
    virtual void paintGL() override;
};

ZENO_NAMESPACE_END

#endif // QDMOPENGLVIEWPORT_H
