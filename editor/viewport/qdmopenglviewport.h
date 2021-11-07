#ifndef QDMOPENGLVIEWPORT_H
#define QDMOPENGLVIEWPORT_H

#include <zeno/common.h>
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include "../nodesys/qdmgraphicsnode.h"
#include "cameradata.h"
#include <memory>
#include <map>

ZENO_NAMESPACE_BEGIN

class Renderable;

class QDMOpenGLViewport : public QOpenGLWidget, public QOpenGLFunctions
{
    Q_OBJECT

    std::map<QDMGraphicsNode *, std::unique_ptr<Renderable>> m_renderables;
    std::unique_ptr<CameraData> m_camera = std::make_unique<CameraData>();
    QPoint m_lastPos;

public:
    explicit QDMOpenGLViewport(QWidget *parent = nullptr);
    ~QDMOpenGLViewport();

    CameraData *getCamera() const;
    virtual QSize sizeHint() const override;
    virtual void initializeGL() override;
    virtual void resizeGL(int nx, int ny) override;
    virtual void paintGL() override;

    virtual void mousePressEvent(QMouseEvent *event) override;
    virtual void mouseMoveEvent(QMouseEvent *event) override;
    virtual void wheelEvent(QWheelEvent *event) override;

public slots:
    void updateNode(QDMGraphicsNode *node, int type);
};

ZENO_NAMESPACE_END

#endif // QDMOPENGLVIEWPORT_H
