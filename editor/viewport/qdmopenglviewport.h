#ifndef QDMOPENGLVIEWPORT_H
#define QDMOPENGLVIEWPORT_H

#include <zeno/common.h>
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include "../nodesys/qdmgraphicsnode.h"
#include <memory>
#include <map>

ZENO_NAMESPACE_BEGIN

class Renderable;

class QDMOpenGLViewport : public QOpenGLWidget, public QOpenGLFunctions
{
    Q_OBJECT

    std::map<QDMGraphicsNode *, std::unique_ptr<Renderable>> m_renderables;

public:
    explicit QDMOpenGLViewport(QWidget *parent = nullptr);
    ~QDMOpenGLViewport();

    virtual QSize sizeHint() const override;
    virtual void initializeGL() override;
    virtual void resizeGL(int nx, int ny) override;
    virtual void paintGL() override;

public slots:
    void addNodeView(QDMGraphicsNode *node);
    void updateNodeView(QDMGraphicsNode *node);
    void removeNodeView(QDMGraphicsNode *node);
};

ZENO_NAMESPACE_END

#endif // QDMOPENGLVIEWPORT_H
