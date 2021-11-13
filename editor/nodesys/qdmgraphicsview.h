#ifndef QDMGRAPHICSVIEW_H
#define QDMGRAPHICSVIEW_H

#include <zeno/common.h>
#include <QGraphicsView>
#include "qdmgraphicsnode.h"
#include <QWidget>
#include <QPointF>

ZENO_NAMESPACE_BEGIN

class QDMGraphicsScene;

class QDMGraphicsView : public QGraphicsView
{
    Q_OBJECT

    QPointF m_lastMousePos;
    bool m_mouseDragging{false};

public:
    explicit QDMGraphicsView(QWidget *parent = nullptr);

    virtual void keyPressEvent(QKeyEvent *event) override;
    virtual void mousePressEvent(QMouseEvent *event) override;
    virtual void mouseMoveEvent(QMouseEvent *event) override;
    virtual void mouseReleaseEvent(QMouseEvent *event) override;
    virtual void wheelEvent(QWheelEvent *event) override;
    virtual QSize sizeHint() const override;
    QDMGraphicsScene *getScene() const;

    static constexpr float ZOOMFACTOR = 1.25f;

public slots:
    void addNodeByType(QString name);
    void invalidateNode(QDMGraphicsNode *node);
    void switchScene(QDMGraphicsScene *newScene);
    void forceUpdate();  // TODO: remove this!

signals:
    void nodeUpdated(QDMGraphicsNode *node, int type);
    void currentNodeChanged(QDMGraphicsNode *node);
};

ZENO_NAMESPACE_END

#endif // QDMGRAPHICSVIEW_H
