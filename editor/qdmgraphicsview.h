#ifndef QDMGRAPHICSVIEW_H
#define QDMGRAPHICSVIEW_H

#include <QGraphicsView>
#include <QWidget>
#include <QPointF>

class QDMGraphicsView : public QGraphicsView
{
    QPointF m_lastMousePos;

public:
    explicit QDMGraphicsView(QWidget *parent = nullptr);

    virtual void mousePressEvent(QMouseEvent *event) override;
    virtual void mouseMoveEvent(QMouseEvent *event) override;
    virtual void mouseReleaseEvent(QMouseEvent *event) override;
    virtual void wheelEvent(QWheelEvent *event) override;

    static constexpr float ZOOMFACTOR = 1.25f;

private slots:
    void addNodeByName(QString name);
};

#endif // QDMGRAPHICSVIEW_H
