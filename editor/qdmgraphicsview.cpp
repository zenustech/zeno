#include "qdmgraphicsview.h"
#include "qdmgraphicsscene.h"
#include "qdmmouseeventeater.h"
#include <QMouseEvent>
#include <QPushButton>

QDMGraphicsView::QDMGraphicsView(QWidget *parent) : QGraphicsView(parent)
{
    setRenderHints(QPainter::Antialiasing
            | QPainter::SmoothPixmapTransform
            | QPainter::TextAntialiasing);
    setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    setContextMenuPolicy(Qt::NoContextMenu);
}

void QDMGraphicsView::mousePressEvent(QMouseEvent *event)
{
    if (event->button() == Qt::MiddleButton) {
        setDragMode(QGraphicsView::ScrollHandDrag);

        installEventFilter(new QDMMouseEventEater);
        QMouseEvent fakeEvent(event->type(),
                              event->position(), event->globalPosition(),
                              Qt::LeftButton, event->buttons() | Qt::LeftButton,
                              event->modifiers());
        QGraphicsView::mousePressEvent(&fakeEvent);
        return;
    }

    if (event->button() == Qt::LeftButton) {
        setDragMode(QGraphicsView::RubberBandDrag);
    }

    QGraphicsView::mousePressEvent(event);
}

void QDMGraphicsView::mouseMoveEvent(QMouseEvent *event)
{
    auto parentScene = static_cast<QDMGraphicsScene *>(scene());
    parentScene->cursorMoved();
    QGraphicsView::mouseMoveEvent(event);
}

void QDMGraphicsView::mouseReleaseEvent(QMouseEvent *event)
{
    if (event->button() == Qt::MiddleButton) {
        setDragMode(QGraphicsView::NoDrag);
    }

    setDragMode(QGraphicsView::NoDrag);
    QGraphicsView::mouseReleaseEvent(event);
}

void QDMGraphicsView::wheelEvent(QWheelEvent *event)
{
    float zoomFactor = 1;
    if (event->angleDelta().y() > 0)
        zoomFactor *= ZOOMFACTOR;
    else if (event->angleDelta().y() < 0)
        zoomFactor /= ZOOMFACTOR;

    scale(zoomFactor, zoomFactor);
}

void QDMGraphicsView::addNodeByName(QString name)
{
    auto parentScene = static_cast<QDMGraphicsScene *>(scene());
    parentScene->addNodeByName(name);
}
