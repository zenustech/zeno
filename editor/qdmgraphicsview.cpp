#include "qdmgraphicsview.h"
#include "qdmgraphicsscene.h"
#include <QMouseEvent>
#include <QPushButton>
#include <QScrollBar>

QSize QDMGraphicsView::sizeHint() const
{
    return QSize(1024, 640);
}

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
        // https://stackoverflow.com/questions/35865161/qt-graphic-scene-view-moving-around-with-mouse
        // https://forum.qt.io/topic/67636/scrolling-a-widget-by-hand-mouse/2
        m_lastMousePos = event->pos();
        m_mouseDragging = true;
        setCursor(Qt::CursorShape::OpenHandCursor);
        return;
    }

    if (event->button() == Qt::LeftButton) {
        setDragMode(QGraphicsView::RubberBandDrag);
    }

    QGraphicsView::mousePressEvent(event);
}

void QDMGraphicsView::mouseMoveEvent(QMouseEvent *event)
{
    if (m_mouseDragging) {
        auto delta = event->pos() - m_lastMousePos;
        horizontalScrollBar()->setValue(horizontalScrollBar()->value() - delta.x());
        verticalScrollBar()->setValue(verticalScrollBar()->value() - delta.y());
        m_lastMousePos = event->pos();
    }

    auto parentScene = static_cast<QDMGraphicsScene *>(scene());
    parentScene->cursorMoved();
    QGraphicsView::mouseMoveEvent(event);
}

void QDMGraphicsView::mouseReleaseEvent(QMouseEvent *event)
{
    if (event->button() == Qt::MiddleButton) {
        setDragMode(QGraphicsView::NoDrag);
        setCursor(Qt::CursorShape::ArrowCursor);
        m_mouseDragging = false;
        return;
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
