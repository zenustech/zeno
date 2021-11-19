#include "qdmgraphicsview.h"
#include "qdmgraphicsscene.h"
#include <zeno/zmt/log.h>
#include <QMouseEvent>
#include <QPushButton>
#include <QScrollBar>

ZENO_NAMESPACE_BEGIN

QSize QDMGraphicsView::sizeHint() const
{
    return QSize(1024, 640);
}

QDMGraphicsView::QDMGraphicsView(QWidget *parent)
    : QGraphicsView(parent)
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

void QDMGraphicsView::switchScene(QDMGraphicsScene *newScene)
{
    auto oldScene = getScene();
    if (oldScene == newScene)
        return;

    if (oldScene) {
        ZENO_DEBUG("switch from oldScene: {}", oldScene);
        oldScene->setCurrentNode(nullptr);
        disconnect(oldScene, SIGNAL(sceneUpdated()),
                   this, SIGNAL(sceneUpdated()));
        disconnect(oldScene, SIGNAL(sceneCreatedOrRemoved()),
                   this, SIGNAL(sceneCreatedOrRemoved()));
        disconnect(oldScene, SIGNAL(currentNodeChanged(QDMGraphicsNode*)),
                   this, SIGNAL(currentNodeChanged(QDMGraphicsNode*)));
    }

    ZENO_DEBUG("switch to newScene: {}", newScene);
    connect(newScene, SIGNAL(sceneUpdated()),
            this, SIGNAL(sceneUpdated()));
    connect(newScene, SIGNAL(sceneCreatedOrRemoved()),
            this, SIGNAL(sceneCreatedOrRemoved()));
    connect(newScene, SIGNAL(currentNodeChanged(QDMGraphicsNode*)),
            this, SIGNAL(currentNodeChanged(QDMGraphicsNode*)));

    setScene(newScene);
}

QDMGraphicsScene *QDMGraphicsView::getScene() const
{
    return static_cast<QDMGraphicsScene *>(scene());
}

void QDMGraphicsView::addNodeByType(QString name)
{
    getScene()->addNodeByType(name);
}

void QDMGraphicsView::addSubNetNode()
{
    getScene()->addSubNetNode();
}

void QDMGraphicsView::keyPressEvent(QKeyEvent *event)
{
    if (event->key() == Qt::Key_Delete) {
        getScene()->deletePressed();

    } else if (event->key() == Qt::Key_C && event->modifiers() & Qt::ControlModifier) {
        getScene()->copyPressed();

    } else if (event->key() == Qt::Key_V && event->modifiers() & Qt::ControlModifier) {
        getScene()->pastePressed();
    }

    QGraphicsView::keyPressEvent(event);
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

    getScene()->cursorMoved();

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

ZENO_NAMESPACE_END
