#include "framework.h"
#include "resizerectitem.h"
#include "zenonode.h"

using namespace std;

ResizableRectItem::ResizableRectItem(qreal x, qreal y, qreal w, qreal h, QGraphicsItem* parent)
    : QGraphicsRectItem(0, 0, w, h, parent)
    , m_mouseHint(MOUSE_DONOTHING)
{
    m_ltcorner = new QGraphicsRectItem(QRectF(0, 0, dragW, dragH), this);
    m_lbcorner = new QGraphicsRectItem(QRectF(0, 0, dragW, dragH), this);
    m_rtcorner = new QGraphicsRectItem(QRectF(0, 0, dragW, dragH), this);
    m_rbcorner = new QGraphicsRectItem(QRectF(0, 0, dragW, dragH), this);
    setFlags(ItemIsMovable | ItemIsSelectable);

    setPos(x, y);
    _adjustItemsPos();
}

void ResizableRectItem::_adjustItemsPos()
{
    QPointF pos = this->pos();
    m_ltcorner->setPos(- dragW / 2., - dragH / 2.);
    m_lbcorner->setPos(-dragW / 2., rect().height() - dragH / 2.);
    m_rtcorner->setPos(rect().width() - dragW / 2., -dragH / 2.);
    m_rbcorner->setPos(rect().width() - dragW / 2., rect().height() - dragH / 2.);
}

void ResizableRectItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
    if (option->state & (QStyle::State_Selected | QStyle::State_HasFocus))
    {
        QPen pen(QColor(21, 152, 255), borderW);
        pen.setJoinStyle(Qt::MiterJoin);
        QBrush brush(QColor(255, 255, 255));

        m_ltcorner->setPen(pen);
        m_ltcorner->setBrush(brush);

        m_lbcorner->setPen(pen);
        m_lbcorner->setBrush(brush);

        m_rtcorner->setPen(pen);
        m_rtcorner->setBrush(brush);

        m_rbcorner->setPen(pen);
        m_rbcorner->setBrush(brush);

        m_ltcorner->show();
        m_lbcorner->show();
        m_rtcorner->show();
        m_rbcorner->show();

        setPen(pen);
        setBrush(Qt::NoBrush);
    }
    else
    {
        QPen pen(QColor(0, 0, 0), borderW);
        pen.setJoinStyle(Qt::MiterJoin);

        m_ltcorner->hide();
        m_lbcorner->hide();
        m_rtcorner->hide();
        m_rbcorner->hide();

        setPen(pen);
        setBrush(Qt::NoBrush);
    }

    painter->setPen(pen());
    painter->setBrush(brush());
    painter->drawRect(rect());
}

QRectF ResizableRectItem::boundingRect() const
{
    QRectF rc = _base::boundingRect();
    return rc.adjusted(-dragW / 2., -dragH / 2, dragW / 2, dragH / 2);
}

bool ResizableRectItem::sceneEventFilter(QGraphicsItem* watched, QEvent* event)
{
    return _base::sceneEventFilter(watched, event);
}

void ResizableRectItem::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
    QPointF scenePos = event->scenePos();

    //construct a "bounding rect" which will fully contain corner if mouse over it.
    QRectF brCorner(scenePos.x() - dragW, scenePos.y() - dragH, 2 * dragW, 2 * dragH);
    QList<QGraphicsItem*> items = scene()->items(brCorner, Qt::ContainsItemShape);
    if (!items.isEmpty())
    {
        int W = rect().width(), H = rect().height();
        if (items[0] == m_ltcorner)
        {
            m_mouseHint = SCALE_LEFT_TOP;
            m_movescale_info.fixed_point = mapToScene(rect().bottomRight());
        }
        if (items[0] == m_lbcorner)
        {
            m_mouseHint = SCALE_LEFT_BOTTOM;
            m_movescale_info.fixed_point = mapToScene(rect().topRight());
        }
        if (items[0] == m_rtcorner)
        {
            m_mouseHint = SCALE_RIGHT_TOP;
            m_movescale_info.fixed_point = mapToScene(rect().bottomLeft());
        }
        if (items[0] == m_rbcorner)
        {
            m_mouseHint = SCALE_RIGHT_BOTTOM;
            m_movescale_info.fixed_point = mapToScene(rect().topLeft());
        }
        m_movescale_info.old_width = rect().width();
        m_movescale_info.old_height = rect().height();
        return;
    }
    else
    {
        m_mouseHint = TRANSLATE;
    }

    _base::mousePressEvent(event);
}

void ResizableRectItem::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
{
    if (m_mouseHint == TRANSLATE)
    {
        _base::mouseMoveEvent(event);
    }
    else
    {
        QPointF pos = event->pos();
        QPointF scenePos = event->scenePos();
        qreal newWidth = 0, newHeight = 0;
        QPointF newTopLeft;

        switch (m_mouseHint)
        {
            case SCALE_LEFT_TOP:
            case SCALE_RIGHT_TOP:
            case SCALE_LEFT_BOTTOM:
            case SCALE_RIGHT_BOTTOM:
            {
                //fixed_bottomright
                qreal left = min(scenePos.x(), m_movescale_info.fixed_point.x());
                qreal right = max(scenePos.x(), m_movescale_info.fixed_point.x());
                qreal top = min(scenePos.y(), m_movescale_info.fixed_point.y());
                qreal bottom = max(scenePos.y(), m_movescale_info.fixed_point.y());
                newWidth = right - left;
                newHeight = bottom - top;
                newTopLeft = QPointF(left, top);
                break;
            }
        }

        setRect(QRectF(0, 0, newWidth, newHeight));
        setPos(newTopLeft);
        _adjustItemsPos();
        update();
    }
}

void ResizableRectItem::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
    m_mouseHint = MOUSE_DONOTHING;
    _base::mouseReleaseEvent(event);
}