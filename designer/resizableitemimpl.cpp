#include "framework.h"
#include "resizableitemimpl.h"
#include "resizecoreitem.h"
#include "zenonode.h"

using namespace std;


ResizableItemImpl::ResizableItemImpl(qreal x, qreal y, qreal w, qreal h, QGraphicsItem* parent)
    : QGraphicsObject(parent)
    , m_width(w)
    , m_height(h)
    , m_mouseHint(NO_DRAG)
    , m_borderitem(nullptr)
    , m_coreitem(nullptr)
    , m_showBdr(false)
{
    setFlags(ItemIsMovable | ItemIsSelectable);
    qreal offset = dragW / 2;
    setPos(x - offset, y - offset); //x,y is about inner rect pos.
    _adjustItemsPos();
}

void ResizableItemImpl::_adjustItemsPos()
{
	if (m_borderitem == nullptr)
		m_borderitem = new QGraphicsRectItem(0, 0, m_width, m_height, this);
	m_borderitem->setPos(QPointF(0, 0));

    qreal w = m_width, h = m_height;
    qreal offset = dragW / 2;
    QVector<QPointF> pts = {
        QPointF(-offset, -offset),
        QPointF(-offset, h / 2 - offset),
        QPointF(-offset, h - offset),

        QPointF(w / 2 - offset, -offset),
        QPointF(w / 2 - offset, h - offset),

        QPointF(w - offset, -offset),
        QPointF(w - offset, h / 2 - offset),
        QPointF(w - offset, h - offset)
    };

    if (m_dragPoints.size() != pts.size())
        m_dragPoints.resize(pts.size());
    for (int i = 0; i < pts.size(); i++)
    {
        if (m_dragPoints[i] == nullptr)
        {
            m_dragPoints[i] = new QGraphicsRectItem(QRectF(0, 0, dragW, dragW), this);
            m_dragPoints[i]->installSceneEventFilter(this);
            m_dragPoints[i]->setAcceptHoverEvents(true);
            m_dragPoints[i]->setFlag(QGraphicsItem::ItemIgnoresTransformations, false);
        }
        m_dragPoints[i]->setPos(pts[i]);
    }

    m_cursor_mapper.insert(make_pair(DRAG_LEFTTOP, Qt::SizeFDiagCursor));
	m_cursor_mapper.insert(make_pair(DRAG_LEFTMID, Qt::SizeHorCursor));
	m_cursor_mapper.insert(make_pair(DRAG_LEFTBOTTOM, Qt::SizeBDiagCursor));
	m_cursor_mapper.insert(make_pair(DRAG_MIDTOP, Qt::SizeVerCursor));
	m_cursor_mapper.insert(make_pair(DRAG_MIDBOTTOM, Qt::SizeVerCursor));
	m_cursor_mapper.insert(make_pair(DRAG_RIGHTTOP, Qt::SizeBDiagCursor));
	m_cursor_mapper.insert(make_pair(DRAG_RIGHTMID, Qt::SizeHorCursor));
	m_cursor_mapper.insert(make_pair(DRAG_RIGHTBOTTOM, Qt::SizeFDiagCursor));
	m_cursor_mapper.insert(make_pair(TRANSLATE, Qt::SizeAllCursor));
}

void ResizableItemImpl::setCoreItem(ResizableCoreItem* pItem)
{
    if (pItem == nullptr)
        return;

    m_coreitem = pItem;
    pItem->setZValue(-100);
    pItem->setParentItem(this);
    pItem->setPos(QPointF(0, 0));
    pItem->resize(QSizeF(m_width, m_height));
}

QRectF ResizableItemImpl::coreItemSceneRect()
{
    QPointF sceneOri = mapToScene(QPointF(0, 0));
    QRectF rc(sceneOri.x(), sceneOri.y(), width(), height());
    return rc;
}

void ResizableItemImpl::setCoreItemSceneRect(const QRectF& sceneRect)
{
    qreal offset = dragW / 2;
    QPointF topLeft = sceneRect.topLeft() - QPointF(offset, offset);
    setPos(topLeft);

    m_width = sceneRect.width();
    m_height = sceneRect.height();
    m_borderitem->setRect(QRectF(0, 0, m_width, m_height));
    if (m_coreitem)
        m_coreitem->resize(QSizeF(m_width, m_height));
    _adjustItemsPos();
    update();
}

void ResizableItemImpl::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
    if (option->state & (QStyle::State_Selected | QStyle::State_HasFocus))
    {
        QPen pen(QColor(21, 152, 255), borderW);
        pen.setJoinStyle(Qt::MiterJoin);
        QBrush brush(QColor(255, 255, 255));

        for (int i = 0; i < m_dragPoints.size(); i++)
        {
            m_dragPoints[i]->setPen(pen);
            m_dragPoints[i]->setBrush(brush);
            m_dragPoints[i]->show();
        }
        m_borderitem->setPen(pen);
        m_borderitem->setBrush(Qt::NoBrush);
        m_borderitem->show();
    }
    else
    {
        for (int i = 0; i < m_dragPoints.size(); i++)
        {
            m_dragPoints[i]->hide();
        }
 
        if (m_showBdr)
        {
			QPen pen(QColor(0, 0, 0), borderW);
			pen.setJoinStyle(Qt::MiterJoin);
			m_borderitem->setPen(pen);
			m_borderitem->setBrush(Qt::NoBrush);
            m_borderitem->show();
        }
        else
		{
			m_borderitem->hide();
        }
    }
}

QRectF ResizableItemImpl::boundingRect() const
{
    QRectF rc = childrenBoundingRect();
    return rc;
}

bool ResizableItemImpl::sceneEventFilter(QGraphicsItem* watched, QEvent* event)
{
    if (event->type() == QEvent::GraphicsSceneHoverEnter ||
        event->type() == QEvent::GraphicsSceneHoverMove)
    {
        QGraphicsSceneHoverEvent* e = static_cast<QGraphicsSceneHoverEvent*>(event);
        if (QGraphicsItem* item = getResizeHandleItem(e->scenePos()))
        {
            QGraphicsRectItem* pItem = qgraphicsitem_cast<QGraphicsRectItem*>(item);
            DRAG_ITEM mouseHint = (DRAG_ITEM)m_dragPoints.indexOf(pItem);
            setCursor(m_cursor_mapper[mouseHint]);
        }
    }
    else if (event->type() == QEvent::GraphicsSceneHoverLeave)
    {
        setCursor(Qt::ArrowCursor);
    }
    return _base::sceneEventFilter(watched, event);
}

QGraphicsItem* ResizableItemImpl::getResizeHandleItem(QPointF scenePos)
{
    qreal hitTextLength = 2 * dragW;
    //construct a "bounding rect" which will fully contain corner if mouse over it.
    QRectF brCorner(scenePos.x() - hitTextLength, scenePos.y() - hitTextLength, 2 * hitTextLength, 2 * hitTextLength);
    QList<QGraphicsItem*> items = scene()->items(brCorner, Qt::ContainsItemShape);
    if (items.isEmpty())
        return nullptr;
    else
        return items[0];
}

void ResizableItemImpl::showBorder(bool bShow)
{
    m_showBdr = bShow;
}

bool ResizableItemImpl::_enableMouseEvent()
{
    return m_coreitem || isSelected();
}

void ResizableItemImpl::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
    if (QApplication::keyboardModifiers() == Qt::ControlModifier)
        return;

	if (!_enableMouseEvent())
		return;

    QPointF itemScenePos = this->scenePos();
    QPointF scenePos = event->scenePos();
    QGraphicsItem* item = getResizeHandleItem(scenePos);
    if (item)
    {
        QRectF rc = QRectF(0, 0, m_width, m_height);
        qreal W = rc.width(), H = rc.height();
        if (item == m_dragPoints[DRAG_LEFTTOP])
        {
            m_mouseHint = DRAG_LEFTTOP;
            m_movescale_info.fixed_point = mapToScene(rc.bottomRight());
        }
        if (item == m_dragPoints[DRAG_LEFTBOTTOM])
        {
            m_mouseHint = DRAG_LEFTBOTTOM;
            m_movescale_info.fixed_point = mapToScene(rc.topRight());
        }
        if (item == m_dragPoints[DRAG_RIGHTTOP])
        {
            m_mouseHint = DRAG_RIGHTTOP;
            m_movescale_info.fixed_point = mapToScene(rc.bottomLeft());
        }
        if (item == m_dragPoints[DRAG_RIGHTBOTTOM])
        {
            m_mouseHint = DRAG_RIGHTBOTTOM;
            m_movescale_info.fixed_point = mapToScene(rc.topLeft());
        }

        if (item == m_dragPoints[DRAG_MIDTOP])
        {
            m_mouseHint = DRAG_MIDTOP;
            m_movescale_info.fixed_x = itemScenePos.x();
            m_movescale_info.fixed_y = itemScenePos.y() + H - 1;
        }
        if (item == m_dragPoints[DRAG_MIDBOTTOM])
        {
            m_mouseHint = DRAG_MIDBOTTOM;
            m_movescale_info.fixed_x = itemScenePos.x();
            m_movescale_info.fixed_y = itemScenePos.y();
        }
        if (item == m_dragPoints[DRAG_LEFTMID])
        {
            m_mouseHint = DRAG_LEFTMID;
            m_movescale_info.fixed_y = itemScenePos.y();
            m_movescale_info.fixed_x = itemScenePos.x() + W - 1;
        }
        if (item == m_dragPoints[DRAG_RIGHTMID])
        {
            m_mouseHint = DRAG_RIGHTMID;
            m_movescale_info.fixed_y = itemScenePos.y();
            m_movescale_info.fixed_x = itemScenePos.x();
        }
    }
    else
    {
        m_mouseHint = TRANSLATE;
        
    }
    m_movescale_info.old_width = m_width;
    m_movescale_info.old_height = m_height;
    _base::mousePressEvent(event);
}

void ResizableItemImpl::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
{
    if (QApplication::keyboardModifiers() == Qt::ControlModifier)
        return;

    if (!_enableMouseEvent())
        return;

    if (m_mouseHint == TRANSLATE)
    {
        _base::mouseMoveEvent(event);
    }
    else
    {
        QPointF scenePos = event->scenePos();
        qreal newWidth = 0, newHeight = 0;
        QPointF newTopLeft;

        switch (m_mouseHint)
        {
            case DRAG_LEFTTOP:
            case DRAG_RIGHTTOP:
            case DRAG_LEFTBOTTOM:
            case DRAG_RIGHTBOTTOM:
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
            case DRAG_LEFTMID:
            case DRAG_RIGHTMID:
            {
                qreal left = min(scenePos.x(), m_movescale_info.fixed_x);
                qreal right = max(scenePos.x(), m_movescale_info.fixed_x);
                qreal top = m_movescale_info.fixed_y;
                newWidth = right - left;
                newHeight = m_movescale_info.old_height;
                newTopLeft = QPointF(left, top);
                break;
            }
            case DRAG_MIDTOP:
            case DRAG_MIDBOTTOM:
            {
                qreal left = m_movescale_info.fixed_x;
                qreal top = min(scenePos.y(), m_movescale_info.fixed_y);
                qreal bottom = max(scenePos.y(), m_movescale_info.fixed_y);
                newWidth = m_movescale_info.old_width;
                newHeight = bottom - top;
                newTopLeft = QPointF(left, top);
                break;
            }
        }
        m_width = newWidth;
        m_height = newHeight;
        m_borderitem->setRect(QRectF(0, 0, newWidth, newHeight));
        if (m_coreitem)
            m_coreitem->resize(QSizeF(newWidth, newHeight));
        
        if (parentItem())
        {
            QPointF localPos = parentItem()->mapFromScene(newTopLeft);
            setPos(localPos);
        }
        else
        {
            setPos(newTopLeft);
        }

        _adjustItemsPos();
        update();
    }
    QPointF topLeft = mapToScene(0, 0);
    emit itemGeoChanged(QRectF(topLeft.x(), topLeft.y(), m_width, m_height));
}

QVariant ResizableItemImpl::itemChange(GraphicsItemChange change, const QVariant& value)
{
    if (change == QGraphicsItem::ItemSelectedHasChanged)
    {
        if (!isSelected())
            emit itemDeselected();
    }
    return value;
}

void ResizableItemImpl::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
    if (QApplication::keyboardModifiers() == Qt::ControlModifier)
        return;

    if (_enableMouseEvent())
	{
		m_mouseHint = NO_DRAG;
		_base::mouseReleaseEvent(event);
    }
}