#include "framework.h"
#include "resizableitemimpl.h"
#include "resizecoreitem.h"
#include "nodetemplate.h"
#include "designermainwin.h"
#include "nodeswidget.h"
#include "util.h"

using namespace std;


ResizableItemImpl::ResizableItemImpl(NODE_TYPE type, const QString& id, const QRectF& sceneRc, QGraphicsItem *parent)
    : QGraphicsObject(parent)
    , m_id(id)
    , m_type(type)
    , m_width(sceneRc.width())
    , m_height(sceneRc.height())
    , m_mouseHint(NO_DRAG)
    , m_borderitem(nullptr)
    , m_coreitem(nullptr)
    , m_showBdr(true)
    , m_bLocked(false)
    , m_content(NC_NONE)
{
    setFlags(ItemIsMovable | ItemIsSelectable | ItemSendsScenePositionChanges | ItemSendsGeometryChanges);
    QPointF topLeft = parent->mapFromScene(sceneRc.topLeft());
    setPos(topLeft);
    _adjustItemsPos();
    setVisible(true);
    resetZValue();
}

void ResizableItemImpl::resetZValue()
{
    if (m_type == NT_COMPONENT)
    {
        setZValue(m_bLocked ? ZVALUE_LOCKED_CP : ZVALUE_COMPONENT);
    }
    else if (m_type == NT_COMPONENT_AS_ELEMENT)
    {
        if (m_id.contains("backboard"))
        {
            setZValue(m_bLocked ? ZVALUE_BACKGROUND : ZVALUE_LOCKED_BG);
        } 
        else
        {
            setZValue(m_bLocked ? ZVALUE_ELEMENT : ZVALUE_LOCKED_ELEM);
        }
    }
    else if (m_type == NT_ELEMENT)
    {
        setZValue(m_bLocked ? ZVALUE_ELEMENT : ZVALUE_LOCKED_ELEM);
    }
}

void ResizableItemImpl::_adjustItemsPos()
{
	if (m_borderitem == nullptr)
		m_borderitem = new QGraphicsRectItem(0, 0, m_width, m_height, this);

	m_borderitem->setPos(QPointF(0, 0));
    m_borderitem->setRect(QRectF(0, 0, m_width, m_height));
    if (m_coreitem) {
        m_coreitem->resize(QSizeF(m_width, m_height));
        m_coreitem->setPos(QPointF(0, 0));
    }

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

    if (m_cursor_mapper.empty())
    {
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
}

void ResizableItemImpl::setCoreItem(ResizableCoreItem* pItem)
{
    if (pItem == nullptr)
        return;

    m_coreitem = pItem;
    pItem->setZValue(ZVALUE_CORE_ITEM);
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

void ResizableItemImpl::_sizeValidate(bool bTranslate)
{
    return;
    QGraphicsItem *parent = this->parentItem();
    QSizeF sz;

    ResizableItemImpl *parentItem = dynamic_cast<ResizableItemImpl *>(parent);

    if (parentItem) {
        sz = QSizeF(parentItem->width(), parentItem->height());
    } else {
        sz = sz = parent->boundingRect().size();
    }

    QPointF pos = this->pos();

    if (bTranslate)
    {
        QRectF rcParent = parent->boundingRect();

        qreal xp = pos.x(), yp = pos.y();

        int xRight = sz.width() - m_width - 1;
        int yBottom = sz.height() - m_height - 1;

        xp = std::min(std::max(0., xp), sz.width() - m_width - 1);
        yp = std::min(std::max(0., yp), sz.height() - m_height - 1);

        pos.setX(xp);
        pos.setY(yp);

        setPos(pos);
    }
    else
    {
        m_width = std::min(m_width, sz.width() - pos.x());
        m_height = std::min(m_height, sz.height() - pos.y());
    }
}

void ResizableItemImpl::_setPosition(QPointF pos)
{
    NodesWidget* pTab = getMainWindow()->getCurrentTab();
    SnapWay snap = pTab->getSnapWay();

    if (m_type == NT_COMPONENT || m_type == NT_COMPONENT_AS_ELEMENT || m_type == NT_ELEMENT && snap == SNAP_GRID)
    {
        int x = pos.x(), y = pos.y(), w = m_width, h = m_height;
        //TODO: it will be a large step when the grid becomes small, should adjust snap to pixel.
        int x_ = x - x % PIXELS_IN_CELL;
        int y_ = y - y % PIXELS_IN_CELL;
        int w_ = w - w % PIXELS_IN_CELL;
        int h_ = h - h % PIXELS_IN_CELL;
        setPos(x_, y_);
        m_width = w_;
        m_height = h_;
        _adjustItemsPos();
    }
    else if (m_type == NT_ELEMENT)
    {
        if (snap == SNAP_PIXEL)
        {
            int x = pos.x(), y = pos.y(), w = m_width, h = m_height;
            setPos(x, y);
            m_width = w;
            m_height = h;
            _adjustItemsPos();
        }
        else
        {
            setPos(pos);
        }
    }
}

void ResizableItemImpl::setCoreItemSceneRect(const QRectF& sceneRect)
{
    QPointF pos = parentItem()->mapFromScene(sceneRect.topLeft());

    QGraphicsItem *parent = this->parentItem();
    QSizeF sz = parent->boundingRect().size();

    setPos(pos);

    m_width = sceneRect.width();
    m_height = sceneRect.height();
    m_borderitem->setRect(QRectF(0, 0, m_width, m_height));
    if (m_coreitem)
    {
        m_coreitem->resize(QSizeF(m_width, m_height));
        m_coreitem->setPos(QPointF(0, 0));
    }
    _adjustItemsPos();
    update();
}

void ResizableItemImpl::_resetDragPoints()
{
    qreal factor = getMainWindow()->getCurrentTab()->factor();
    qreal w = m_width, h = m_height;

    qreal offset = (dragW / 2) / factor;
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

    for (int i = 0; i < pts.size(); i++)
    {
        m_dragPoints[i]->setRect(0, 0, dragW / factor, dragH / factor);
        m_dragPoints[i]->setPos(pts[i]);
    }
}

void ResizableItemImpl::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
    qreal factor = 1.0;
    bool bShowBdr = true;
    auto pTab = getMainWindow()->getCurrentTab();
    if (pTab)
    {
        factor = pTab->factor();
        bShowBdr = pTab->showBorder();
    }
    if (option->state & (QStyle::State_Selected | QStyle::State_HasFocus))
    {
        QPen pen(QColor(21, 152, 255), borderW / factor);
        pen.setJoinStyle(Qt::MiterJoin);
        QBrush brush(QColor(255, 255, 255));

        _resetDragPoints();
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
 
        if (bShowBdr)
        {
			QPen pen(QColor(0, 0, 0), borderW / factor);
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
    qreal factor = getMainWindow()->getCurrentTab()->factor();
    qreal hitTextLength = 2 * dragW / factor;
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
    return !m_bLocked;
}

void ResizableItemImpl::setLocked(bool bLock)
{
    m_bLocked = bLock;
    if (m_bLocked) {
        setFlag(ItemIsSelectable, false);
        m_showBdr = true;
        resetZValue();
    } else {
        setFlag(ItemIsSelectable, true);
        m_showBdr = true;
        resetZValue();
    }
}

bool ResizableItemImpl::isLocked() const
{
    return m_bLocked;
}

QString ResizableItemImpl::getId() const
{
    return m_id;
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
        _sizeValidate(true);
        _setPosition(pos());
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
            case NO_DRAG:
                return;
        }
        m_width = newWidth;
        m_height = newHeight;
        newTopLeft = parentItem()->mapFromScene(newTopLeft);

        _setPosition(newTopLeft);
        _sizeValidate(false);
        _adjustItemsPos();
        update();
    }

    QPointF topLeft = mapToScene(0, 0);
    emit gvItemGeoChanged(m_id, QRectF(topLeft.x(), topLeft.y(), m_width, m_height));
}

QVariant ResizableItemImpl::itemChange(GraphicsItemChange change, const QVariant& value)
{
    switch (change)
    {
        case QGraphicsItem::ItemSelectedHasChanged:
            emit gvItemSelectedChange(m_id, isSelected());
            break;

        /*
        case QGraphicsItem::ItemTransformHasChanged:
        case QGraphicsItem::ItemPositionHasChanged:
            QPointF topLeft = mapToScene(0, 0);
            emit gvItemGeoChanged(m_id, QRectF(topLeft.x(), topLeft.y(), m_width, m_height));
            break;
        */
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